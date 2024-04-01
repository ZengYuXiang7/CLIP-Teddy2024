# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import torch
import argparse

from tqdm import *

from utils.config import get_config
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed

global log

torch.set_default_dtype(torch.float32)


class experiment:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        string = args.path + '/' + args.dataset + 'Matrix' + '.txt'
        tensor = np.loadtxt(open(string, 'rb'))
        return tensor

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        return data



# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = self.get_train_valid_test_dataset(self.data, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor, self.test_tensor, exper_type, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

    def get_dataset(self, train_tensor, valid_tensor, test_tensor, exper_type, args):
        return (
            TensorDataset(train_tensor, exper_type, args),
            TensorDataset(valid_tensor, exper_type, args),
            TensorDataset(test_tensor, exper_type, args)
        )

    def get_train_valid_test_dataset(self, tensor, args):
        quantile = np.percentile(tensor, q=100)
        # tensor[tensor > quantile] = 0
        tensor = tensor / (np.max(tensor))  # 如果数据有分布偏移，记得处理数据
        trainsize = int(np.prod(tensor.size) * args.density)
        validsize = int((np.prod(tensor.size)) * 0.05)
        rowIdx, colIdx = tensor.nonzero()
        p = np.random.permutation(len(rowIdx))
        rowIdx, colIdx = rowIdx[p], colIdx[p]
        trainRowIndex = rowIdx[:trainsize]
        trainColIndex = colIdx[:trainsize]
        traintensor = np.zeros_like(tensor)
        traintensor[trainRowIndex, trainColIndex] = tensor[trainRowIndex, trainColIndex]
        validStart = trainsize
        validRowIndex = rowIdx[validStart:validStart + validsize]
        validColIndex = colIdx[validStart:validStart + validsize]
        validtensor = np.zeros_like(tensor)
        validtensor[validRowIndex, validColIndex] = tensor[validRowIndex, validColIndex]
        testStart = validStart + validsize
        testRowIndex = rowIdx[testStart:]
        testColIndex = colIdx[testStart:]
        testtensor = np.zeros_like(tensor)
        testtensor[testRowIndex, testColIndex] = tensor[testRowIndex, testColIndex]
        return traintensor, validtensor, testtensor, quantile


class TensorDataset(torch.utils.data.Dataset):

    def __init__(self, tensor, exper, args):
        self.tensor = tensor
        self.indices = self.get_pytorch_index(tensor)

    def __getitem__(self, idx):
        output = self.indices[idx, :-1]  # 去掉最后一列
        inputs = tuple(torch.as_tensor(output[i]).long() for i in range(output.shape[0]))
        value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
        return inputs, value

    def __len__(self):
        return self.indices.shape[0]

    def get_pytorch_index(self, data):
        userIdx, servIdx = data.nonzero()
        values = data[userIdx, servIdx]
        idx = torch.as_tensor(np.vstack([userIdx, servIdx, values]).T)
        return idx


class Model(torch.nn.Module):
    def __init__(self, train_graph, args):
        super(Model, self).__init__()
        self.args = args
        self.user_num = args.user_num
        self.serv_num = args.serv_num
        self.hyper_graph = HyperModel(train_graph, args)

    def forward(self, inputs, test=False):
        userIdx, servIdx = inputs
        estimated = self.hyper_graph(userIdx, servIdx)
        # print(estimated.shape)
        return estimated.flatten()

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in dataModule.train_loader:
            inputs, value = train_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            pred = self.forward(inputs, False)
            loss = self.loss_function(pred, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        writeIdx = 0
        val_loss = 0.
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        for valid_Batch in (dataModule.valid_loader):
            inputs, value = valid_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            pred = self.forward(inputs)
            val_loss += self.loss_function(pred, value)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        self.scheduler.step(val_loss)
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        for test_Batch in (dataModule.test_loader):
            inputs, value = test_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            pred = self.forward(inputs)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        return test_error


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule.train_tensor, args)
    monitor = EarlyStopping(args)

    # Setup training tool
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value
    train_time = []
    # for epoch in trange(args.epochs, disable=not args.program_test):
    for epoch in range(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track_one_epoch(epoch, model, valid_error)
        train_time.append(time_cost)
        log.show_epoch_error(runId, epoch, epoch_loss, valid_error, train_time)
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule)
    log.show_test_error(runId, monitor, results, sum_time)
    return {
        'MAE': results["MAE"],
        'RMSE': results["RMSE"],
        'NMAE': results["NMAE"],
        'NRMSE': results["NRMSE"],
        'TIME': sum_time,
    }, results['Acc']


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)
    for runId in range(args.rounds):
        runHash = int(time.time())
        results, acc = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])
        for key, item in zip(['Acc1', 'Acc5', 'Acc10'], [0, 1, 2]):
            metrics[key].append(acc[item])
    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    if args.record:
        log.save_result(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20 + '\n')
    return metrics


if __name__ == '__main__':
    args = get_config()
    set_settings(args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)