# coding : utf-8
# Author : yuxiang Zeng
import collections
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import argparse

from tqdm import *

from eval_recall import get_performance
from run_main import high_speed_retreive
from utils.config import get_config
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

global log

torch.set_default_dtype(torch.float32)


class experiment:
    def __init__(self, args):
        self.args = args
        _, self.preprocess = load_from_name("ViT-B-16", download_root='../BigDataSource/')


    @staticmethod
    def load_data(args):
        # 直接获取张量
        all_image = os.listdir('../BigDataSource/Teddy2024/附件1/ImageData')
        all_text = pd.read_csv('../BigDataSource/Teddy2024/附件1/ImageWordData.csv').to_numpy()[:, 1]
        all_image_address = []
        all_raw_text = []
        for i in range(len(all_image)):
            image_address = '../BigDataSource/Teddy2024/附件1/ImageData/'
            file_name = image_address + all_image[i]
            all_image_address.append(file_name)
            all_raw_text.append(all_text[i])
        log('Raw tensor loaded!')
        all_image_address, all_raw_text = np.array(all_image_address), np.array(all_raw_text)
        return all_image_address, all_raw_text

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        return data



# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.all_image_address, self.all_raw_text = exper_type.load_data(args)
        self.train_image_tensor, self.train_text_tensor, self.valid_image_tensor, self.valid_text_tensor, self.test_image_tensor, self.test_text_tensor = self.get_train_valid_test_dataset(self.all_image_address, self.all_raw_text, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_image_tensor, self.train_text_tensor, self.valid_image_tensor, self.valid_text_tensor, self.test_image_tensor, self.test_text_tensor, exper_type.preprocess, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

    def get_dataset(self, train_image_tensor, train_text_tensor, valid_image_tensor, valid_text_tensor, test_image_tensor, test_text_tensor, preprocess, args):
        return (
            TensorDataset(train_image_tensor, train_text_tensor, preprocess, args),
            TensorDataset(valid_image_tensor, valid_text_tensor, preprocess, args),
            TensorDataset(test_image_tensor, test_text_tensor, preprocess, args)
        )

    def get_train_valid_test_dataset(self, image_tensor, text_tensor, args):
        np.random.shuffle(image_tensor)

        train_size = int(len(image_tensor) * args.density)
        valid_size = int(len(image_tensor) * 0.01)

        train_image_tensor = image_tensor[:train_size]
        train_text_tensor = text_tensor[:train_size]

        valid_image_tensor = image_tensor[train_size:train_size + valid_size]
        valid_text_tensor = text_tensor[train_size:train_size + valid_size]

        test_image_tensor = image_tensor[train_size + valid_size:]
        test_text_tensor = image_tensor[train_size + valid_size:]

        return train_image_tensor, train_text_tensor, valid_image_tensor, valid_text_tensor, test_image_tensor, test_text_tensor


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, all_image_address, all_raw_text, preprocess, args):
        self.args = args
        self.all_image_address = all_image_address
        self.all_raw_text = all_raw_text
        self.preprocess = preprocess

    def __len__(self):
        return len(self.all_image_address)

    def __getitem__(self, idx):
        file_name = self.all_image_address[idx]
        raw_text = self.all_raw_text[idx]
        raw_image = Image.open(file_name)
        image_tensor = self.preprocess(raw_image)
        text_tensor = clip.tokenize(raw_text).squeeze(0)
        # print(text_tensor.shape)
        return image_tensor, text_tensor


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.retrieve_method = args.retrieve_method
        self.model, self.preprocess = load_from_name("ViT-B-16", download_root='../BigDataSource/')

    def forward(self, image_tensor, text_tensor):
        image_features, text_features, _ = self.model.forward(image_tensor, text_tensor)
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        ground_truth = torch.arange(len(image_tensor), dtype=torch.long)
        return logits_per_image, logits_per_text, ground_truth

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_img = get_loss_function(args).to(args.device)
        self.loss_txt = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

    def train_one_epoch(self, dataModule):
        loss = None
        all_image_features, all_text_features = [], []
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader):
            image_tensor, text_tensor = train_Batch
            image_tensor, text_tensor = image_tensor.to(self.args.device), text_tensor.to(self.args.device)
            logits_per_image, logits_per_text, ground_truth = self.forward(image_tensor, text_tensor)
            ground_truth = ground_truth.to(self.args.device)
            loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            del logits_per_image, logits_per_text, ground_truth
            image_feature, text_feature = self.model.encode_image(image_tensor).detach(), self.model.encode_text(text_tensor).detach()
            all_image_features.append(image_feature)
            all_text_features.append(text_feature)

        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        self.scheduler.step(loss)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        print(all_image_features.shape)
        print(all_text_features.shape)
        print(all_image_features)
        print(all_text_features)
        self.args.log('-' * 80)
        all_pred_rank = high_speed_retreive(all_image_features, all_text_features, self.retrieve_method, 100)
        all_real_rank = np.array([[i] for i in range(len(dataModule.train_loader.dataset))])
        print(all_pred_rank)
        for k in [5, 10, 20, 50]:
            valid_error = get_performance(all_real_rank, all_pred_rank, k)
            string = f"Recall@{k}="
            self.args.log(f"{string:15s}{valid_error['recalls'].mean():.6f}")
            string = f"NDCG@{k}="
            self.args.log(f"{string:15s}{valid_error['ndcgs'].mean():.6f}")

        return loss, valid_error, t2 - t1


    def valid_one_epoch(self, dataModule):
        self.eval()
        val_loss = 0.
        all_image_features, all_text_features = [], []
        for valid_Batch in tqdm(dataModule.valid_loader):
            image_tensor, text_tensor = valid_Batch
            # 梯度折损记录
            image_tensor, text_tensor = image_tensor.to(self.args.device), text_tensor.to(self.args.device)
            logits_per_image, logits_per_text, ground_truth = self.forward(image_tensor, text_tensor)
            ground_truth = ground_truth.to(self.args.device)
            val_loss += (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2

            image_feature, text_feature = self.model.encode_image(image_tensor), self.model.encode_text(text_tensor)
            all_image_features.append(image_feature)
            all_text_features.append(text_feature)

        self.scheduler.step(val_loss)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        print(all_image_features.shape)
        print(all_text_features.shape)
        print(all_image_features)
        print(all_text_features)
        self.args.log('-' * 80)
        all_pred_rank = high_speed_retreive(all_image_features, all_text_features, self.retrieve_method, 100)
        all_real_rank = np.array([[i] for i in range(len(dataModule.valid_loader.dataset))])
        print(all_pred_rank)
        for k in [5, 10, 20, 50]:
            valid_error = get_performance(all_real_rank, all_pred_rank, k)
            string = f"Recall@{k}="
            self.args.log(f"{string:15s}{valid_error['recalls'].mean():.6f}")
            string = f"NDCG@{k}="
            self.args.log(f"{string:15s}{valid_error['ndcgs'].mean():.6f}")
        return valid_error

    def test_one_epoch(self, dataModule):
        all_image_features, all_text_features = [], []
        for test_Batch in tqdm(dataModule.test_loader):
            image_tensor, text_tensor = test_Batch
            image_feature, text_feature = self.model.encode_image(image_tensor), self.model.encode_text(text_tensor)
            all_image_features.append(image_feature)
            all_text_features.append(text_feature)

        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        all_pred_rank = high_speed_retreive(all_image_features, all_text_features, self.retrieve_method, 100)
        all_real_rank = np.array([[i] for i in range(len(dataModule.valid_loader.dataset))])

        self.log('-' * 80)
        for k in [5, 10, 20, 50]:
            test_error = get_performance(all_real_rank, all_pred_rank, k)
            string = f"Recall@{k}="
            self.args.log(f"{string:15s}{test_error['recalls'].mean():.6f}")
            string = f"NDCG@{k}="
            self.args.log(f"{string:15s}{test_error['ndcgs'].mean():.6f}")
        return test_error


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(args)
    monitor = EarlyStopping(args)

    # Setup training tool
    model.setup_optimizer(args)
    train_time = []
    # for epoch in trange(args.epochs, disable=not args.program_test):
    for epoch in range(args.epochs):
        epoch_loss, valid_error, time_cost = model.train_one_epoch(datamodule)
        # valid_error = model.valid_one_epoch(datamodule)
        monitor.track_one_epoch(epoch, model, valid_error)
        train_time.append(time_cost)
        log.show_epoch_error(runId, epoch, epoch_loss, valid_error, train_time)
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    torch.save(model.state_dict(), f"./model_epoch.pth")
    sum_time = sum(train_time[: monitor.best_epoch])
    epoch_loss, results, time_cost = model.train_one_epoch(datamodule)
    # results = model.test_one_epoch(datamodule)
    log.show_test_error(runId, monitor, results, sum_time)
    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)
    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])
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