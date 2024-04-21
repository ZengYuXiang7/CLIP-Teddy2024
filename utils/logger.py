# coding : utf-8
# Author : yuxiang Zeng
# 日志
import logging
import pickle
import sys
import time
import numpy as np
import platform

from utils.utils import makedir

class Logger:


    def __init__(self, args):
        self.args = args
        makedir('./results/log/')
        # 日志记录文件
        self.filename = f'{args.logger}__{args.model}_{args.dataset}_{args.density}_{args.dimension}'
        if args.experiment:
            ts = time.asctime().replace(' ', '_').replace(':', '_')
            if args.dimension == None:
                address = f'./results/log/Machine_learning_{args.dataset}_{args.density}'
            else:
                address = f'./results/log/' + self.filename
            logging.basicConfig(level=logging.INFO, filename=f'{address}_{ts}.log', filemode='w')
        else:
            logging.basicConfig(level=logging.INFO, filename=f'./' + 'None.log', filemode='a')
        self.logger = logging.getLogger(self.args.model)

    def save_result(self, metrics):
        args = self.args
        makedir('./results/metrics/')
        if args.dimension == None:
            address = f'./results/metrics/Machine_learning_{args.dataset}_{args.density}'
        else:
            address = f'./results/metrics/' + self.filename
        for key in metrics:
            pickle.dump(np.mean(metrics[key]), open(address + key + 'mean.pkl', 'wb'))
            pickle.dump(np.std(metrics[key]), open(address + key + 'std.pkl', 'wb'))

    # 日志记录
    def log(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        final_string = time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time())) + string
        green_string = f'\033[92m{final_string}\033[0m'
        self.logger.info(final_string[:-1])
        print(green_string)

    def __call__(self, string):
        if self.args.verbose:
            self.log(string)

    def only_print(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        final_string = time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time())) + string
        green_string = f'\033[92m{final_string}\033[0m'
        print(green_string)

    def show_epoch_error(self, runId, epoch, epoch_loss, result_error, train_time):
        if self.args.verbose and epoch % self.args.verbose == 0 and not self.args.program_test:
            pass

    def show_test_error(self, runId, monitor, results, sum_time):
        if self.args.classification:
            pass
