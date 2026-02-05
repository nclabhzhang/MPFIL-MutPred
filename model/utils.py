import numpy as np
import torch
from torch import manual_seed
from torch.cuda import manual_seed_all
from random import seed
from time import localtime, strftime, time
from math import sqrt
from os.path import exists
from os import makedirs

def set_random_seed(seed_num):
    torch.cuda.manual_seed(seed_num)
    manual_seed(seed_num)
    manual_seed_all(seed_num)
    np.random.seed(seed_num)
    seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def timestamp():
    now = int(round(time() * 1000))
    now = strftime('%m%d%H%M', localtime(now/1000))
    return now

def make_save_dir(time_stamp):
    save_path = './ckpt/'+time_stamp
    adam_path = './optimizers/'+time_stamp
    result_path = './result/'+time_stamp
    test_save_path_p = './result_ddg/' + time_stamp + '/' + 'best_pcc_predict' + '/'
    test_save_path_r = './result_ddg/' + time_stamp + '/' + 'best_rmse_predict' + '/'
    test_save_path_m = './result_ddg/' + time_stamp + '/' + 'best_mae_predict' + '/'
    if not exists(save_path):
        makedirs(save_path)
        print(f"made dir {save_path}")
    if not exists(adam_path):
        makedirs(adam_path)
        print(f"made dir {adam_path}")
    if not exists(result_path):
        makedirs(result_path)
        print(f"made dir {result_path}")
    if not exists(test_save_path_p):
        makedirs(test_save_path_p)
    if not exists(test_save_path_r):
        makedirs(test_save_path_r)
    if not exists(test_save_path_m):
        makedirs(test_save_path_m)

def calc_std_dev(lst):
    n = len(lst)
    mean = sum(lst) / n
    variance = sum((x - mean) ** 2 for x in lst) / n
    std_dev = sqrt(variance)
    return std_dev

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (1e-11 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)