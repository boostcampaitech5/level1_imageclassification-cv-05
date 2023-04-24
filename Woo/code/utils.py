import numpy as np
import torch
import torch.nn as nn
import math
from tqdm.notebook import tqdm


def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset, ascii=True)])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[..., 0].mean()
    mean_g = mean_[..., 1].mean()
    mean_b = mean_[..., 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset, ascii=True)])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[..., 0].mean()
    std_g = std_[..., 1].mean()
    std_b = std_[..., 2].mean()
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


def linear_param_initialize(model, n_feature):
    
    for m in model.fc.children():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            stdv = 1 / math.sqrt(n_feature)
            m.bias.data.uniform_(-stdv, stdv)
            n_feature //= 2

class train_mean_std:
    mean = (0.5601906, 0.5240981, 0.50145423)
    std = (0.2331904, 0.24300218, 0.2456751)
    
class eval_mean_std:
    mean = (0.53205013, 0.47565353, 0.44818786)
    std = (0.23798905, 0.24770516, 0.2427656)