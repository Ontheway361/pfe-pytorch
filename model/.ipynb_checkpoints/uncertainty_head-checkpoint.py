#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/04/23
author: lujie
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from IPython import embed

class UncertaintyHead(nn.Module):
    ''' Evaluate the log(sigma^2) '''
    
    def __init__(self, in_feat = 512):

        super(UncertaintyHead, self).__init__()
        self.fc1   = Parameter(torch.Tensor(in_feat, in_feat))
        self.bn1   = nn.BatchNorm1d(in_feat, affine=True)
        self.relu  = nn.ReLU(in_feat)
        self.fc2   = Parameter(torch.Tensor(in_feat, in_feat))
        self.bn2   = nn.BatchNorm1d(in_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta  = Parameter(torch.Tensor([0.0]))   # default = -7.0
        
        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(F.linear(x, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x


if __name__ == "__main__":

    unh = UncertaintyHead(in_feat=3)
    
    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]])
    
    muX = torch.from_numpy(mu_data).float()
    log_sigma_sq = unh(muX)
    print(log_sigma_sq)
