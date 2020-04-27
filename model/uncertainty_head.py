#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/04/23
author: lujie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from IPython import embed

class UncertaintyHead(nn.Module):
    ''' Evaluate the log(sigma^2) '''
    
    def __init__(self, in_feat = 512):

        super(UncertaintyHead, self).__init__()
        self.fc1   = Parameter(torch.FloatTensor(in_feat, in_feat))
        self.bn1   = nn.BatchNorm1d(in_feat)
        self.relu  = nn.PReLU(in_feat)
        self.fc2   = Parameter(torch.FloatTensor(in_feat, in_feat))
        self.bn2   = nn.BatchNorm1d(in_feat)
        self.register_buffer('gamma', torch.ones(1) * 1e-4)
        self.register_buffer('beta', torch.zeros(1) - 7.0)
        nn.init.xavier_uniform_(self.fc1)
        nn.init.xavier_uniform_(self.fc2)


    def forward(self, x):

        x = self.relu(self.bn1(F.linear(x, self.fc1)))
        x = self.bn2(F.linear(x, self.fc2))
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))
        return x


if __name__ == "__main__":

    mls = UncertaintyHead(in_feat=5)
    muX = torch.randn((20, 5))
    diff = mls(muX)
    print(diff)
