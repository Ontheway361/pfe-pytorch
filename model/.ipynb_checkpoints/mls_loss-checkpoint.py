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
from IPython import embed

class MLSLoss(nn.Module):

    def __init__(self, mean = False):

        super(MLSLoss, self).__init__()
        self.mean = mean

    def negMLS(self, mu_X, sigma_sq_X):

        if self.mean:
            XX = torch.mul(mu_X, mu_X).sum(dim=1, keepdim=True)
            YY = torch.mul(mu_X.T, mu_X.T).sum(dim=0, keepdim=True)
            XY = torch.mm(mu_X, mu_X.T)
            mu_diff = XX + YY - 2 * XY
            sig_sum = sigma_sq_X.mean(dim=1, keepdim=True) + sigma_sq_X.T.sum(dim=0, keepdim=True)
            diff    = mu_diff / (1e-8 + sig_sum) + mu_X.size(1) * torch.log(sig_sum)
            return diff
        else:
            mu_diff = mu_X.unsqueeze(1) - mu_X.unsqueeze(0)
            sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)
            diff    = torch.mul(mu_diff, mu_diff) / (1e-10 + sig_sum) + torch.log(sig_sum)  # BUG
            diff    = diff.sum(dim=2, keepdim=False)
            return diff

    def forward(self, mu_X, log_sigma_sq, gty):
        
        mu_X     = F.normalize(mu_X) # if mu_X was not normalized by l2
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if gty.device.type == 'cuda':
            non_diag_mask = non_diag_mask.cuda(0)      
        sig_X    = torch.exp(log_sigma_sq)
        loss_mat = self.negMLS(mu_X, sig_X)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        pos_mask = (non_diag_mask * gty_mask) > 0
        pos_loss = loss_mat[pos_mask].mean()
        return pos_loss


if __name__ == "__main__":

    mls = MLSLoss(mean=False)
    gty = torch.Tensor([1, 2, 3, 2, 3, 3, 2])
    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]])
    
    si_data = np.array([[-0.28463233, -2.5517333 ,  1.4781238 ],
                        [-0.10505871, -0.31454122, -0.29844758],
                        [-1.3067418 ,  0.48718405,  0.6779812 ],
                        [ 2.024449  , -1.3925922 , -1.6178994 ],
                        [-0.08328865, -0.396574  ,  1.0888542 ],
                        [ 0.13096762, -0.14382902,  0.2695235 ],
                        [ 0.5405067 , -0.67946523, -0.8433032 ]])
    
    muX = torch.from_numpy(mu_data)
    siX = torch.from_numpy(si_data)
    print(muX.shape)
    diff = mls(muX, siX, gty)
    print(diff)
