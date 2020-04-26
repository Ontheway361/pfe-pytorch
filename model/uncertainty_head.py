#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/04/23
author: lujie
"""
import torch
import torch.nn as nn
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
            diff    = torch.mul(mu_diff, mu_diff) / (1e-10 + sig_sum) + torch.log(sig_sum)
            diff    = diff.sum(dim=2, keepdim=False)
            return diff

    def forward(self, gty, mu_X, log_sigma_sq):

        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        sig_X    = torch.exp(log_sigma_sq)
        loss_mat = self.negMLS(mu_X, sig_X)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        pos_mask = (non_diag_mask * gty_mask).bool()
        pos_loss = loss_mat[pos_mask].mean()
        return pos_loss


if __name__ == "__main__":

    mls = MLSLoss(mean=False)
    gty = torch.Tensor([1, 2, 3, 2, 3, 3, 2])
    muX = torch.randn((7, 3))
    siX = torch.rand((7,3))
    diff = mls(gty, muX, siX)
    print(diff)
