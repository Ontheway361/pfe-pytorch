#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import random
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as T

import model as mlib

torch.backends.cudnn.bencmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" # TODO

from IPython import embed


class ProbFace(object):

    def __init__(self, args):

        self.args   = args
        self.model  = dict()
        self.trans = T.Compose([T.ToTensor(), \
                                T.Normalize(mean=[0.5, 0.5, 0.5], \
                                             std=[0.5, 0.5, 0.5])])
        self.device = args.use_gpu and torch.cuda.is_available()
        self._model_loader()
        

    def _model_loader(self):

        self.model['backbone']  = mlib.MobileFace(self.args.in_feats, self.args.drop_ratio)
        self.model['uncertain'] = mlib.UncertaintyHead(self.args.in_feats)
        self.model['criterion'] = mlib.MLSLoss(mean=False)
        
        if self.device:
            self.model['backbone']  = self.model['backbone'].cuda()
            self.model['uncertain'] = self.model['uncertain'].cuda()
            self.model['criterion'] = self.model['criterion'].cuda()

        if self.device and len(self.args.gpu_ids) > 1:
            self.model['backbone']  = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['uncertain'] = torch.nn.DataParallel(self.model['uncertain'], device_ids=self.args.gpu_ids)
            print('Parallel mode was going ...')
        elif self.device:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['uncertain'].load_state_dict(checkpoint['uncertain'])
            print('Resuming the train process at %3d epoches ...' % checkpoint['epoch'])
        
        self.model['backbone'].eval()
        self.model['uncertain'].eval()
        print('Model loading was finished ...')

    
    @staticmethod
    def cal_pair_mls(mu1, mu2, logsig_sq1=None, logsig_sq2=None):
        ''' Calculate the mls of pair faces '''
        
        sig_sq1 = torch.exp(logsig_sq1)
        sig_sq2 = torch.exp(logsig_sq2)
        sig_sq_mutual = sig_sq1 + sig_sq2
        mu_diff       = mu1 - mu2
        mls_pointwise = torch.mul(mu_diff, mu_diff) / sig_sq_mutual + torch.log(sig_sq_mutual)
        mls_score     = mls_pointwise.sum(dim=1).item()
        return mls_score
    
    
    def _process_pair(self, face1, face2):
        ''' Get the mls score of pair faces '''

        mls_score = None
        if face1 is None or  face2 is None:
            mls_score = None
        else:
            face1 = self.trans(face1).unsqueeze(0)
            face2 = self.trans(face2).unsqueeze(0)
            
            if self.device == 'cuda':
                face1 = face1.cuda()
                face2 = face2.cuda()
            try:
                mu1, feat1 = self.model['backbone'](face1)
                mu2, feat2 = self.model['backbone'](face2)
                logsig_sq1 = self.model['uncertain'](feat1)
                logsig_sq2 = self.model['uncertain'](feat2)
            except Exception as e:
                print(e)
            else:
                mls_score = self.cal_pair_mls(mu1, mu2, logsig_sq1, logsig_sq2)
        return mls_score


cp_dir = '/home/jovyan/jupyter/checkpoints_zoo/face-recognition'

def infer_args():

    parser = argparse.ArgumentParser(description='PyTorch for ProbFace')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--workers', type=int,  default=0)

    # -- model
    parser.add_argument('--in_size',    type=tuple, default=(112, 112))   # FIXED
    parser.add_argument('--in_feats',   type=int,   default=512)
    parser.add_argument('--drop_ratio', type=float, default=0.4)          # TODOqq
    parser.add_argument('--resume',     type=str,   default=os.path.join(cp_dir, 'pfe/sota.pth.tar'))           # checkpoint

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    probu = ProbFace(infer_args())
    face1 = cv2.imread('test_img/Pedro_Solbes_0003.jpg')
    face2 = cv2.imread('test_img/Pedro_Solbes_0004.jpg')
    face3 = cv2.imread('test_img/Zico_0003.jpg')
    mls_score = probu._process_pair(face2, face3)
    print(mls_score)
    embed()
