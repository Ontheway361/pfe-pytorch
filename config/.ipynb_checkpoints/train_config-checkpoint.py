#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

root_dir  = '/home/jovyan/jupyter/benchmark_images/faceu'
lfw_dir   = osp.join(root_dir, 'face_verfication/lfw')
casia_dir = osp.join(root_dir, 'face_recognition/casia_webface')
cp_dir    = '/home/jovyan/jupyter/checkpoints_zoo/face-recognition'

def training_args():

    parser = argparse.ArgumentParser(description='PyTorch metricface')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1, 2, 3])
    parser.add_argument('--workers', type=int,  default=0)

    # -- model
    parser.add_argument('--in_size',    type=tuple,  default=(112, 112))   # FIXED
    parser.add_argument('--offset',     type=int,    default=2)            # FIXED
    parser.add_argument('--t',          type=float,  default=0.2)          # MV
    parser.add_argument('--margin',     type=float,  default=0.5)          # FIXED
    parser.add_argument('--easy_margin',type=bool,   default=True)
    parser.add_argument('--scale',      type=float,  default=32)           # FIXED
    parser.add_argument('--backbone',   type=str,    default='resnet18')           # TODO | iresse50
    parser.add_argument('--in_feats',   type=int,    default=512)
    parser.add_argument('--drop_ratio', type=float,  default=0.4)          # TODOqq

    parser.add_argument('--fc_mode',    type=str,    default='arcface',    choices=['softmax', 'sphere', 'cosface', 'arcface', 'mvcos', 'mvarc'])
    parser.add_argument('--hard_mode',  type=str,    default='adaptive', choices=['fixed', 'adaptive']) # MV
    parser.add_argument('--loss_mode',  type=str,    default='ce',       choices=['ce', 'focal_loss', 'hardmining'])
    parser.add_argument('--hard_ratio', type=float,  default=0.9)          # hardmining
    parser.add_argument('--loss_power', type=int,    default=2)            # focal_loss
    parser.add_argument('--classnum',   type=int,    default=10574)        # CASIA (10574)
    
    # fine-tuning
    parser.add_argument('--resume',          type=str,  default=osp.join(cp_dir, 'pfe/epoch_5_train_loss_-1079.0558.pth.tar'))           # checkpoint
    parser.add_argument('--fine_tuning',     type=bool, default=False)        # just fine-tuning
    parser.add_argument('--freeze_backbone', type=bool, default=True)   # TODO
    
    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)        # 
    parser.add_argument('--end_epoch',   type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=64)      # NOTE : 
    parser.add_argument('--num_face_pb', type=int,   default=4)
    parser.add_argument('--base_lr',     type=float, default=1e-3)      # TODO : [0.1 for backbone]
    parser.add_argument('--lr_adjust',   type=list,  default=[2, 3, 4]) # TODO : [16, 25, 35]
    parser.add_argument('--gamma',       type=float, default=0.1)      # FIXED
    parser.add_argument('--weight_decay',type=float, default=5e-4)     # FIXED
    
    # -- dataset
    parser.add_argument('--casia_dir',  type=str, default=casia_dir)  
    parser.add_argument('--lfw_dir',    type=str, default=osp.join(lfw_dir,   'align_112_112'))
    parser.add_argument('--train_file', type=str, default=osp.join(casia_dir, 'anno_file/casia_org_join_align.txt'))
    parser.add_argument('--pairs_file', type=str, default=osp.join(lfw_dir,   'anno_file/pairs.txt'))
    parser.add_argument('--try_times',  type=int, default=5)
    
    
    # -- verification
    parser.add_argument('--n_folds',   type=int,   default=10)
    parser.add_argument('--thresh_iv', type=float, default=0.005)
    
    # -- save or print
    parser.add_argument('--is_debug',  type=str,   default=False)   # TODO 
    parser.add_argument('--save_to',   type=str,   default=osp.join(cp_dir, 'pfe'))
    parser.add_argument('--print_freq',type=int,   default=100)  # v0 : <64， 166> | <128, 83>
    parser.add_argument('--save_freq', type=int,   default=1)  # TODO 
    
    args = parser.parse_args()

    return args
