#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/04/26
author: lujie
"""


import os
import cv2
import random
import numpy as np
import torchvision
import pandas as pd
from torch.utils import data

from IPython import embed


class CASIAWebFacePFE(data.Dataset):

    def __init__(self, args, mode = 'train'):

        super(CASIAWebFacePFE, self).__init__()
        self.args       = args
        self.mode       = mode
        self.transforms = torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], \
                                                                std=[0.5, 0.5, 0.5])])
        self._pfe_process()
        

    def _pfe_process(self):
        
        with open(self.args.train_file, 'r') as f:
            self.lines  = f.readlines()
        f.close()
        if self.args.is_debug:
            self.lines = self.lines[:1024]  # just for debug
            print('debug version for casia ...')
            
        pids_dict = {}
        for line in self.lines:
            
            line = line.strip().split(' ')
            if line[-1] in pids_dict.keys():
                pids_dict[line[-1]].append(line[0])
            else:
                pids_dict[line[-1]] = [line[0]]
        
        self.lines = pd.Series(pids_dict).to_frame()
        self.lines['pid'] = self.lines.index
        self.lines.index  = range(len(self.lines))
        self.lines = np.array(self.lines[['pid', 0]]).tolist() * 6
        random.shuffle(self.lines)
        self.lines = np.array(self.lines)
        
        
    
    def _random_samples_from_class(self, files_list):
        
        indices = []
        random.shuffle(files_list)
        if len(files_list) >= self.args.num_face_pb:
            indices = files_list[:self.args.num_face_pb]
        else:
            extend_times = int(np.ceil(self.args.num_face_pb / max(1, len(files_list)))) - 1
            tmp_list = files_list
            for i in range(extend_times):
                tmp_list.extend(files_list)
            indices = tmp_list[:self.args.num_face_pb]
        return indices
            
        
    def _load_imginfo(self, files_list):
        
        sample_files = self._random_samples_from_class(files_list)
        
        imgs = []
        try:
            for file in sample_files:
                img_path = os.path.join(self.args.casia_dir, 'align_112_112', file)
                img = cv2.resize(cv2.imread(img_path), self.args.in_size)  #  TODO
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                img = self.transforms(img)
                imgs.append(img)
        except Exception as e:
            imgs = []
        return imgs


    def __getitem__(self, index):

        info = self.lines[index]
        imgs = self._load_imginfo(info[1])
        while len(imgs) == 0:
            idx  = np.random.randint(0, len(self.lines) - 1)
            info = self.lines[idx]
            imgs = self._load_imginfo(info[1])
            
        return (imgs, int(info[0]))


    def __len__(self):
        return len(self.lines)
