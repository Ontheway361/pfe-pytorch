#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/04/26
author: lujie
"""

import os
import numpy as np
import pandas as pd

from IPython import embed

if __name__ == "__main__":

    data_dir = '/home/jovyan/jupyer/benchmark_images/faceu/face_recognition/casia_webface'
    txt_file = os.path.join(data_dir, 'anno_file/casia_landmark.txt')
    with open(txt_file, 'r') as f:
        casia_org = f.readlines()
    f.close()
    org_file = []
    for line in casia_org:
        
        line = line.strip().split('\t')[:2]
        org_file.append(line)
    df_org = pd.DataFrame(org_file, columns=['pid_face', 'uid'])
    
    align_file = []
    align_dir  = os.path.join(data_dir, 'align_112_112')
    for pid in os.listdir(align_dir):
        
        if '.DS' in pid or '._' in pid or '.ipy' in pid:
            continue
        
        for face in os.listdir(os.path.join(align_dir, pid)):
            
            if '.DS' in face or '._' in face or '.ipy' in face:
                continue
            align_file.append([pid + '/' + face, 0])
    df_align = pd.DataFrame(align_file, columns=['pid_face', 'psedo'])
    
    df_inter = pd.merge(df_org, df_align, on='pid_face', how='inner')
    print('num_org : %4d, num_align : %4d, num_inter : %4d' % (len(df_org), len(df_align), len(df_inter)))
    df_inter = df_inter[['pid_face', 'uid']]
    data_inter = []
    for idx, row in df_inter.iterrows():
        
        row  = list(row)
        line = ' '.join(row) + '\n'
        data_inter.append(line)
    
    inter_file = os.path.join(data_dir, 'anno_file/casia_org_join_align.txt')
    with open(inter_file, 'w') as f:
        f.writelines(data_inter)
    f.close()
    
    df_inter.to_csv(os.path.join(data_dir, 'anno_file/casia_org_join_align.csv'), index=None)
        
        
