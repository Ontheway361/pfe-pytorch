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

    data_dir = '/Users/relu/data/benchmark_images/faceu/casia_webface'
    txt_file = os.path.join(data_dir, 'anno_file/casia_landmark.txt')
