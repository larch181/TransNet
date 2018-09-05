#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import sys
import os
import trimesh
from  trimesh import sample
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)

INPUT_FILE = os.path.join(BASE_DIR,'data/wholesphere256.obj')
OUT_FILE =  os.path.join(BASE_DIR,'data/wholesphere256.xyz')
mesh = trimesh.load(INPUT_FILE)
sample_list = sample.sample_surface_even(mesh,500)
sample_point = np.array(sample_list[0])
sample_idx = np.array(sample_list[1])
np.savetxt(OUT_FILE,sample_point,fmt='%0.6f')

print(len(sample_list))
print(sample_point.shape)
print(sample_idx.shape)


