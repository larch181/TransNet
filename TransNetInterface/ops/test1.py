#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li
import os
import sys
import numpy as np
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
sys.path.append( os.path.join(BASE_PATH,'build/lib.win-amd64-3.6/ops'))
import DataAnalyze


# point_removed = np.random.randint(0,2,size=100)
# vertices =np.random.randn(100,3).astype(np.float32)
# data = np.zeros((point_removed.shape[0], 3), dtype=np.float32)
# count = DataAnalyze.fetch_data(point_removed,vertices,data)
# print(count)
# print(data)
from time import time
import DataOperator
points = np.random.rand(10000,4)
intrinsic = np.array([389.284,389.284,320.569,238.517])
out_pixel = np.zeros([points.shape[0],2]).astype(np.int32)
K=8
index = np.zeros(K).astype(np.int32)
print(index)
#DataOperator.farthest_point_samplin_func(points,index,8)
t = time()
DataOperator.proj_point2pixel_func(points,intrinsic,out_pixel)
print(time()-t)
print(out_pixel.shape)
#print(index)
