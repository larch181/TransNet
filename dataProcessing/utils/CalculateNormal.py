#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li


import sys
import os
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

POINTCLOUD_FOLDER = os.path.dirname(os.path.dirname(ROOT_DIR))
print(POINTCLOUD_FOLDER)

INPUT_DATA_FOLDER = os.path.join(POINTCLOUD_FOLDER,'Dataset/Data')
RAW_MODEL = os.path.join(INPUT_DATA_FOLDER,'einstein_normalized_s.xyz')
INPUT_MODEL = os.path.join(INPUT_DATA_FOLDER,'einstein_noise.xyz')
OUTPUT_MODEL = os.path.join(INPUT_DATA_FOLDER,'einstein_noise@norm.xyz')

def calculateNormal():
    raw_data = np.loadtxt(RAW_MODEL)

    raw_points = raw_data[:,:3]
    raw_normals = raw_data[:,3:6]

    input_data = np.loadtxt(INPUT_MODEL)

    print(raw_data.shape)
    print(input_data.shape)

    outdata = np.zeros((input_data.shape[0],6))
    for i in tqdm(range(input_data.shape[0])):
        point = input_data[i,:].reshape((1,-1))
        distance = np.dot(point,raw_points.T)
        idx = np.argmin(distance)
        outdata[i,:3] = input_data[i,:3]
        outdata[i,3:6] = raw_normals[idx,:3]

    np.savetxt(OUTPUT_MODEL,outdata, fmt='%0.6f')
    #for i in range(input_data.shape[0]):





if __name__=='__main__':
    calculateNormal()
