#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import os
import sys

if sys.version_info >= (3, 0):
    from functools import reduce
import scipy.misc
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import math
import Tools3D
import pc_util
import FileOperator

HOME_DIR = 'E:/DeepLearning/PointCloud/Dataset/Data/'


SamplingTrainFile = os.path.join(HOME_DIR, 'wholesphere256/quaternion_1000_50_train.xyz')
SamplingTestFile = os.path.join(HOME_DIR, 'wholesphere256/quaternion_1000_50_train.xyz')

OUTDATA_FOLDER = os.path.join(HOME_DIR, 'dataset/')

POINT_NUM = 1024

if not os.path.exists(OUTDATA_FOLDER):
    os.mkdir(OUTDATA_FOLDER)


# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def save_ply_file(data, filename):
    vertex = np.zeros(data.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i in range(data.shape[0]):
        vertex[i] = (data[i][0], data[i][1], data[i][2])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_out.write(filename)


# ----------------------------------------------------------------
# Following are the generation functions
# ----------------------------------------------------------------

# Step1: normalized all the dataset

def normalized(data):
    centroid = np.mean(data, axis=0, keepdims=True)
    data = data - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(abs(data) ** 2, axis=-1)))
    data = data / furthest_distance
    return data


def load_xyz(dataFolder):
    print(dataFolder)
    file_list = glob.glob(dataFolder)
    file_list.sort()
    file_count = len(file_list)
    dataList = np.zeros((file_count, POINT_NUM, 3), dtype=float)
    index = 0
    for file_path in file_list:
        print(file_path)
        data_T = np.loadtxt(file_path)
        dataList[index, :, :] = data_T[:, 0:3]
        index += 1
    np.squeeze(dataList)
    print(dataList)
    return dataList


def load_xyz_file(pointNum=1024, dataFile=None):
    if dataFile == None:
        dataFile = 'E:/DeepLearning/PointCloud/Dataset/sampled{0}/einstein_sampled.xyz'.format(str(pointNum))
    #print(dataFile)
    data = np.loadtxt(dataFile)
    # data = data_T[:, 0:3]
    return data


def generate_random_rotation(angleNum=10000):
    angles = np.zeros((angleNum, 3), dtype=float)

    for k in range(angleNum):
        angles[k, :] = np.random.uniform(size=3) * 2 * np.pi
    return angles


def generate_random_rotationVec(angleNum=10000):
    angles = np.zeros((angleNum, 4), dtype=float)
    for k in range(angleNum):
        rot_axis = np.random.uniform(size=3)
        norm = Tools3D.vector_length(rot_axis)

        rot_axis = (rot_axis[0] / norm, rot_axis[1] / norm, rot_axis[2] / norm)
        angles[k, 0:-1] = rot_axis
        angles[k, 3] = np.random.uniform() * 2 * np.pi
    return angles


# Write numpy array data and label to h5_filename
def save_h5_data_label(h5_filename, data, label, angles=None, data_dtype='float32', label_dtype='int32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)

    if angles != None:
        h5_fout.create_dataset(
            'angles', data=angles,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)

    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_angle_h5(h5_filename, data, angles, data_dtype='float'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)

    h5_fout.create_dataset(
        'angles', data=angles,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)

    h5_fout.close()


def load_angle_h5(h5_filename):
    with h5py.File(h5_filename) as f:
        data = f['data'][:]
        labels = f['label'][:]
    return data, labels


def euler2mat(z=0, y=0, x=0):
    # Return matrix for rotations around z, y and x axes
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def generateDataset(data, angles):
    pcData = np.zeros((angles.shape[0], data.shape[0], data.shape[1]), dtype=float)
    print(pcData.shape)
    for k in range(angles.shape[0]):
        z, y, x = angles[k, :]
        rotationMat = euler2mat(z, y, x)
        pcData[k, :, :] = np.dot(data, rotationMat)

    return pcData


def generateDataset_rotationVector(data, rotationVectors):
    pcData = np.zeros((rotationVectors.shape[0], data.shape[0], data.shape[1]), dtype=float)
    print(pcData.shape)
    print(rotationVectors.shape)
    for i in range(rotationVectors.shape[0]):
        assert (rotationVectors.shape[1] == 4)
        rotationMat = Tools3D.rotation_vector_To_rotation_matrix(rotationVectors[i, 0:-1], rotationVectors[i, -1])
        pcData[i, :, :] = np.dot(data, rotationMat)

    return pcData


def generateDataset_Quaternions(data, quaternions):
    pcData = np.zeros((quaternions.shape[0], data.shape[0], data.shape[1]), dtype=float)
    print(pcData.shape)
    print(quaternions.shape)
    for i in range(quaternions.shape[0]):
        assert (quaternions.shape[1] == 4)
        rotationMat = Tools3D.quaternion_To_rotation_matrix(quaternions[i, :])
        pcData[i, :, :] = np.dot(data, rotationMat)

    return pcData


def generate_dataFile(fileName):
    pc = load_xyz_file(dataFile=fileName)
    trainFile = fileName.replace('.xyz', '_train.h5')
    angles_train = generate_random_rotation()
    lables_train = np.ones(angles_train.shape[0], dtype=np.uint8)
    data_train = generateDataset(pc, angles_train)
    save_h5_data_label(trainFile, data_train, lables_train, angles_train)

    testFile = fileName.replace('.xyz', '_test.h5')
    angles_test = generate_random_rotation(2000)
    data_test = generateDataset(pc, angles_test)
    lables_test = np.zeros(angles_test.shape[0], dtype=np.uint8)
    save_h5_data_label(testFile, data_test, lables_test, angles_test)


def generate_dataFile_rotationVec(fileName):
    pc = load_xyz_file(dataFile=fileName)
    name = fileName.split('/')[-1].replace('.xyz', '')

    angles_train = load_xyz_file(dataFile=SamplingTrainFile)

    # print(angles_train[0:32,:])
    # trainFile = fileName.replace('.xyz', '_rotationvector_train.h5')

    trainFile = os.path.join(OUTDATA_FOLDER, name + '_rotationvector_train.h5')

    lables_train = np.ones(angles_train.shape[0], dtype=np.uint8)
    data_train = generateDataset_rotationVector(pc, angles_train)
    save_h5_data_label(trainFile, data_train, lables_train, angles_train)

    # testFile = fileName.replace('.xyz', '_rotationvector_test.h5')

    testFile = os.path.join(OUTDATA_FOLDER, name + '_rotationvector_test.h5')

    angles_test = load_xyz_file(dataFile=SamplingTestFile)  # generate_random_rotationVec(2000)
    data_test = generateDataset_rotationVector(pc, angles_test)
    lables_test = np.zeros(angles_test.shape[0], dtype=np.uint8)
    save_h5_data_label(testFile, data_test, lables_test, angles_test)


def generate_dataFile_quaternion(fileName):
    pc = load_xyz_file(dataFile=fileName)

    name = fileName.split('/')[-1].replace('.xyz', '')

    angles_train = load_xyz_file(dataFile=SamplingTrainFile)
    # trainFile = fileName.replace('.xyz', '_quaternion_train.h5')

    trainFile = os.path.join(OUTDATA_FOLDER, name + '_quaternion_train.h5')

    lables_train = np.ones(angles_train.shape[0], dtype=np.uint8)
    data_train = generateDataset_Quaternions(pc, angles_train)
    save_h5_data_label(trainFile, data_train, lables_train, angles_train)

    # testFile = fileName.replace('.xyz', '_quaternion_test.h5')
    testFile = os.path.join(OUTDATA_FOLDER, name + '_quaternion_test.h5')

    angles_test = load_xyz_file(dataFile=SamplingTestFile)  # generate_random_rotationVec(2000)
    data_test = generateDataset_Quaternions(pc, angles_test)
    lables_test = np.zeros(angles_test.shape[0], dtype=np.uint8)
    save_h5_data_label(testFile, data_test, lables_test, angles_test)


def batch_generate(dataFolder):
    print(dataFolder)
    file_list = glob.glob(dataFolder)
    file_list.sort()

    # For Testing
    if False:
        for file_path in file_list:
            print(file_path)
            data, angles = load_angle_h5(file_path.replace('.xyz', '.h5'))
            print(data.shape)
            print(angles.shape)
    else:
        for file_path in file_list:
            print(file_path)
            generate_dataFile_rotationVec(file_path)

from tqdm import tqdm
def dataSet_generate(dataFolder,channel=3):
    type_num, lable_num, filePrefix = FileOperator.get_train_data(dataFolder)
    train_lables = np.zeros([lable_num, type_num])
    train_data = np.zeros([lable_num, POINT_NUM * type_num, channel])
    for i in tqdm(range(lable_num)):
        train_lables[i, :] = i
        dataT = np.zeros([POINT_NUM * type_num, channel], dtype=np.float32)
        for j in range(type_num):
            file_name = (filePrefix + "_%d_%d.xyz") % (i, j)
            dataT[j * POINT_NUM:(j + 1) * POINT_NUM, :] = load_xyz_file(pointNum=POINT_NUM, dataFile=file_name)
            # dataT = np.concatenate((dataT, data_file), axis=0)
        train_data[i, :, :] = dataT

    name = filePrefix.split('/')[-1]
    trainFile = os.path.join(OUTDATA_FOLDER, 'einstein_train%s.h5'%(PREFIX))

    save_h5_data_label(trainFile, train_data, train_lables)
    #loadingTest(trainFile)
    print('Finished')

def loadingTest(filename):
    data, lables = load_angle_h5(filename)
    print(data.shape)
    print(lables.shape)
    print(lables[256:289,:])
    print(data[234:245,0:5,0])

PREFIX='S2'

SegmentFolder = os.path.join(HOME_DIR, 'segment%s/postprocessing/'%(PREFIX))

if __name__ == '__main__':
    print(SegmentFolder)
    dataSet_generate(SegmentFolder,channel=6)
    filename = os.path.join(OUTDATA_FOLDER, 'einstein_train%s.h5'%(PREFIX))
    loadingTest(filename)