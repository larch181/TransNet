#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import os
import sys
import scipy.misc
from glob import glob
from tqdm import tqdm
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
import numpy as np
import h5py
import Tools3D




POINTCLOUD_DIR = os.path.dirname(ROOT_DIR)
print('POINTCLOUD_DIR=', POINTCLOUD_DIR)
HOME_DIR = os.path.join(POINTCLOUD_DIR, 'Dataset/Data')
INPUT_DATA_FILE = os.path.join(HOME_DIR, 'partialface/partial_dataset/einstein_normal.xyz')
SAMPLING_FOLDER = os.path.join(POINTCLOUD_DIR, 'Dataset/sample_transformation')
OUTPUT_FOLDER = os.path.join(ROOT_DIR,'PartialNet/data')

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


def load_xyz_file(pointNum=1024, dataFile=None):
    # print(dataFile)
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
def save_h5_data_diffnum(h5_filename, data, numbers, label, angles, data_dtype='float32', label_dtype='int32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'numbers', data=numbers,
        compression='gzip', compression_opts=4,
        dtype=label_dtype)
    h5_fout.create_dataset(
        'angles', data=angles,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()

def load_angle_h5(h5_filename):
    with h5py.File(h5_filename) as f:
        data = f['data'][:]
        angles = f['angles'][:]
        numbers = f['numbers'][:]
        labels = f['label'][:]
    return data,numbers, angles, labels


def batch_data_sampling(data, patch_size=1024, normalization=True):
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(0)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    B, N, C = data.shape
    output_data = np.zeros((B, patch_size, C))
    print('start to sampling:', patch_size)
    for i in tqdm(range(B)):
        point = np.squeeze(data[i, ...])
        if normalization:
            point = normalized(point)

        pointNum = point.shape[0]
        idx = np.arange(pointNum)
        if pointNum < patch_size:

            offset = pointNum - patch_size
            idx = np.concatenate([np.arange(pointNum), np.random.randint(pointNum, size=offset)], axis=0)
            np.random.shuffle(idx)

        elif pointNum > patch_size:

            sample_data = np.expand_dims(point, axis=0)
            sample_seed = tf_sampling.farthest_point_sample(patch_size, sample_data)

            # with tf.Session(config=config) as sess:
            idx = sess.run(sample_seed)

            idx = np.squeeze(idx)

        output_data[i, ...] = point[idx, ...]
    return output_data


def data_sampling(point, sess, patch_size=1024, normalization=True):
    # idx = np.arange(point.shape[0])
    # np.random.shuffle(idx)
    # idx = idx[:patch_size]
    # point = point[idx,...]
    if normalization:
        point = normalized(point)
    # return point

    pointNum = point.shape[0]
    idx = np.arange(pointNum)
    if pointNum < patch_size:

        offset = pointNum - patch_size
        idx = np.concatenate([np.arange(pointNum), np.random.randint(pointNum, size=offset)], axis=0)
        np.random.shuffle(idx)

    elif pointNum > patch_size:

        sample_data = np.expand_dims(point, axis=0)
        sample_seed = tf_sampling.farthest_point_sample(patch_size, sample_data)

        # with tf.Session(config=config) as sess:
        idx = sess.run(sample_seed)

        idx = np.squeeze(idx)

    point = point[idx, ...]
    if normalization:
        point = normalized(point)

    return point


def single_data_sampling(inputFile, patch_size=1024, normalization=True):
    import tf_sampling

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(0)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    point = np.loadtxt(inputFile)

    if normalization:
        point = normalized(point)
    pointNum = point.shape[0]
    idx = np.arange(pointNum)
    if pointNum < patch_size:

        offset = pointNum - patch_size
        idx = np.concatenate([np.arange(pointNum), np.random.randint(pointNum, size=offset)], axis=0)
        np.random.shuffle(idx)

    elif pointNum > patch_size:

        sample_data = np.expand_dims(point, axis=0)
        sample_seed = tf_sampling.farthest_point_sample(patch_size, sample_data)

        # with tf.Session(config=config) as sess:
        idx = sess.run(sample_seed)

        idx = np.squeeze(idx)

    return point[idx, ...]


def batch_normalization(inputfolder, outputfolder):
    samples = glob(input_folder + "/*.xyz")
    samples.sort(reverse=False)

    totalNum = len(samples)
    print(samples[0])
    print('Total Num :', totalNum)
    for i in tqdm(range(totalNum)):
        fileName = samples[i].split('\\')[-1]
        point = load_xyz_file(dataFile=samples[i])
        point = normalized(point)
        centroid = np.mean(point, axis=0, keepdims=True)
        np.savetxt(os.path.join(outputfolder, fileName), point, fmt='%0.6f')


def batch_generate_dataFile(input_folder, max_point_num=10000, channel=3, is_normalized=True, name='pwd'):

   # name = 'einstein_partial'
    folder = 'partial_scan/'
    # folder = 'partial_dataset/'
    print('input_folder:',input_folder)
    samples = glob(input_folder + "/*.xyz")
    samples.sort(reverse=False)

    fileName = input_folder.split('/')[-1]
    print(fileName)
    SamplingFile = os.path.join(SAMPLING_FOLDER, fileName + '.xyz')

    angles = load_xyz_file(dataFile=SamplingFile)
    outH5File = os.path.join(OUTPUT_FOLDER, fileName + '_%s.h5'%(name))

    print('SamplingFile:', SamplingFile)

    print('H5File:', outH5File)

    num_data = angles.shape[0]
    print(len(samples),num_data)
    assert len(samples) == num_data

    points = np.zeros((num_data, max_point_num, channel))
    numbers = np.zeros(num_data).astype(np.int32)
    labels = np.arange(num_data)
    for i in tqdm(range(num_data)):

        fileName = samples[i].split('/')[-1].replace('.xyz', '')
        point = load_xyz_file(dataFile=samples[i])
        if is_normalized:
            point = normalized(point)
        numbers[i] = point.shape[0]
        points[i, :numbers[i], :] = point
        index = int(fileName.split('_')[-1])
        labels[i] = index

    angles = angles[labels, ...]
    save_h5_data_diffnum(outH5File,points,numbers, labels, angles)

def loadingTest(filename):
    data,numbers, angles, labels = load_angle_h5(filename)
    print(np.max(numbers))
    print(angles[100:108,:4])
    print(np.max(np.sum(np.square(data[100,...]),axis=1)))

    #print(np.mean(data[108:110, ...], axis=1))

def real_dataSampling():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(0)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    PC_PATH = os.path.join(POINTCLOUD_DIR, 'Dataset/Data/partialface/real_scan/')
    INPUT_FOLDER = os.path.join(PC_PATH, 'raw')
    OUTPUT_FOLDER = os.path.join(PC_PATH, 'sampled')

    samples = glob(INPUT_FOLDER + "/*.xyz")
    samples.sort(reverse=False)
    point_num = 1024
    total_num = len(samples)

    for i in tqdm(range(total_num)):
        pc = load_xyz_file(dataFile=samples[i])

        output_data = data_sampling(pc, sess, patch_size=point_num, normalization=True)
        assert output_data.shape[0] == point_num
        fileName = samples[i].split('/')[-1]
        outfile = os.path.join(OUTPUT_FOLDER, fileName)
        np.savetxt(outfile, output_data, fmt='%0.6f')

    real_folder = os.path.join(POINTCLOUD_DIR, )


def real_data_nomalization():
    PC_PATH = os.path.join(POINTCLOUD_DIR, 'Dataset/Data/partialface/real_scan/')
    INPUT_FOLDER = os.path.join(PC_PATH, 'raw')
    OUTPUT_FOLDER = os.path.join(PC_PATH, 'raw1')

    samples = glob(INPUT_FOLDER + "/*.xyz")
    samples.sort(reverse=False)
    point_num = 1024
    total_num = len(samples)

    for i in tqdm(range(1)):
        point = load_xyz_file(dataFile=samples[i])

        min_Y = np.mean(point[:,0])
        lis = np.where((point[:, 1] < np.max(point[:,1])-0.01))
        print(min_Y,np.max(point[:,0]),len(lis[0]))
        point = point[lis]

        centroid = np.mean(point, axis=0, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((point - centroid) ** 2, axis=-1)),
                                    keepdims=True)
        point = (point - centroid) / furthest_distance
        fileName = samples[i].split('\\')[-1]
        outfile = os.path.join(OUTPUT_FOLDER, fileName)
        np.savetxt(outfile, point, fmt='%0.6f')


def point_sampling(points, patch_size=1024):
    pointNum = points.shape[0]
    idx = np.arange(pointNum)
    if pointNum < patch_size:
        offset = patch_size - pointNum
        idx = np.concatenate([np.arange(pointNum), np.random.randint(pointNum, size=offset)], axis=0)
        np.random.shuffle(idx)
    else:
        np.random.shuffle(idx)
        idx = idx[:patch_size]

    points = points[idx, ...]

    return points


def point_erosion(points, ratio):
    holes = [0.1, 0.2, 0.3]

    N, _ = points.shape

    num = int(N * ratio)
    print(points.shape[0] - num)
    while (num >= 0) and (points.shape[0] >= 1024):
        N, _ = points.shape

        idx = np.random.randint(0, points.shape[0])
        id = int(np.random.randint(0, 3, size=1))
        distance = np.sum(np.sqrt(np.square(points - points[idx])), axis=1)

        remain_idx = np.where(distance > holes[id])[0]
        points = points[remain_idx, ...]

        num = num - (N - points.shape[0])
    return points


def paritial_data_augment():
    PC_PATH = os.path.join(POINTCLOUD_DIR, 'Dataset/Data/partialface/partial_scan1/')
    INPUT_FOLDER = os.path.join(PC_PATH, 'quaternion_32_8_test')
    OUTPUT_FOLDER = os.path.join(PC_PATH, 'augment')

    samples = glob(INPUT_FOLDER + "/*.xyz")
    samples.sort(reverse=False)
    point_num = 1024
    total_num = len(samples)

    for i in tqdm(range(1)):
        points = load_xyz_file(dataFile=samples[i])
        points = point_erosion(points, ratio=0.2)
        points = point_sampling(points)
        points = point_normalization(points)

        fileName = samples[i].split('\\')[-1]
        outfile = os.path.join(OUTPUT_FOLDER, fileName)
        np.savetxt(outfile, points, fmt='%0.6f')

def gt_data_sampling():
    point_num = 10240
    #NAME = 'pwd'
    NAME = 'einstein'
    INPUT_FOLDER = os.path.join(POINTCLOUD_DIR,'Dataset/%s'%(NAME))
    filename = os.path.join(INPUT_FOLDER, '%s_gt.xyz'%(NAME))
    out_filename = os.path.join(INPUT_FOLDER, '%s_gt_%d.xyz'%(NAME,point_num))
    point_sampled = single_data_sampling(filename,patch_size=point_num,normalization=False)

    # points = np.loadtxt(filename)
    # idx= PointCloudOperator.farthest_sampling(points,20480)
    # point_sampled = points[idx]
    # point_sampled = point_normalization(point_sampled)
    np.savetxt(out_filename,point_sampled, fmt='%0.6f')


def translation_data_sampling():
    output_file = os.path.join(POINTCLOUD_DIR, 'Dataset/wholesphere256/translation.xyz')



def rotation_test():
    print('-------------')
    folder_name='transformation_256_8_4_4_2'
    input_folder = os.path.join(POINTCLOUD_DIR, 'Dataset/%s/partialface' % 'einstein', "partial_scan/%s"%(folder_name))
    TestFolder = os.path.join(POINTCLOUD_DIR,'Dataset/test')
    SamplingFile = os.path.join(SAMPLING_FOLDER,"%s.xyz"%(folder_name))
    samplingData = np.loadtxt(SamplingFile)

    NAME = 'einstein'
    INPUT_FOLDER = os.path.join(POINTCLOUD_DIR, 'Dataset/%s' % (NAME))
    filename = os.path.join(INPUT_FOLDER, '%s_gt.xyz' % (NAME))
    gt_data = np.loadtxt(filename)
    samples = glob(input_folder + "/*.xyz")

    rotMat = Tools3D.quaternion_To_rotation_matrix(samplingData[0,:4])

    data = np.loadtxt(samples[0])
    dataT = np.dot(gt_data,rotMat)
    filename = samples[0].split('\\')[-1].replace('.xyz','')

    np.savetxt(os.path.join(TestFolder,filename+'_p.xyz'), data, fmt='%0.6f')
    np.savetxt(os.path.join(TestFolder,filename+'_gt.xyz'), dataT, fmt='%0.6f')
    print(rotMat)

if __name__ == '__main__':
    #rotation_test()
    #gt_data_sampling()
    #real_data_nomalization()
    #paritial_data_augment()
    #real_dataSampling()
    # generate_dataFile_quaternion('E:/DeepLearning/PointCloud/Dataset/Data/test/einstein_noise@n.xyz')
    # batch_sampling_all()
    # generate_dataFile_quaternion(INPUT_DATA_FILE)
    # batch_generate_dataFile_quaternion()

    #input_folder = os.path.join(PARTIAL_FACE_ROOT_FOLDER, "partial_scan1/quaternion_512_16_train")
    name = 'pwd'
    name = 'einstein'
    folder_name = 'transformation_256_8_4_4_2'
    input_folder = os.path.join(POINTCLOUD_DIR,'Dataset/%s/partialface'%name, "partial_scan/%s"%(folder_name))

    # rotated_test(input_folder)
    # output_folder = os.path.join(PARTIAL_FACE_ROOT_FOLDER, "partial_scan/raw/train_n")
    #
   # batch_normalization(input_folder,output_folder)
    # input_folder = PARTIAL_FACE_TRAIN_DATA_SAMPLE_FOLDER
    # print('input_folder:',input_folder)
    batch_generate_dataFile(input_folder, max_point_num=15000, channel=3, name=name)
    #outH5File = os.path.join(OUTPUT_FOLDER, 'transformation_512_8_1_1_1_%s.h5' % (name))
    #loadingTest(outH5File)

    # input_folder = os.path.join(PARTIAL_FACE_ROOT_FOLDER, "partial_scan/raw/test")
    # input_folder = PARTIAL_FACE_TEST_DATA_SAMPLE_FOLDER
# print('input_folder:', input_folder)

# output_folder = os.path.join(PARTIAL_FACE_ROOT_FOLDER, "partial_scan/raw/test_n")

# batch_normalization(input_folder, output_folder)
# batch_generate_dataFile_quaternion(input_folder, point_num=1024, channel=3, istrain=False)

# data = single_data_sampling(os.path.join(HOME_DIR,'einstein_noise.xyz'),patch_size=5120,normalization=False)
# np.savetxt(os.path.join(HOME_DIR,'einstein_gt.xyz'),data, fmt='%0.6f')
# print('Done!')
# filename = os.path.join(BASE_DIR,'data/quaternion_32_8_test.h5')
# loadingTest(filename)
