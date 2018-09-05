#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import PointCloudOperator

ROTATION_DATABASE_FILE = os.path.join(ROOT_DIR, 'data/quaternion_1000_50_train.xyz')

ROTATION_DATABASE = np.loadtxt(ROTATION_DATABASE_FILE)


class TransNetH5Dataset(object):
    def __init__(self, list_filename, batch_size=32, num_patch=32, type_patch=2, npoints=1024, shuffle=True,usenormal=False,isTrain=True):
        self.list_filename = list_filename
        self.raw_batch_size = batch_size
        self.rawpoints = npoints
        self.num_patch = num_patch
        self.sample_num = type_patch  # number of patch type [default:2]

        self.npoints = self.sample_num * self.rawpoints
        self.batch_size = self.raw_batch_size * int(self.num_patch / self.sample_num)
        self.usenormal = usenormal
        self.shuffle = shuffle
        self.h5_files = PointCloudOperator.getDataFiles(self.list_filename)
        self.isTrain = isTrain
        self.initParameter()

    def initParameter(self):

        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.whole_data = None
        self.whole_label = None
        self.batch_idx = 0
        self.file_idxs = np.arange(0, len(self.h5_files))
        self._load_data_file(self._get_data_filename())

    def _reset(self):
        ''' reset order of h5 files '''
        self.batch_idx = 0

        self._shuffle_all()

        #if self.shuffle: np.random.shuffle(self.file_idxs)

    def _shuffle_all(self):
        if self.current_data is not None:
            self.current_data, self.current_label, _ = PointCloudOperator.shuffle_data(self.current_data,
                                                                                       self.current_label)

    def _augment_batch_data(self, batch_data):

        shifted_data = PointCloudOperator.shift_point_cloud(batch_data)
        jittered_data = PointCloudOperator.jitter_point_cloud(shifted_data)
        rotated_data = PointCloudOperator.random_rotate_point_cloud(jittered_data, ROTATION_DATABASE,hasnormal=self.usenormal)
        return PointCloudOperator.shuffle_points(rotated_data)

    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        self.current_data, self.current_label = PointCloudOperator.load_h5(os.path.join(ROOT_DIR, filename),
                                                                           pointNum=self.npoints)

        if not self.usenormal:
            self.current_data = self.current_data[:,:,:3]

        self.current_label = np.squeeze(self.current_label)

        self.whole_data = self.current_data.copy().reshape([-1, self.rawpoints, 3])
        self.whole_label = self.current_label.copy().reshape(-1)

        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_all()

    def _has_next_batch_in_file(self):
        return self.batch_idx * self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        if self.usenormal:
            return 6
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        # if (self.current_data is None) or (not self._has_next_batch_in_file()):
        #     if self.current_file_idx >= len(self.h5_files):
        #         return False
        #     print('reload data')
        #
        #     self._load_data_file(self._get_data_filename())
        #     self.batch_idx = 0
        #     self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def _cal_batch_matrix(self, batch_label):
        batch_match_matrix = np.zeros([self.raw_batch_size, self.num_patch, self.num_patch])

        for i in range(self.raw_batch_size):
            label_base = np.array(batch_label[0])[np.newaxis, :].repeat(self.num_patch, axis=0)
            match_matrix = 1 - np.sign(np.abs(label_base - label_base.T))
            batch_match_matrix[i, :, :] = match_matrix

        return batch_match_matrix

    def get_whole_batch(self):
        return self.whole_data, self.whole_label

    def get_total_len(self):
        return self.whole_data.shape[0]


    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx

        batch_data = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        batch_label = self.current_label[start_idx:end_idx, :].copy()


        batch_data = batch_data.reshape([self.batch_size * self.sample_num, self.rawpoints, -1])
        batch_label = batch_label.reshape([self.batch_size * self.sample_num, -1])

        self.batch_idx = self.batch_idx + 1

        if augment: batch_data = self._augment_batch_data(batch_data)


        if self.isTrain:
            batch_data = batch_data.reshape([self.raw_batch_size, self.num_patch, self.rawpoints, -1])
            batch_label = batch_label.reshape([self.raw_batch_size, self.num_patch])

            batch_data, batch_label = PointCloudOperator.shuffle_batch_data(batch_data, batch_label)

            batch_data = batch_data.reshape([self.batch_size * self.sample_num, self.rawpoints, -1])
        else:

            #batch_data, batch_label = PointCloudOperator.shuffle_batch_data2(batch_data, batch_label)

            batch_label = batch_label.reshape([self.raw_batch_size, self.num_patch])

        batch_match_matrix = self._cal_batch_matrix(batch_label)
        #print(np.sum(batch_match_matrix, axis=(1, 2)))

        return batch_data, batch_label, batch_match_matrix


if __name__ == '__main__':
    BATCH_SIZE = 4
    NUM_POINT = 1024
    NUM_PATCH = 16
    TRAIN_DATASET = TransNetH5Dataset(
        os.path.join(ROOT_DIR, 'data/train_files.txt'), batch_size=BATCH_SIZE,
        npoints=NUM_POINT, num_patch=NUM_PATCH, shuffle=True)

    print(os.path.join(ROOT_DIR, 'data/train_files.txt'))
    print(TRAIN_DATASET.has_next_batch())
    for i in range(2):
        print('epoch %d'%(i))
        while TRAIN_DATASET.has_next_batch():
             batch_data, batch_label, _ = TRAIN_DATASET.next_batch(augment=True)
        TRAIN_DATASET._reset()

    # print(batch_data)
    print(batch_data.shape)

    print(batch_label[0])
    # print(batch_label[0].shape)
