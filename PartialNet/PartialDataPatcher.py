#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

import os
import sys
import numpy as np
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
import PointCloudOperator


def shuffle_data(data,numbers, angles, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...],numbers[idx], angles[idx, ...], labels[idx], idx

def load_h5(h5_filename):
    with h5py.File(h5_filename) as f:
        data = f['data'][:]
        angles = f['angles'][:]
        numbers = f['numbers'][:]
        label = f['label'][:]

    print('loading data:', data.shape)
    return data,numbers, angles, label

class TransNetH5Dataset(object):
    def __init__(self, list_filename, batch_size=32, npoints=1024, shuffle=True):
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.h5_files = PointCloudOperator.getDataFiles(self.list_filename)
        self.initParameter()

    def initParameter(self):

        self.current_data = None
        self.current_data_numbers = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0
        self.file_idxs = np.arange(0, len(self.h5_files))
        self._load_data_file(self._get_data_filename())

    def reset(self):
        self.batch_idx = 0
        self._shuffle_all()

    def _augment_batch_data(self, batch_data,batch_data_numbers,sess=None):

        simulated_data = PointCloudOperator.pointcloud_simulation(batch_data,batch_data_numbers,patch_size= self.npoints,sess=sess)
        simulated_data = PointCloudOperator.shift_point_cloud(simulated_data)
        simulated_data = PointCloudOperator.jitter_point_cloud(simulated_data)
        simulated_data = PointCloudOperator.random_scale_point_cloud(simulated_data)
        simulated_data = PointCloudOperator.rotate_perturbation_point_cloud(simulated_data)

        return simulated_data

    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _shuffle_all(self):
        if self.current_data is not None:
            self.current_data, self.current_data_numbers,self.current_angles, self.current_label, _ = shuffle_data(self.current_data,self.current_data_numbers,
                                                                                         self.current_angles,
                                                                                         self.current_label)
    def _load_data_file(self, filename):
        #print(filename)
        self.current_data,self.current_data_numbers,self.current_angles, self.current_label = load_h5(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_all()

    def _has_next_batch_in_file(self):
        return self.batch_idx * self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        # if (self.current_data is None) or (not self._has_next_batch_in_file()):
        #     if self.current_file_idx >= len(self.h5_files):
        #         return False
        #     self._load_data_file(self._get_data_filename())
        #     self.batch_idx = 0
        #     self.current_file_idx += 1
        return self._has_next_batch_in_file()
    def get_data_length(self):
        return self.current_data.shape
    def get_all_data(self):
        return self.current_data,self.current_angles,self.current_label

    def next_batch(self, augment=False,sess=None):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self.current_data.shape[0])

        batch_data = self.current_data[start_idx:end_idx, :, :].copy()
        batch_data_numbers = self.current_data_numbers[start_idx:end_idx].copy()
        batch_angles = self.current_angles[start_idx:end_idx, :].copy()
        batch_label = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data,batch_data_numbers,sess=sess)
        return batch_data, batch_angles, batch_label


if __name__ == '__main__':

    FILE_TYPE = 'rotationvector'
    BATCH_SIZE = 32
    NUM_POINT = 1024
    TRAIN_DATASET = TransNetH5Dataset(
        os.path.join(BASE_DIR, 'data/train_files.txt'), batch_size=BATCH_SIZE,
        npoints=NUM_POINT, shuffle=True)

    print(TRAIN_DATASET.has_next_batch())
    batch_data, batch_angle, _ = TRAIN_DATASET.next_batch(augment=True)

    print(batch_angle)
    print(batch_angle.shape)
