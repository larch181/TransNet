#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))

import FileOperator
import loss_util
import DataPatcher
import CommonTools
import PointCloudOperator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='transnet_basic', help='Model name [default: pointnet_trans]')
parser.add_argument('--log_dir', default='log/transnet@normal', help='Log dir [default: log]')
parser.add_argument('--model_path', default='log/transnet@normal/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--num_patch', type=int, default=8, help='Patch Pair Number [default: 32]')
parser.add_argument('--type_patch', type=int, default=2, help='Patch Type Number [default: 2]')
parser.add_argument('--distance_theta', type=float, default=10, help='Threshold for matching distance')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information')

FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_PATCH = FLAGS.num_patch
TYPE_PATCH = FLAGS.type_patch
THETA = FLAGS.distance_theta
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
# os.system('cp %s.py %s' % (FLAGS.model, LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

LOG_FEATURE_FILE = os.path.join(LOG_DIR,'log_features.txt')

HOSTNAME = socket.gethostname()

print(FLAGS.normal)
TRAIN_DATASET = DataPatcher.TransNetH5Dataset(
    os.path.join(ROOT_DIR, 'data/train_files.txt'), batch_size=BATCH_SIZE,
    npoints=NUM_POINT, num_patch=NUM_PATCH, type_patch=TYPE_PATCH, shuffle=False, isTrain=False, usenormal=FLAGS.normal)
TEST_DATASET = DataPatcher.TransNetH5Dataset(
    os.path.join(ROOT_DIR, 'data/test_files.txt'), batch_size=BATCH_SIZE, num_patch=NUM_PATCH,
    npoints=NUM_POINT, type_patch=TYPE_PATCH, shuffle=False, isTrain=False, usenormal=FLAGS.normal)

IMAGE_FOLDER = os.path.join(ROOT_DIR, LOG_DIR,'dump')
LOG_EVALUATE = os.path.join(ROOT_DIR, LOG_DIR,'eval')
if os.path.exists(IMAGE_FOLDER):
    FileOperator.CleanDir(IMAGE_FOLDER)
else: os.mkdir(IMAGE_FOLDER)
if os.path.exists(LOG_EVALUATE):
    FileOperator.CleanDir(LOG_EVALUATE)
else: os.mkdir(LOG_EVALUATE)

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, match_pair = MODEL.placeholder_inputs(BATCH_SIZE, NUM_PATCH, NUM_POINT,TRAIN_DATASET.num_channel())

            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)

            output = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay,channel=TRAIN_DATASET.num_channel())

            output_features = tf.reshape(output, [BATCH_SIZE, NUM_PATCH, -1])
            loss_util.get_N_tuple_loss(output_features, match_pair, THETA)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = '0'
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl' : pointclouds_pl,
               'labels_pl'      : labels_pl,
               'match_pair'     : match_pair,
               'output_features': output_features,
               'is_training_pl' : is_training_pl,
               'loss'           : total_loss}

        evaluate_one_epoch(sess, ops,TRAIN_DATASET)
        evaluate_one_epoch(sess, ops,TEST_DATASET)



def evaluate_one_epoch(sess, ops,DATASET):
    """ ops: dict mapping from string to tf ops """
    """ ops: dict mapping from string to tf ops """
    is_training = False

    log_string(str(datetime.now()))

    _batch_size = BATCH_SIZE * NUM_PATCH
    _totalDataLen = DATASET.get_total_len()
    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE * NUM_PATCH, NUM_POINT, DATASET.num_channel()))
    cur_batch_labels = np.zeros((BATCH_SIZE, NUM_PATCH))
    cur_batch_match_pairs = np.zeros((BATCH_SIZE, NUM_PATCH, NUM_PATCH))

    output_features = np.zeros([_totalDataLen, 64])

    # print(output_features.shape)
    loss_sum = 0
    batch_idx = 0

    print('Testing')
    while DATASET.has_next_batch():

        batch_data, batch_label, batch_match_pairs = DATASET.next_batch(augment=True)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_labels[0:bsize, ...] = batch_label
        cur_batch_match_pairs[0:bsize, ...] = batch_match_pairs
        # print(cur_batch_data[0,0:4,0])
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']     : cur_batch_labels,
                     ops['match_pair']    : cur_batch_match_pairs,
                     ops['is_training_pl']: is_training, }

        loss_val, batch_features = sess.run([ops['loss'], ops['output_features']], feed_dict=feed_dict)

        batch_features = np.squeeze(np.array(batch_features)).reshape([BATCH_SIZE * NUM_PATCH, -1])

        # pairDistance = np.squeeze(CommonTools.get_pairwise_distance(batch_features))
        # print('max =',np.max(pairDistance),'min = ',np.min(pairDistance))

        # pairDistance = CommonTools.Normalize(pairDistance)

        start_idx = batch_idx * _batch_size
        end_idx = (batch_idx + 1) * _batch_size
        output_features[start_idx:end_idx, :] = batch_features
        loss_sum += loss_val
        if (batch_idx + 1) % 20 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('Testing mean loss: %f' % (loss_sum / 20))
            loss_sum = 0
        batch_idx += 1

    output_features = PointCloudOperator.shuffle_batch_data2(output_features, None)
    pairDistance = np.squeeze(CommonTools.get_pairwise_distance(output_features))
    pairDistance = CommonTools.Normalize(pairDistance)
    pairDistance = np.expand_dims(pairDistance, axis=-1)
    # paddingData = np.zeros(pairDistance.shape)
    # paddingData = pairDistance.copy()
    # paddingData = reduce_sysmetric(np.squeeze(paddingData))
    # paddingData = np.expand_dims(paddingData, axis=-1)
    imageData = np.concatenate([1 - pairDistance, 1 - pairDistance, 1 - pairDistance], axis=-1)

    from PIL import Image
    img = Image.fromarray(np.uint8(imageData * 255.0))
    img_filename = '%s.jpg' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    img.save(os.path.join(LOG_EVALUATE, img_filename))
    # img.save('log/images/%s' % (img_filename))
    #_, whole_label = TEST_DATASET.get_whole_batch()
    # print(pairDistance.shape)

    #leng = batch_features.shape[0]

    np.savetxt(LOG_FEATURE_FILE, batch_features, fmt='%0.6f', delimiter="\n")
    # with open(LOG_FEATURE_FILE, 'w') as file:
    #     for i in range(leng):
    #         file.write('label = ')
    #         file.write(str(whole_label[i]) + '_%d' % (2))
    #         file.write('\t ')
    #         file.write(str(batch_features[i, :]))
    #         file.write('\n')

    TEST_DATASET._reset()


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
