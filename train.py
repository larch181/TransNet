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
import shutil
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
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--num_patch', type=int, default=8, help='Patch Pair Number [default: 32]')
parser.add_argument('--type_patch', type=int, default=2, help='Patch Type Number [default: 2]')
parser.add_argument('--distance_theta', type=float, default=10, help='Threshold for matching distance')
parser.add_argument('--max_epoch', type=int, default=3001, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_PATCH = FLAGS.num_patch
TYPE_PATCH = FLAGS.type_patch
THETA = FLAGS.distance_theta
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
# os.system('cp %s.py %s' % (FLAGS.model, LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

print(FLAGS.normal)
TRAIN_DATASET = DataPatcher.TransNetH5Dataset(
    os.path.join(ROOT_DIR, 'data/train_files.txt'), batch_size=BATCH_SIZE,
    npoints=NUM_POINT, num_patch=NUM_PATCH, type_patch=TYPE_PATCH, shuffle=True, isTrain=True, usenormal=FLAGS.normal)
TEST_DATASET = DataPatcher.TransNetH5Dataset(
    os.path.join(ROOT_DIR, 'data/test_files.txt'), batch_size=BATCH_SIZE, num_patch=NUM_PATCH,
    npoints=NUM_POINT, type_patch=TYPE_PATCH, shuffle=False, isTrain=False, usenormal=FLAGS.normal)

TRAIN_FOLDER = os.path.join(LOG_DIR, 'train')
TEST_FOLDER = os.path.join(LOG_DIR, 'test')
IMAGE_FOLDER = os.path.join(ROOT_DIR, LOG_DIR,'images')

if os.path.exists(TRAIN_FOLDER):
    FileOperator.CleanDir(TRAIN_FOLDER)
else: os.mkdir(TRAIN_FOLDER)
if os.path.exists(TEST_FOLDER):
    FileOperator.CleanDir(TEST_FOLDER)
else: os.mkdir(TEST_FOLDER)
if os.path.exists(IMAGE_FOLDER):
    FileOperator.CleanDir(IMAGE_FOLDER)
else: os.mkdir(IMAGE_FOLDER)

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, match_pair = MODEL.placeholder_inputs(BATCH_SIZE, NUM_PATCH, NUM_POINT,TRAIN_DATASET.num_channel())

            is_training_pl = tf.placeholder(tf.bool, shape=())
            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # input = tf.squeeze(pointclouds_pl[:, 0, :, :])
            # output = MODEL.get_model(input, is_training_pl, bn_decay=bn_decay)
            # output_features = tf.expand_dims(output, axis=1)
            # with tf.variable_scope(tf.get_variable_scope(),reuse=True):
            #     for i in range(NUM_PATCH - 1):
            #         input = tf.squeeze(pointclouds_pl[:, i + 1, :, :])
            #         output = MODEL.get_model(input, is_training_pl, bn_decay=bn_decay)
            #         output_features = tf.concat([output_features, tf.expand_dims(output, axis=1)], axis=1)

            output = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay,channel=TRAIN_DATASET.num_channel())

            output_features = tf.reshape(output, [BATCH_SIZE, NUM_PATCH, -1])
            loss_util.get_N_tuple_loss(output_features, match_pair, THETA)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # train_op = optimizer.minimize(loss)
                train_op = optimizer.minimize(total_loss, global_step=batch)

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

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TRAIN_FOLDER, sess.graph)
        test_writer = tf.summary.FileWriter(TEST_FOLDER, sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl' : pointclouds_pl,
               'labels_pl'      : labels_pl,
               'match_pair'     : match_pair,
               'output_features': output_features,
               'is_training_pl' : is_training_pl,
               'loss'           : total_loss,
               'train_op'       : train_op,
               'merged'         : merged,
               'step'           : batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            evaluate_one_epoch(sess, ops, test_writer)

            # save_patch_feature(sess, ops)

            # if (epoch % 10 == 49):
            #     eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 9:
                # evaluate_one_epoch(sess, ops, test_writer)
                # save_patch_feature(sess, ops, TRAIN_DATASET)
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE * NUM_PATCH, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_labels = np.zeros((BATCH_SIZE, NUM_PATCH))
    cur_batch_match_pairs = np.zeros((BATCH_SIZE, NUM_PATCH, NUM_PATCH))

    loss_sum = 0
    batch_idx = 0
    print("Training")
    while TRAIN_DATASET.has_next_batch():

        batch_data, batch_label, batch_match_pairs = TRAIN_DATASET.next_batch(augment=True)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_labels[0:bsize, ...] = batch_label
        cur_batch_match_pairs[0:bsize, ...] = batch_match_pairs
        # print(batch_match_pairs)

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']     : cur_batch_labels,
                     ops['match_pair']    : cur_batch_match_pairs,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, batch_features = sess.run([ops['merged'], ops['step'],
                                                               ops['train_op'], ops['loss'], ops['output_features']],
                                                              feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        # batch_features = np.squeeze(np.array(batch_features)).reshape([BATCH_SIZE * NUM_PATCH, -1])

        # pairDistance = np.squeeze(CommonTools.get_pairwise_distance(batch_features))

        # print('max =',np.max(pairDistance),'min = ',np.min(pairDistance))
        # pairDistance = CommonTools.Normalize(pairDistance)

        loss_sum += loss_val
        if (batch_idx + 1) % 20 == 0:
            # log_string('loss: %f' % (loss_val))
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / 20))
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET._reset()

def reduce_sysmetric(data):
    _len = data.shape[0]
    for i in range(_len):
        for j in range(_len):
            if i==j:
                data[i,j] = 1
            if i==_len-1-j:
                data[i, j] = 1
    return data


def evaluate_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    log_string(str(datetime.now()))

    _batch_size = BATCH_SIZE * NUM_PATCH
    _totalDataLen = TEST_DATASET.get_total_len()
    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE * NUM_PATCH, NUM_POINT, TEST_DATASET.num_channel()))
    cur_batch_labels = np.zeros((BATCH_SIZE, NUM_PATCH))
    cur_batch_match_pairs = np.zeros((BATCH_SIZE, NUM_PATCH, NUM_PATCH))

    output_features = np.zeros([_totalDataLen, 64])

    #print(output_features.shape)
    loss_sum = 0
    batch_idx = 0

    print('Testing')
    while TEST_DATASET.has_next_batch():

        batch_data, batch_label, batch_match_pairs = TEST_DATASET.next_batch(augment=True)
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

        summary, step, loss_val, batch_features = sess.run([ops['merged'], ops['step'],
                                                            ops['loss'], ops['output_features']],
                                                           feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        #loss_val, batch_features = sess.run([ops['loss'], ops['output_features']], feed_dict=feed_dict)

        batch_features = np.squeeze(np.array(batch_features)).reshape([BATCH_SIZE * NUM_PATCH, -1])

        #pairDistance = np.squeeze(CommonTools.get_pairwise_distance(batch_features))
        #print('max =',np.max(pairDistance),'min = ',np.min(pairDistance))

        #pairDistance = CommonTools.Normalize(pairDistance)

        start_idx = batch_idx * _batch_size
        end_idx = (batch_idx + 1) * _batch_size
        output_features[start_idx:end_idx, :] = batch_features
        loss_sum += loss_val
        if (batch_idx + 1) % 20 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('Testing mean loss: %f' % (loss_sum / 20))
            loss_sum = 0
        batch_idx += 1

    output_features = PointCloudOperator.shuffle_batch_data2(output_features,None)
    pairDistance = np.squeeze(CommonTools.get_pairwise_distance(output_features))
    pairDistance = CommonTools.Normalize(pairDistance)
    pairDistance = np.expand_dims(pairDistance, axis=-1)
    #paddingData = np.zeros(pairDistance.shape)
    # paddingData = pairDistance.copy()
    # paddingData = reduce_sysmetric(np.squeeze(paddingData))
    # paddingData = np.expand_dims(paddingData, axis=-1)
    imageData = np.concatenate([1 - pairDistance, 1 - pairDistance, 1 - pairDistance], axis=-1)

    from PIL import Image
    img = Image.fromarray(np.uint8(imageData * 255.0))
    img_filename = '%s.jpg' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    img.save(os.path.join(IMAGE_FOLDER,img_filename))
    #img.save('log/images/%s' % (img_filename))
    _, whole_label = TEST_DATASET.get_whole_batch()
    # print(pairDistance.shape)

    leng = batch_features.shape[0]
    with open('log/features.txt', 'w') as file:
        for i in range(leng):
            file.write('label = ')
            file.write(str(whole_label[i]) + '_%d' % (2))
            file.write('\t ')
            file.write(str(batch_features[i, :]))
            file.write('\n')

    TEST_DATASET._reset()


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
