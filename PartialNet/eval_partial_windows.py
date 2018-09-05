'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
POINTCLOUD_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'partialNet'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))
import model_utils

import PartialDataPatcher
import pc_util
import Tools3D
import PointCloudOperator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='PartialNet', help='Model name [default: pointnet_trans]')
parser.add_argument('--log_dir', default='log/PartialNet11/', help='Log dir [default: log]')
parser.add_argument('--model_path', default='log/PartialNet7/model.ckpt')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL_PATH = os.path.join(ROOT_DIR,FLAGS.model_path)

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'PartialNet', FLAGS.model + '.py')
LOG_DIR = os.path.join(ROOT_DIR, FLAGS.log_dir)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
DUMP_DIR = os.path.join(LOG_DIR, 'eval')
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
# os.system('cp train_partial.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 40

FILE_TYPE = 'quaternion'

EVAL_DATASET = PartialDataPatcher.TransNetH5Dataset(
    os.path.join(BASE_DIR, 'data/eval_files.txt'), batch_size=BATCH_SIZE,
    npoints=NUM_POINT, shuffle=False)

POINT_CLOUD_FOLDER = os.path.dirname(ROOT_DIR)



filename = os.path.join(POINT_CLOUD_FOLDER, 'Dataset/Data/einstein_gt_s.xyz')
pointcloud_gt_s = np.loadtxt(filename)

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
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
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
            pointclouds_pl, pointclouds_angle, pointclouds_gt = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # # Note the global_step=batch parameter to minimize.
            # # That tells the optimizer to helpfully increment the 'batch' parameter
            # # for you every time it trains.
            # batch = tf.get_variable('batch', [],
            #                         initializer=tf.constant_initializer(0), trainable=False)
            # bn_decay = get_bn_decay(batch)
            # tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred_angle = MODEL.get_model(pointclouds_pl, is_training_pl)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        # config.gpu_options.visible_device_list = '1'
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl'   : pointclouds_pl,
               'is_training_pl'   : is_training_pl,
               'pred_angle'       : pred_angle}

        #save_one_epoch(sess, ops, test_writer)
        eval_real_epoch(sess, ops, test_writer)

def eval_real_epoch(sess, ops, test_writer):
    PC_PATH = os.path.join(POINTCLOUD_DIR, 'Dataset/Data/partialface/real_scan/')
    INPUT_FOLDER = os.path.join(PC_PATH, 'sampled')
    OUTPUT_FOLDER = os.path.join(PC_PATH, 'pred')

    from glob import glob
    is_training = False


    samples = glob(INPUT_FOLDER + "/*.xyz")
    samples.sort(reverse=False)
    total_num = len(samples)
    for i in range(total_num):
        filename = samples[i].split('\\')[-1].replace('.xyz','')
        print(filename)
        pointclouds_pl = np.loadtxt(samples[i])
        pointclouds_pl = np.expand_dims(pointclouds_pl,axis=0)

        feed_dict = {ops['pointclouds_pl']   : pointclouds_pl,
                     ops['is_training_pl']   : is_training, }
        # loss_val, pred_angle = sess.run([ops['loss'], ops['pred_angle']], feed_dict=feed_dict)
        pred_angle = sess.run([ops['pred_angle']], feed_dict=feed_dict)
        pred_angle = np.squeeze(pred_angle)
        print(pred_angle.shape)

        print(pred_angle)
        transform_xyz = Tools3D.quaternion_To_rotation_matrix(np.squeeze(pred_angle))
        transform_xyz = np.array(transform_xyz)
        print(transform_xyz)
        np.savetxt(os.path.join(INPUT_FOLDER,filename+'.txt'), np.expand_dims(pred_angle,axis=0), fmt='%0.6f')

        point_cloud_transformed = np.matmul( pointcloud_gt_s, transform_xyz)

        # _point_cloud_transformed = sess.run(point_cloud_transformed, feed_dict=feed_dict)
        img_filename = '%d_coarse.png' % (i)  # , datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        img_filename = os.path.join(OUTPUT_FOLDER, img_filename)

        point_input = np.squeeze(pointclouds_pl)
        points_gt = np.squeeze(pointcloud_gt_s)
        points_rotated = np.squeeze(point_cloud_transformed)

        print(points_gt.shape,points_rotated.shape,point_cloud_transformed.shape)

        info = 'Nothing'

        pc_util.point_cloud_three_points(point_input, point_cloud_transformed, point_cloud_transformed, img_filename, info)


if __name__ == "__main__":
    print('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
