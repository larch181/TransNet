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
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'partialNet'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))
import model_utils

import loss_util
import PartialDataPatcher
import pc_util
import Tools3D
import PointCloudOperator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='PartialNet', help='Model name [default: pointnet_trans]')
parser.add_argument('--log_dir', default='log/PartialNet4/', help='Log dir [default: log]')
parser.add_argument('--model_path', default='log/PartialNet4/model.ckpt')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 16]')
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


def load_gt_xyz_file(batch_size, num_point, filename=None):
    if filename == None:
        filename = os.path.join(POINT_CLOUD_FOLDER, 'Dataset/Data/einstein_gt.xyz')
    print(filename)
    data_T = np.loadtxt(filename)
    data = data_T[:, 0:3]
    pointcloud_gt = np.repeat(data[np.newaxis, ...], batch_size, axis=0)

    filename = os.path.join(POINT_CLOUD_FOLDER, 'Dataset/Data/einstein_gt_s.xyz')
    data_S = np.loadtxt(filename)
    data2 = data_S[:, 0:3]
    pointcloud_gt_s = np.repeat(data2[np.newaxis, ...], batch_size, axis=0)

    return pointcloud_gt, pointcloud_gt_s


pointclouds_gt_val, pointcloud_gt_s = load_gt_xyz_file(BATCH_SIZE, NUM_POINT)
print(pointclouds_gt_val.shape[1]/pointcloud_gt_s.shape[1])
RATIO = int(pointclouds_gt_val.shape[1]/pointcloud_gt_s.shape[1])

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

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred_angle = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)

            # loss_util.get_transform_loss2(pred_angle, pointclouds_angle)
            cd_dists, knn_dists = loss_util.get_partialNet_loss(pred_angle, pointclouds_angle, pointclouds_pl,
                                                                pointclouds_gt)

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
            # train_op = optimizer.minimize(total_loss, global_step=batch)

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
               'pointclouds_gt'   : pointclouds_gt,
               'pointclouds_angle': pointclouds_angle,
               'is_training_pl'   : is_training_pl,
               'pred_angle'       : pred_angle,
               'loss'             : total_loss,
               'train_op'         : train_op,
               'merged'           : merged,
               'step'             : batch,
               'cd_dists'         : cd_dists,
               'knn_dists'        : knn_dists}

        save_one_epoch(sess, ops, test_writer)
        #eval_one_epoch(sess, ops, test_writer)


def save_one_epoch(sess, ops, test_writer):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, EVAL_DATASET.num_channel()))
    cur_batch_angle = np.zeros((BATCH_SIZE, 4))
    cur_batch_label = np.zeros((BATCH_SIZE))

    loss_sum = 0
    batch_idx = 0
    partial_data, gt_quat, labels = EVAL_DATASET.get_all_data()
    rotated_data = np.zeros(partial_data.shape)
    pred_quat = np.zeros(gt_quat.shape)
    index=0
    while EVAL_DATASET.has_next_batch():
        batch_data, batch_angel, batch_data_label = EVAL_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]

        print('Batch: %03d, batch size: %d' % (batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_angle[0:bsize, ...] = batch_angel
        cur_batch_label[0:bsize] = batch_data_label
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_angle[0:bsize, ...] = batch_angel
        feed_dict = {ops['pointclouds_pl']   : cur_batch_data,
                     ops['pointclouds_gt']   : pointclouds_gt_val,
                     ops['pointclouds_angle']: cur_batch_angle,
                     ops['is_training_pl']   : is_training, }
        # loss_val, pred_angle = sess.run([ops['loss'], ops['pred_angle']], feed_dict=feed_dict)
        summary, step, loss_val, pred_angle, cd_dists, knn_dists = sess.run(
            [ops['merged'], ops['step'], ops['loss'], ops['pred_angle'], ops['cd_dists'], ops['knn_dists']],
            feed_dict=feed_dict)

        test_writer.add_summary(summary, step)

        loss_sum += loss_val
        batch_idx += 1

        #print(pred_angle.shape)

        start_idx = (batch_idx - 1) * BATCH_SIZE
        end_idx = min(batch_idx * BATCH_SIZE, partial_data.shape[0])



        transform_xyz = Tools3D.batch_quaternion2mat(pred_angle)
        point_cloud_transformed = np.matmul(pointcloud_gt_s, transform_xyz)

        for i in range(bsize):
            np.savetxt(os.path.join(BASE_DIR, 'test/%d.xyz'%(index)), point_cloud_transformed[i,...], fmt='%0.6f')
            index = index + 1

        rotated_data[start_idx:end_idx, :, :] = point_cloud_transformed[0:bsize, ...]
        pred_quat[start_idx:end_idx, :] = pred_angle[0:bsize, :]

        # _point_cloud_transformed = sess.run(point_cloud_transformed, feed_dict=feed_dict)
    filename = os.path.join(ROOT_DIR, 'TuneNet/data/quaternion_512_32_train4.h5')
    #PointCloudOperator.save_fineTune_data(filename, partial_data, rotated_data, pred_quat, gt_quat, labels)

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    # EVAL_DATASET.reset()


def eval_one_epoch(sess, ops, test_writer):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, EVAL_DATASET.num_channel()))
    cur_batch_angle = np.zeros((BATCH_SIZE, 4))
    cur_batch_label = np.zeros((BATCH_SIZE))

    loss_sum = 0
    batch_idx = 0

    while EVAL_DATASET.has_next_batch():
        batch_data, batch_angel, batch_data_label = EVAL_DATASET.next_batch(augment=True)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d' % (batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_angle[0:bsize, ...] = batch_angel
        cur_batch_label[0:bsize] = batch_data_label
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_angle[0:bsize, ...] = batch_angel
        feed_dict = {ops['pointclouds_pl']   : cur_batch_data,
                     ops['pointclouds_gt']   : pointclouds_gt_val,
                     ops['pointclouds_angle']: cur_batch_angle,
                     ops['is_training_pl']   : is_training, }
        # loss_val, pred_angle = sess.run([ops['loss'], ops['pred_angle']], feed_dict=feed_dict)
        summary, step, loss_val, pred_angle, cd_dists, knn_dists = sess.run(
            [ops['merged'], ops['step'], ops['loss'], ops['pred_angle'], ops['cd_dists'], ops['knn_dists']],
            feed_dict=feed_dict)

        test_writer.add_summary(summary, step)

        loss_sum += loss_val
        batch_idx += 1

        transform_xyz_input = Tools3D.batch_quaternion2mat(cur_batch_angle)

        transform_xyz = Tools3D.batch_quaternion2mat(pred_angle)
        point_cloud_transformed = np.matmul(pointcloud_gt_s, transform_xyz)
        point_cloud_gt_transformed = np.matmul(pointcloud_gt_s, transform_xyz_input)

        # _point_cloud_transformed = sess.run(point_cloud_transformed, feed_dict=feed_dict)

        for i in range(bsize):
            index = cur_batch_label[i]
            img_filename = '%d.png' % (index)  # , datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            img_filename = os.path.join(DUMP_DIR, img_filename)

            points_gt = np.squeeze(point_cloud_gt_transformed[i, :, :])
            points_rotated = np.squeeze(cur_batch_data[i, :, :])
            points_align = np.squeeze(point_cloud_transformed[i, :, :])
            # print(points_rotated.shape)
            # print(point_cloud_transformed.shape)
            # print(points_align.shape)

            info_input = pc_util.log_visu_vec('Input Data %d' % (index), cur_batch_angle[i, :])
            pre_angle = pc_util.log_visu_vec('Pred Data', pred_angle[i, :])
            matrix_input = pc_util.log_visu_matrix('Input Matrix', np.squeeze(transform_xyz_input[i, :, :]))
            matrix_pred = pc_util.log_visu_matrix('Pred Matrix', np.squeeze((transform_xyz[i, :, :])))
            # print(point_cloud_transformed[i,:,:].shape)
            cd_loss = cd_dists[i]
            knn_loss = knn_dists[i]
            matloss = np.sum(np.square(transform_xyz_input[i, :, :] - transform_xyz[i, :, :])) / 2
            vecloss = np.sum(np.square(pred_angle[i, :] - cur_batch_angle[i, :])) / 2

            loss_cd = pc_util.log_visu_loss('CD  Loss', cd_loss)
            loss_knn = pc_util.log_visu_loss('KNN  Loss', knn_loss)
            loss_mat = pc_util.log_visu_loss('MAT Loss', matloss)
            vec_mat = pc_util.log_visu_loss('VEC Loss', vecloss)

            info = info_input + pre_angle + matrix_input + matrix_pred + vec_mat + loss_mat + loss_cd + loss_knn

            pc_util.point_cloud_three_points(points_rotated, points_gt, points_align, img_filename, info)

            # scipy.misc.imsave(img_filename, output_img)

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    # EVAL_DATASET.reset()


if __name__ == "__main__":
    print('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
