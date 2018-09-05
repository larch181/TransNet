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
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'partialNet'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))

import loss_util
import PartialDataPatcher
import pc_util
import Tools3D
import model_utils
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='How many gpus to use [default: 1]')
parser.add_argument('--model', default='PartialNet', help='Model name [default: pointnet_trans]')
parser.add_argument('--log_dir', default='log/PartialNet', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

NUM_GPUS = FLAGS.num_gpus
BATCH_SIZE = FLAGS.batch_size
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = int(BATCH_SIZE / NUM_GPUS)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'partialNet', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
DUMP_DIR = os.path.join(LOG_DIR, 'images')
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train_partial.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 40

FILE_TYPE = 'quaternion'

assert (NUM_POINT <= 2048)
TRAIN_DATASET = PartialDataPatcher.TransNetH5Dataset(
    os.path.join(BASE_DIR, 'partialNet/data/train_files.txt'), batch_size=BATCH_SIZE,
    npoints=NUM_POINT, shuffle=True)
TEST_DATASET = PartialDataPatcher.TransNetH5Dataset(
    os.path.join(BASE_DIR, 'partialNet/data/test_files.txt'), batch_size=BATCH_SIZE,
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

    return pointcloud_gt,pointcloud_gt_s


pointclouds_gt_val, pointcloud_gt_s = load_gt_xyz_file(BATCH_SIZE, NUM_POINT)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #for g, _ in grad_and_vars:
    for g, v in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

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
        with tf.device('/cpu:0'):
            pointclouds_pl, pointclouds_angle, pointclouds_gt = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # Get model and loss 
            MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            tower_grads = []
            pred_gpu = []
            emd_gpu=[]
            total_loss_gpu = []
            for i in range(NUM_GPUS):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d' % (i)), tf.name_scope('gpu_%d' % (i)) as scope:
                        # Evenly split input data to each GPU
                        pc_batch = tf.slice(pointclouds_pl,
                                            [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])
                        angle_batch = tf.slice(pointclouds_angle,
                                               [i * DEVICE_BATCH_SIZE,0], [DEVICE_BATCH_SIZE,-1])
                        gt_batch = tf.slice(pointclouds_gt,
                                            [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])

                        pred = MODEL.get_model(pc_batch,is_training=is_training_pl, bn_decay=bn_decay)
                        _, emd_dist = loss_util.get_transformnet_loss_quaternion(pred, angle_batch,
                                                                                 pc_batch,
                                                                                gt_batch)
                        losses = tf.get_collection('losses', scope)
                        total_loss = tf.add_n(losses, name='total_loss')
                        for l in losses + [total_loss]:
                            tf.summary.scalar(l.op.name, l)

                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)

                        pred_gpu.append(pred)
                        total_loss_gpu.append(total_loss)
                        emd_gpu.append(emd_dist)

            # Merge pred and losses from multiple GPUs
            pred_angle = tf.concat(pred_gpu, 0)
            emd_dist = tf.concat(emd_gpu, 0)
            total_loss = tf.reduce_mean(total_loss_gpu)
            # Get training operator
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=batch)


        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=6)

        # Create a session
        config = tf.ConfigProto()
        #config.gpu_options.visible_device_list = '0'
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl'   : pointclouds_pl,
               'pointclouds_gt'   : pointclouds_gt,
               'pointclouds_angle': pointclouds_angle,
               'is_training_pl'   : is_training_pl,
               'pred_angle'       : pred_angle,
               'loss'             : total_loss,
               'train_op'         : train_op,
               'merged'           : merged,
               'step'             : batch,
               'emd_dist'         : emd_dist}

        # restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(LOG_DIR)
        # global LOG_FOUT
        # if restore_epoch == 0:
        #     LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
        #     LOG_FOUT.write(str(socket.gethostname()) + '\n')
        #     LOG_FOUT.write(str(FLAGS) + '\n')
        # else:
        #     LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
        #     saver.restore(sess, checkpoint_path)

        restore_epoch=0
        for epoch in range(restore_epoch,MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            # eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                eval_one_epoch(sess, ops, test_writer)
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_angle = np.zeros((BATCH_SIZE, 4))

    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():

        batch_data, batch_angle, _ = TRAIN_DATASET.next_batch(augment=True)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_angle[0:bsize, ...] = batch_angle

        feed_dict = {ops['pointclouds_pl']   : cur_batch_data,
                     ops['pointclouds_gt']   : pointclouds_gt_val,
                     ops['pointclouds_angle']: cur_batch_angle,
                     ops['is_training_pl']   : is_training, }
        summary, step, _, loss_val = sess.run([ops['merged'], ops['step'],
                                               ops['train_op'], ops['loss']],
                                              feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        loss_sum += loss_val
        if (batch_idx + 1) % 50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / 50))
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()


def eval_one_epoch(sess, ops,test_writer):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TEST_DATASET.num_channel()))
    cur_batch_angle = np.zeros((BATCH_SIZE, 4))

    loss_sum = 0
    batch_idx = 0

    while TEST_DATASET.has_next_batch():
        batch_data, batch_angel, batch_data_raw = TEST_DATASET.next_batch(augment=True)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d' % (batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_angle[0:bsize, ...] = batch_angel

        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_angle[0:bsize, ...] = batch_angel
        feed_dict = {ops['pointclouds_pl']   : cur_batch_data,
                     ops['pointclouds_gt']   : pointclouds_gt_val,
                     ops['pointclouds_angle']: cur_batch_angle,
                     ops['is_training_pl']   : is_training, }
        # loss_val, pred_angle = sess.run([ops['loss'], ops['pred_angle']], feed_dict=feed_dict)
        summary, step,loss_val, pred_angle, emd_dist = sess.run([ops['merged'], ops['step'],ops['loss'], ops['pred_angle'], ops['emd_dist']],
                                                  feed_dict=feed_dict)

        test_writer.add_summary(summary, step)

        loss_sum += loss_val
        batch_idx += 1

        transform_xyz_input = Tools3D.batch_quaternion2mat(cur_batch_angle)

        transform_xyz = Tools3D.batch_quaternion2mat(pred_angle)
        point_cloud_transformed = np.matmul(pointcloud_gt_s, transform_xyz)

        # _point_cloud_transformed = sess.run(point_cloud_transformed, feed_dict=feed_dict)

        for i in range(bsize):
            img_filename = '%d.jpg' % (i+(batch_idx-1)*BATCH_SIZE)#, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            img_filename = os.path.join(DUMP_DIR, img_filename)

            points_gt = np.squeeze(pointcloud_gt_s[i, :, :])
            points_rotated = np.squeeze(cur_batch_data[i, :, :])
            points_align = np.squeeze(point_cloud_transformed[i, :, :])
            # print(points_rotated.shape)
            # print(point_cloud_transformed.shape)
            # print(points_align.shape)

            info_input = pc_util.log_visu_vec('Input Data', cur_batch_angle[i, :])
            pre_angle = pc_util.log_visu_vec('Pred Data', pred_angle[i, :])
            matrix_input = pc_util.log_visu_matrix('Input Matrix', np.squeeze(transform_xyz_input[i, :, :]))
            matrix_pred = pc_util.log_visu_matrix('Pred Matrix', np.squeeze((transform_xyz[i, :, :])))
            # print(point_cloud_transformed[i,:,:].shape)
            align_loss = emd_dist[i]  # , _ = loss_util.get_emd_loss2(point_cloud_transformed[i,:,:], cur_batch_data[i, :, :])
            matloss = np.sum(np.square(transform_xyz_input[i, :, :] - transform_xyz[i, :, :])) / 2
            vecloss = np.sum(np.square(pred_angle[i, :] - cur_batch_angle[i, :])) / 2
            norm_loss = np.abs(1 - np.sum(np.square(pred_angle[i, :]))) / 2

            loss_emd = pc_util.log_visu_loss('EMD Loss', align_loss)
            loss_mat = pc_util.log_visu_loss('MAT Loss', matloss)
            vec_mat = pc_util.log_visu_loss('VEC Loss', vecloss)
            loss_norm = pc_util.log_visu_loss('Norm Loss', norm_loss)

            info = info_input + pre_angle + matrix_input + matrix_pred + vec_mat + loss_mat + loss_emd + loss_norm

            pc_util.point_cloud_three_points(points_rotated, points_gt, points_align, img_filename, info)

            # scipy.misc.imsave(img_filename, output_img)

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    TEST_DATASET.reset()

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
