#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import tf_util2 as tf_util
#import loss_util

def placeholder_inputs(batch_size, num_patch, num_point,channel=3):
    pointclouds = tf.placeholder(tf.float32, shape=(batch_size*num_patch, num_point, channel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_patch))
    match_pair = tf.placeholder(tf.int32, shape=(batch_size, num_patch, num_patch))
    return pointclouds, labels_pl, match_pair


def get_model(point_cloud, is_training, bn_decay=None,channel=3):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)

    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1, channel],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
    #net = tf.reduce_max(net, axis=[2], keep_dims=True)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope='dp1')
    net = tf_util.fully_connected(net, 64, activation_fn=None, scope='fc3')
    net = tf.nn.l2_normalize(net,dim=1)

    return net



def get_feature_pairwise_distance(batch_features):
    """Compute pairwise distance of a point cloud.

    Args:
      batch_features: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    og_batch_size = batch_features.get_shape().as_list()[0]
    if og_batch_size == 1:
        batch_features = tf.expand_dims(batch_features, 0)

    batch_features_transpose = tf.transpose(batch_features, perm=[0, 2, 1])
    batch_features_inner = tf.matmul(batch_features, batch_features_transpose)

    batch_features_inner = -2 * batch_features_inner
    batch_features_square = tf.reduce_sum(tf.square(batch_features), axis=-1, keep_dims=True)
    batch_features_square_tranpose = tf.transpose(batch_features_square, perm=[0, 2, 1])

    return batch_features_square + batch_features_inner + batch_features_square_tranpose



def cal_matching_matrix(batch_label):
    B, N = batch_label.shape
    batch_match_matrix = np.zeros([B, N, N])

    for i in range(B):
        label_base = np.array(batch_label[0])[np.newaxis, :].repeat(N, axis=0)
        match_matrix = 1 - np.sign(np.abs(label_base - label_base.T))
        batch_match_matrix[i, :, :] = match_matrix

    return batch_match_matrix


if __name__ == '__main__':

    batch_size = 4
    num_patch = 16
    num_point = 32
    threshold = 0.2
    with tf.Graph().as_default():
        with tf.device('/cpu:' + str(0)):
            with tf.variable_scope("siamese") as scope:
                pointclouds_pl, labels_pl, match_pair = placeholder_inputs(batch_size, num_patch, num_point)
                input = tf.squeeze(pointclouds_pl[:, 0, :, :])
                output = get_model(input, tf.constant(False))
                output_list = tf.expand_dims(output, axis=1)
                scope.reuse_variables()
                for i in range(num_patch - 1):
                    input = tf.squeeze(pointclouds_pl[:, i + 1, :, :])
                    output = get_model(input, tf.constant(False))
                    output_list = tf.concat([output_list, tf.expand_dims(output, axis=1)], axis=1)

            # matching loss: sum(M.*D/|M|^2)
            D = get_feature_pairwise_distance(output_list)
            M = tf.cast(match_pair,dtype=tf.float32)
            M_square = tf.reduce_sum(tf.square(M),axis=[1,2])
            M_not = tf.cast(1 - match_pair,dtype=tf.float32)
            M_not_square = tf.reduce_sum(tf.square(M_not),axis=[1,2])

            matching_loss = tf.reduce_sum(tf.square(tf.multiply(M,D)),axis=[1,2])
            matching_loss = tf.divide(matching_loss,M_square)
            matching_loss_total = tf.reduce_sum(matching_loss)

            non_matching_loss2 = tf.multiply(M_not,D)

            non_matching_loss = tf.maximum(0.00005 - tf.multiply(M_not,D),tf.constant(0.0))
            non_matching_loss = tf.reduce_sum(tf.square(non_matching_loss),axis=[1,2])
            non_matching_loss = tf.divide(non_matching_loss,M_not_square)
            non_matching_loss_total = tf.reduce_sum(non_matching_loss)

            totalLoss = matching_loss_total + 2 * non_matching_loss_total

            train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss=totalLoss)

        ops = {'pointclouds_pl'         : pointclouds_pl,
               'labels_pl'              : labels_pl,
               'match_pair'             : match_pair,
               'output_list'            : output_list,
               'batch_pairwise_distance': D,
               'matching_loss'          : matching_loss,
               'non_matching_loss'      : non_matching_loss,
               'non_matching_loss2': non_matching_loss2,
               'totalLoss':totalLoss,
               'train_op': train_op,
               }
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            inputs = np.random.rand(batch_size, num_patch, num_point, 3)

            lables = np.array([[2, 3, 2, 3],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1],
                               [4, 1, 4, 1]]).T
            # lables = np.random.randint(5, size=[batch_size, num_patch])
            match_pairs = cal_matching_matrix(lables)


            for i in range(1):
                feed_dict = {ops['pointclouds_pl']: inputs,
                             ops['labels_pl']     : lables,
                             ops['match_pair']    : match_pairs}

                _,output_features, pair_feature_distance, _match_pair, _matching_loss,_non_matching_loss,_totalLoss,_non_matching_loss2 = sess.run(
                    [ops['train_op'],ops['output_list'], ops['batch_pairwise_distance'], ops['match_pair'], ops['matching_loss'],
                     ops['non_matching_loss'],ops['totalLoss'],ops['non_matching_loss2']],
                    feed_dict=feed_dict)


            print(output_features[0,0:3,:])
            print(np.min(pair_feature_distance))
            print(np.max(pair_feature_distance))

            #print(_match_pair[0, ...])
            #print(np.min(_non_matching_loss2))

            #print(np.max(_non_matching_loss2))
            #print(_non_matching_loss)
          #  print(_totalLoss)
            output_features = output_features.reshape([batch_size*num_patch,-1])
           # print(output_features.shape)

            whole_label = lables.reshape(-1)
            with open('features.txt', 'w') as file:
                for i in range(batch_size*num_patch):
                    file.write('label = ')
                    file.write(str(whole_label[i]) + '_%d' % (2))
                    file.write('\t ')
                    file.write(str(output_features[i, :]))
                    file.write('\n')



