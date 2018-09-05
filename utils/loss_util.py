#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li
import numpy as np
import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))

from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from tf_ops.grouping.tf_grouping import query_ball_point, group_point

import Tools3D


def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12, dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square / h ** 2)
    uniform_loss = tf.reduce_mean(radius - dist * weight)
    return uniform_loss


def get_emd_loss2(pred, gt):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = 1  # pred.shape[0]
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    dist_norm = dist

    emd_loss = tf.reduce_mean(dist_norm)
    return emd_loss, matchl_out


def get_emd_loss(pred, gt):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    dist_norm = dist

    emd_loss = tf.reduce_mean(dist_norm)
    return emd_loss, dist


def get_cd_loss(pred, gt, radius):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, gt)
    # print(dists_backward.get_shape())
    # dists_forward is for each element in gt, the cloest distance to this element
    CD_dist = 1.0 * dists_forward  # + 0.2 * dists_backward
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist / radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss, CD_dist_norm


def get_transform_loss(pred, label, end_points, pointclouds_pl, pointclouds_gt, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform_feature']  # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat loss', mat_diff_loss)

    # Enforce the xyz transformation is the same as the standard one
    transform_xyz = end_points['transform_xyz']  # BxKxK
    point_cloud_transformed = tf.matmul(pointclouds_pl, transform_xyz)
    align_loss, _ = get_emd_loss(point_cloud_transformed, pointclouds_gt)

    total_loss = classify_loss + 100 * align_loss + mat_diff_loss * reg_weight
    tf.add_to_collection('losses', total_loss)

    return total_loss


def get_transformnet_loss_rotationvector(pred_angle, pointclouds_angle, pointclouds_pl, pointclouds_gt):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Enforce the transformation as orthogonal matrix
    transform_xyz_pred = Tools3D.batch_rotationVector2mat_tf(pred_angle)
    point_cloud_transformed = tf.matmul(pointclouds_gt, transform_xyz_pred)
    align_loss, emd_dist = get_emd_loss(point_cloud_transformed, pointclouds_pl)
    #
    # align_loss = 100 * align_loss
    # tf.summary.scalar('alignloss', align_loss)
    #
    # tf.add_to_collection('alignloss', align_loss)

    transform_xyz_raw = Tools3D.batch_rotationVector2mat_tf(pointclouds_angle)
    # mat_diff_loss = tf.nn.l2_loss(transform_xyz_pred-transform_xyz_raw)

    mat_diff_loss = tf.nn.l2_loss(pred_angle - pointclouds_angle)
    tf.summary.scalar('matloss', mat_diff_loss)
    tf.add_to_collection('matloss', mat_diff_loss)

    # Enforce the xyz transformation is the same as the standard one

    total_loss = 10 * mat_diff_loss + tf.losses.get_regularization_loss()

    tf.add_to_collection('losses', total_loss)

    return total_loss, emd_dist


def get_partialNet_loss(pred_result, pointclouds_quaternion, pointclouds_pl, pointclouds_gt,pointclouds_gt_big,pointclouds_gt_small, ratio=5):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Enforce the transformation as orthogonal matrix
    BATCH_SIZE = pred_result.get_shape()[0].value
    _, NUM_POINT, _ = pointclouds_gt.get_shape()
    #pred_transform = pred_result[:,4:8]
    pred_quaternion = pred_result[:,:4]
    transform_xyz_pred = Tools3D.batch_quaternion2mat_tf(pred_quaternion)
    # transform_xyz_pred = tf.transpose(transform_xyz_pred,[0,2,1])
    point_cloud_transformed = tf.matmul(pointclouds_gt_big, transform_xyz_pred)
    # align_loss, emd_dist = get_emd_loss(point_cloud_transformed, pointclouds_pl)
    # align_loss, cd_dist = get_cd_loss(pointclouds_pl, point_cloud_transformed, tf.constant(1.0))

    cd_dists, knn_dists = get_cd_loss_circle(pointclouds_pl, point_cloud_transformed,
                                             tf.constant(ratio, dtype=tf.int32))
    cd_loss = 5000 * tf.reduce_mean(cd_dists)
    knn_loss = 10 * 1000 * tf.reduce_mean(knn_dists)

    transform_xyz_gt = Tools3D.batch_quaternion2mat_tf(pointclouds_quaternion[:,:4])

    point_cloud_pred_transformed = tf.matmul(pointclouds_gt_small, transform_xyz_pred)
    point_cloud_gt_transformed = tf.matmul(pointclouds_gt_small, transform_xyz_gt)
    emd_loss, emd_dist = get_emd_loss(point_cloud_pred_transformed, point_cloud_gt_transformed)
    emd_loss = 2000 * emd_loss
    #
    # align_loss = 100 * align_loss
    #
    # tf.summary.scalar('alignloss',  align_loss)
    #
    # tf.add_to_collection('alignloss',  align_loss)
    #
    #transform_xyz_raw = Tools3D.batch_quaternion2mat_tf(pointclouds_quaternion)
    # transform_xyz_raw = tf.transpose(transform_xyz_raw, [0, 2, 1])
    mat_diff_loss = tf.nn.l2_loss(transform_xyz_pred - transform_xyz_gt)

    sign_pred = tf.reshape(tf.sign(pred_quaternion), [-1])
    sign_gt = tf.reshape(tf.sign(pointclouds_quaternion), [-1])

    sign_loss = tf.sign(pred_result) - tf.sign(pointclouds_quaternion)
    sign_loss = tf.nn.l2_loss(sign_loss)
    # mat_diff_loss = tf.nn.l2_loss(pred_quaternion-pointclouds_quaternion)

    # mat_diff_loss = mat_diff_loss #/ 10
    vec_diff_loss = tf.nn.l2_loss(pred_result - pointclouds_quaternion)

    mat_diff_loss = 1 * mat_diff_loss
    vec_diff_loss = 10 * vec_diff_loss
    sign_loss = 0.8 * sign_loss
    total_loss = vec_diff_loss + emd_loss + tf.losses.get_regularization_loss()

    tf.summary.scalar('emd_loss', emd_loss)
    tf.add_to_collection('emd_loss', emd_loss)

    tf.summary.scalar('cd_loss', cd_loss)
    tf.add_to_collection('cd_loss', cd_loss)
    tf.summary.scalar('knn_loss', knn_loss)
    tf.add_to_collection('knn_loss', knn_loss)
    tf.summary.scalar('sign_loss', sign_loss)
    tf.add_to_collection('sign_loss', sign_loss)
    tf.summary.scalar('vecloss', vec_diff_loss)
    tf.add_to_collection('vecloss', vec_diff_loss)
    tf.add_to_collection('losses', total_loss)

    return cd_dists, knn_dists


def get_tuneNet_loss(pred_quaternion, input_quaternion, gt_quaternion, pointclouds_pl, pointclouds_gt, ratio=1):
    """ pred: B*NUM_CLASSES,
        label: B, """

    # Enforce the transformation as orthogonal matrix

    pred_matrix = Tools3D.batch_quaternion2mat_tf(pred_quaternion)

    #input_quaternion_T = Tools3D.batch_quaternion_T_tf(input_quaternion)

    pred_gt = input_quaternion#Tools3D.batch_quaternion_Mul_quaternion_tf(input_quaternion_T, gt_quaternion)
    # pred_gt_matrix = Tools3D.batch_quaternion2mat_tf(pred_gt)
    #
    # pred_gt_matrix = Tools3D.batch_quaternion2mat_tf(pred_gt)
    # point_cloud_gt_transformed = tf.matmul(pointclouds_pl, pred_gt_matrix)
    #
    # gt_matrix = Tools3D.batch_quaternion2mat_tf(gt_quaternion)
    # point_cloud_gt_transformed = tf.matmul(pointclouds_gt, pred_gt_matrix)

    pointclouds_pl_transformed = tf.matmul(pointclouds_gt, pred_matrix)

    emd_loss, emd_dist = get_emd_loss(pointclouds_pl_transformed, pointclouds_pl)
    emd_loss = 20000 * emd_loss

    sign_loss = tf.sign(input_quaternion) - tf.sign(pred_quaternion)
    sign_loss = tf.nn.l2_loss(sign_loss)


    vec_diff_loss = pred_gt - pred_quaternion

    vec_diff_loss = 10 * tf.nn.l2_loss(vec_diff_loss)

    total_loss = vec_diff_loss + emd_loss + tf.losses.get_regularization_loss()

    tf.summary.scalar('emd_loss', emd_loss)
    tf.add_to_collection('emd_loss', emd_loss)
    tf.summary.scalar('vecloss', vec_diff_loss)
    tf.add_to_collection('vecloss', vec_diff_loss)
    tf.summary.scalar('sign_loss', sign_loss)
    tf.add_to_collection('sign_loss', sign_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss, emd_dist




def get_tuneNet_loss5(pred_quaternion, input_quaternion, gt_quaternion, pointclouds_pl, pointclouds_gt, ratio=1):
    """ pred: B*NUM_CLASSES,
        label: B, """

    # Enforce the transformation as orthogonal matrix
    BATCH_SIZE, NUM_POINT, _ = pointclouds_pl.get_shape()

    pred_matrix = Tools3D.batch_quaternion2mat_tf(pred_quaternion)

    input_matrix = Tools3D.batch_quaternion2mat_tf(input_quaternion)
    ouput_matrix = tf.matmul(input_matrix, pred_matrix)

    output_quaternion = Tools3D.batch_quaternion_Mul_quaternion_tf(input_quaternion, pred_quaternion)

    input_quaternion_T = Tools3D.batch_quaternion_T_tf(input_quaternion)

    pred_gt = Tools3D.batch_quaternion_Mul_quaternion_tf(input_quaternion_T, gt_quaternion)

    #output_quaternion = tf.nn.l2_normalize(output_quaternion, dim=1)

    # transform_xyz_pred = tf.transpose(transform_xyz_pred,[0,2,1])
    point_cloud_transformed = tf.matmul(pointclouds_pl, pred_matrix)

    gt_matrix = Tools3D.batch_quaternion2mat_tf(gt_quaternion)
    point_cloud_gt_transformed = tf.matmul(pointclouds_gt, gt_matrix)

    emd_loss, emd_dist = get_emd_loss(point_cloud_transformed, point_cloud_gt_transformed)
    emd_loss = 5000 * emd_loss


    sign_loss = tf.sign(gt_quaternion) - tf.sign(output_quaternion)
    sign_loss = 10*tf.nn.l2_loss(sign_loss)

    vec_diff_loss = pred_gt - pred_quaternion

    vec_diff_loss = 10 * tf.nn.l2_loss(vec_diff_loss)
    sign_loss = 0.1 * sign_loss

    total_loss = vec_diff_loss #+ tf.losses.get_regularization_loss()

    tf.summary.scalar('emd_loss', emd_loss)
    tf.add_to_collection('emd_loss', emd_loss)
    tf.summary.scalar('vecloss', vec_diff_loss)
    tf.add_to_collection('vecloss', vec_diff_loss)
    tf.summary.scalar('sign_loss', sign_loss)
    tf.add_to_collection('sign_loss', sign_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss, emd_dist



def get_tuneNet_loss2(pred_quaternion, input_quaternion, gt_quaternion, pointclouds_pl, pointclouds_gt, ratio=1):
    """ pred: B*NUM_CLASSES,
        label: B, """

    # Enforce the transformation as orthogonal matrix
    BATCH_SIZE, NUM_POINT, _ = pointclouds_pl.get_shape()

    pred_matrix = Tools3D.batch_quaternion2mat_tf(pred_quaternion)

    input_matrix = Tools3D.batch_quaternion2mat_tf(input_quaternion)
    ouput_matrix = tf.matmul(input_matrix, pred_matrix)

    output_quaternion = Tools3D.batch_quaternion_Mul_quaternion_tf(input_quaternion, pred_quaternion)
    output_quaternion = tf.nn.l2_normalize(output_quaternion, dim=1)
    refined_matrix = Tools3D.batch_quaternion2mat_tf(output_quaternion)

    # transform_xyz_pred = tf.transpose(transform_xyz_pred,[0,2,1])
    point_cloud_transformed = tf.matmul(pointclouds_gt, ouput_matrix)

    gt_matrix = Tools3D.batch_quaternion2mat_tf(gt_quaternion)
    point_cloud_gt_transformed = tf.matmul(pointclouds_gt, gt_matrix)

    emd_loss, emd_dist = get_emd_loss(point_cloud_transformed, point_cloud_gt_transformed)
    emd_loss = 5000 * emd_loss

    abs_loss = tf.square(point_cloud_transformed - point_cloud_gt_transformed)
    abs_loss = 1000 * tf.reduce_mean(abs_loss)

    cd_dists, knn_dists = get_cd_loss_circle(point_cloud_transformed, point_cloud_gt_transformed,
                                             tf.constant(ratio, dtype=tf.int32))
    cd_loss = 5000 * tf.reduce_mean(cd_dists)
    knn_loss = 5000 * tf.reduce_mean(knn_dists)
    # mat_diff = tf.matmul(transform_xyz_pred, tf.transpose(transform_xyz_pred, perm=[0, 2, 1]))



    # mat_diff = transform_xyz_raw - refined_matrix
    # mat_diff_loss = tf.nn.l2_loss(mat_diff)

    orientation_loss1 = 1000*tf.reduce_mean(tf.reduce_sum(gt_quaternion[:,:-1] * pred_quaternion[:,:-1],axis=1))
    orientation_loss2 = 1000*tf.reduce_mean(tf.reduce_sum(input_quaternion[:,:-1] * pred_quaternion[:,:-1],axis=1))





    mat_diff_loss = tf.nn.l2_loss(refined_matrix - ouput_matrix)

    cos_loss = 1.0 - tf.squeeze(tf.reduce_sum(gt_matrix * ouput_matrix, axis=1))

    # cos_loss = 1.0 - tf.squeeze(tf.reduce_sum(output_quaternion*gt_quaternion,axis=1))
    cos_loss = 150 * tf.reduce_mean(cos_loss)

    sign_loss = tf.sign(gt_quaternion) - tf.sign(output_quaternion)
    sign_loss = 10*tf.nn.l2_loss(sign_loss)

    vec_diff_loss = gt_quaternion - output_quaternion

    vec_diff_loss = 10 * tf.nn.l2_loss(vec_diff_loss)
    # vec_diff_loss = 1.0 - tf.squeeze(tf.reduce_sum(gt_quaternion * output_quaternion, axis=1))
    # emd_loss = 20000 * emd_loss #1000

    #  vec_diff_loss = 200 * tf.reduce_mean(vec_diff_loss)  # *100
    sign_loss = 0.1 * sign_loss

    total_loss = emd_loss + cos_loss + tf.losses.get_regularization_loss()

    tf.summary.scalar('orientation_loss1', orientation_loss1)
    tf.add_to_collection('orientation_loss1', orientation_loss1)
    tf.summary.scalar('orientation_loss2', orientation_loss2)
    tf.add_to_collection('orientation_loss2', orientation_loss2)

    tf.summary.scalar('cos_loss', cos_loss)
    tf.add_to_collection('cos_loss', cos_loss)

    tf.summary.scalar('abs_loss', abs_loss)
    tf.add_to_collection('abs_loss', abs_loss)

    tf.summary.scalar('emd_loss', emd_loss)
    tf.add_to_collection('emd_loss', emd_loss)
    tf.summary.scalar('vecloss', vec_diff_loss)
    tf.add_to_collection('vecloss', vec_diff_loss)
    tf.summary.scalar('matloss', mat_diff_loss)
    tf.add_to_collection('matloss', mat_diff_loss)

    tf.summary.scalar('cd_loss', cd_loss)
    tf.add_to_collection('cd_loss', cd_loss)

    tf.summary.scalar('knn_loss', knn_loss)
    tf.add_to_collection('knn_loss', knn_loss)
    tf.summary.scalar('sign_loss', sign_loss)
    tf.add_to_collection('sign_loss', sign_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss, cd_dists


def get_transform_loss2(pred_angle, pointclouds_angle):
    """ pred: B*NUM_CLASSES,
        label: B, """

    mat_diff_loss = tf.nn.l2_loss(pointclouds_angle - pred_angle)
    tf.summary.scalar('mat loss', mat_diff_loss)

    total_loss = mat_diff_loss
    tf.add_to_collection('losses', total_loss)

    return total_loss


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


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx


def get_N_tuple_loss(batch_features, match_pairs, theta=0.0005):
    D = get_feature_pairwise_distance(batch_features)

    M = tf.cast(match_pairs, dtype=tf.float32)
    M_square = tf.reduce_sum(M, axis=[1, 2])
    M_not = tf.cast(1 - match_pairs, dtype=tf.float32)
    M_not_square = tf.reduce_sum(M_not, axis=[1, 2])

    # matching loss: sum(M.*D)/|M|^2
    matching_loss = tf.reduce_sum(tf.square(tf.multiply(M, D)), axis=[1, 2])
    matching_loss = tf.divide(matching_loss, M_square)
    matching_loss_total = tf.reduce_sum(matching_loss)
    matching_loss_total = 20 * matching_loss_total
    # matching loss: sum(max(theta - (1-M).*D, 0))/|1-M|^2
    non_matching_loss = 2.0 - tf.multiply(M_not, D)
    non_matching_loss = tf.maximum(non_matching_loss, tf.constant(0.0))

    non_matching_loss = tf.reduce_sum(tf.square(non_matching_loss), axis=[1, 2])
    non_matching_loss = tf.divide(non_matching_loss, M_not_square)
    non_matching_loss_total = tf.reduce_sum(non_matching_loss)
    totalLoss = matching_loss_total + non_matching_loss_total

    tf.summary.scalar('matchingloss', matching_loss_total)
    tf.add_to_collection('matchingloss', matching_loss_total)

    tf.summary.scalar('non-matchingloss', non_matching_loss_total)
    tf.add_to_collection('non-matchingloss', non_matching_loss_total)

    tf.summary.scalar('MinDistance', tf.reduce_min(D))
    tf.summary.scalar('MaxDistance', tf.reduce_max(D))

    tf.add_to_collection('losses', totalLoss)

    return totalLoss


def get_cd_loss_circle(input1, input2, ratio=2, k=6):
    cd_left_dists, leftIDs, _, _ = tf_nndistance.nn_distance(input1, input2)
    adj = get_feature_pairwise_distance(input1)
    nn_idx = knn(adj, k=k)  # (batch, num_points, k)

    batch_size = input1.get_shape()[0].value
    num_points_1 = input1.get_shape()[1].value
    num_points_2 = num_points_1 * ratio
    num_dims = input1.get_shape()[2].value

    idx_1 = tf.range(batch_size) * num_points_1
    idx_1 = tf.reshape(idx_1, [batch_size, 1, 1])

    idx_2 = tf.range(batch_size) * num_points_2
    idx_2 = tf.reshape(idx_2, [batch_size, 1, 1])

    input1_flat = tf.reshape(leftIDs, [-1])
    input1_neighbors = tf.gather(input1_flat, nn_idx + idx_1)
    input1_neighbors = tf.squeeze(input1_neighbors)

    input2_flat = tf.reshape(input2, [-1, num_dims])
    input2_neighbors = tf.gather(input2_flat, input1_neighbors + idx_2)

    input2_center_idx = tf.expand_dims(leftIDs, -1)
    input2_centers = tf.gather(input2_flat, input2_center_idx + idx_2)
    input2_centers = tf.tile(input2_centers, [1, 1, k, 1])

    neighbour_dists = input2_neighbors - input2_centers
    neighbour_dists = tf.reduce_mean(tf.square(neighbour_dists), axis=[-2, -1], keep_dims=True)
    neighbour_dists = tf.squeeze(neighbour_dists)
    # return cd_left_dists, neighbour_dists
    return tf.reduce_mean(cd_left_dists, axis=1), tf.reduce_mean(neighbour_dists, axis=1)


def gather_features(point_cloud, nn_idx, idx_, batch_size=64, num_points=1024, num_dims=3):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    # point_cloud_shape = point_cloud.get_shape()
    # batch_size = point_cloud_shape[0].value
    # num_points = point_cloud_shape[1].value
    # num_dims = point_cloud_shape[2].value

    # idx_ = tf.range(batch_size) * num_points
    # idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

    return point_cloud_neighbors


def get_edge_feature(point_cloud, nn_idx, k=6):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature


if __name__ == '__main__':
    if __name__ == '__main__':
        import numpy as np
        import random
        import time
        from tensorflow.python.ops.gradient_checker import compute_gradient

        random.seed(100)
        np.random.seed(100)
        with tf.Session('') as sess:
            xyz1 = np.random.randn(4, 16, 3).astype('float32')
            xyz2 = np.random.randn(4, 64, 3).astype('float32')
            inp1 = tf.Variable(xyz1)
            inp2 = tf.constant(xyz2)
            dists, neighbours = get_cd_loss_circle(inp1, inp2)
            # neighbors2 = get_neighBour_index(neighbour, nn_idx)

            sess.run(tf.global_variables_initializer())

            dists, neighbours = sess.run([dists, neighbours])
            # n2 = neighbour[:, nn_idx[0, :, :]]

            print(np.sum(dists), np.sum(neighbours))
            print(dists.shape, neighbours.shape)

            # print(xyz2[n2,0])
            # print(nn_idx.shape)
        # print(dists[0,:])
