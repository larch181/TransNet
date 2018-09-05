#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import os
import sys
import math
import numpy as np
import tensorflow as tf

if sys.version_info >= (3, 0):
    from functools import reduce

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vector_length(a):
    a = np.array(a)
    return np.sqrt(np.sum(np.power(a, 2)))


def normalized(vec):
    len = vector_length(vec)
    return vec / len


# Calculate Rotation from a to b
def cal_rotate_matrix(a, b):
    rot_axis = cross(b, a)
    rot_angle = math.acos(dot(a, b) / vector_length(a) / vector_length(b))

    norm = vector_length(rot_axis)

    rot_axis = (rot_axis[0] / norm, rot_axis[1] / norm, rot_axis[2] / norm)

    return rotation_vector_To_rotation_matrix(rot_axis, rot_angle)


def euler_To_rotation_matrix(angle):
    # Return matrix for rotations around z, y and x axes
    rot_mat = np.zeros((3, 3), dtype="float32")

    cx = math.cos(angle[0])
    sx = math.sin(angle[0])

    cy = math.cos(angle[1])
    sy = math.sin(angle[1])

    cz = math.cos(angle[2])
    sz = math.sin(angle[2])

    rot_mat[0, 0] = cy * cz
    rot_mat[0, 1] = cz * sx * sy - cx * sz
    rot_mat[0, 2] = sx * sz + cx * cz * sy

    rot_mat[1, 0] = cy * sz
    rot_mat[1, 1] = cx * cz + sx * sy * sz
    rot_mat[1, 2] = cx * sy * sz - cz * sx

    rot_mat[2, 0] = -sy
    rot_mat[2, 1] = cy * sx
    rot_mat[2, 2] = cx * cy

    return np.matrix(rot_mat)


def rotation_matrix_To_euler(rot_mat):
    x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
    y = math.atan2(-rot_mat[2, 0], math.sqrt(rot_mat[2, 1] * rot_mat[2, 1] + rot_mat[2, 2] * rot_mat[2, 2]))
    z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])

    return np.array(x, y, z)


def euler2mat_tf(angles):
    Rx = [[1, 0, 0],
          [0, tf.cos(angles[0]), -tf.sin(angles[0])],
          [0, tf.sin(angles[0]), tf.cos(angles[0])]]
    Ry = [[tf.cos(angles[1]), 0, tf.sin(angles[1])],
          [0, 1, 0],
          [-tf.sin(angles[1]), 0, tf.cos(angles[1])]]
    Rz = [[tf.cos(angles[2]), -tf.sin(angles[2]), 0],
          [tf.sin(angles[2]), tf.cos(angles[2]), 0],
          [0, 0, 1]]
    rotation_matrix = tf.matmul(Rz, tf.matmul(Ry, Rx))

    return rotation_matrix


def rotation_vector_To_rotation_matrix(rot_axis, rot_angle):
    rot_mat = np.zeros((3, 3), dtype="float32")

    rot_axis = normalized(rot_axis)
    cosV = math.cos(rot_angle)
    sinV = math.sin(rot_angle)

    rot_mat[0, 0] = cosV + rot_axis[0] * rot_axis[0] * (1 - cosV)
    rot_mat[0, 1] = rot_axis[0] * rot_axis[1] * (1 - cosV) - rot_axis[2] * sinV
    rot_mat[0, 2] = rot_axis[1] * sinV + rot_axis[0] * rot_axis[2] * (1 - cosV)

    rot_mat[1, 0] = rot_axis[2] * sinV + rot_axis[0] * rot_axis[1] * (1 - cosV)
    rot_mat[1, 1] = cosV + rot_axis[1] * rot_axis[1] * (1 - cosV)
    rot_mat[1, 2] = -rot_axis[0] * sinV + rot_axis[1] * rot_axis[2] * (1 - cosV)

    rot_mat[2, 0] = -rot_axis[1] * sinV + rot_axis[0] * rot_axis[2] * (1 - cosV)
    rot_mat[2, 1] = rot_axis[0] * sinV + rot_axis[1] * rot_axis[2] * (1 - cosV)
    rot_mat[2, 2] = cosV + rot_axis[2] * rot_axis[2] * (1 - cosV)
    return np.matrix(rot_mat)


def rotation_matrix_To_quaternion(rot_mat):
    w = math.sqrt(1 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2

    x = (rot_mat[2, 1] - rot_mat[1, 2]) / (4 * w)
    y = (rot_mat[0, 2] - rot_mat[2, 0]) / (4 * w)
    z = (rot_mat[1, 0] - rot_mat[0, 1]) / (4 * w)

    return np.array([x, y, z, w])


import math


def rotMatrix_To_quaternion(rotMatrix):
    # math.sqrt()
    w = 1 + rotMatrix[0, 0] + rotMatrix[1, 1] + rotMatrix[2, 2]
    w = 0.5 * np.sqrt(w)
    x = 1 + rotMatrix[0, 0] - rotMatrix[1, 1] - rotMatrix[2, 2]
    x = 0.5 * np.sqrt(x)
    y = 1 - rotMatrix[0, 0] + rotMatrix[1, 1] - rotMatrix[2, 2]
    y = 0.5 * np.sqrt(y)
    z = 1 - rotMatrix[0, 0] - rotMatrix[1, 1] + rotMatrix[2, 2]
    z = 0.5 * np.sqrt(z)

    x = np.sign(rotMatrix[1, 2] - rotMatrix[2, 1]) * x
    y = np.sign(rotMatrix[2, 0] - rotMatrix[0, 2]) * y
    z = np.sign(rotMatrix[0, 1] - rotMatrix[1, 0]) * z

    return np.array([x, y, z, w], dtype=np.float32)


def quaternion_Mul_quaternion(left, right):
    # https://blog.csdn.net/shenshikexmu/article/details/53608224?locationNum=8&fps=1

    x1 = left[0]
    y1 = left[1]
    z1 = left[2]
    w1 = left[3]

    x2 = right[0]
    y2 = right[1]
    z2 = right[2]
    w2 = right[3]

    out = np.zeros(4)

    out[0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
    out[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
    out[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z
    out[3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w


    return normalized(out)


def quaternion_Mul_quaternion_tf(left, right):
    # print(rot_axis.shape)
    x1 = left[0]
    y1 = left[1]
    z1 = left[2]
    w1 = left[3]

    x2 = right[0]
    y2 = right[1]
    z2 = right[2]
    w2 = right[3]

    output_list = []

    output_list.append(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2)
    output_list.append(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2)
    output_list.append(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    output_list.append(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)

    outputs = tf.stack(output_list)
   # outputs = tf.reshape(outputs, [4, ])

    return tf.reshape(outputs, [4])


def quaternion_To_rotation_matrix(quaternion):
    # https://blog.csdn.net/shenshikexmu/article/details/53608224?locationNum=8&fps=1
    # transposed matrix
    rot_mat = np.zeros((3, 3), dtype="float32")
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    rot_mat[0, 0] = 1 - 2 * y * y - 2 * z * z
    rot_mat[0, 1] = 2 * (x * y + z * w)
    rot_mat[0, 2] = 2 * (x * z - y * w)

    rot_mat[1, 0] = 2 * (x * y - z * w)
    rot_mat[1, 1] = 1 - 2 * x * x - 2 * z * z
    rot_mat[1, 2] = 2 * (y * z + x * w)

    rot_mat[2, 0] = 2 * (x * z + y * w)
    rot_mat[2, 1] = 2 * (y * z - x * w)
    rot_mat[2, 2] = 1 - 2 * x * x - 2 * y * y

    return np.matrix(rot_mat).T


def quaternion_To_rotation_matrix_tf(quaternion):
    # print(rot_axis.shape)
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    output_list = []

    output_list.append(1 - 2 * y * y - 2 * z * z)
    output_list.append(2 * (x * y + z * w))
    output_list.append(2 * (x * z - y * w))

    output_list.append(2 * (x * y - z * w))
    output_list.append(1 - 2 * x * x - 2 * z * z)
    output_list.append(2 * (y * z + x * w))

    output_list.append(2 * (x * z + y * w))
    output_list.append(2 * (y * z - x * w))
    output_list.append(1 - 2 * x * x - 2 * y * y)

    outputs = tf.stack(output_list)
    outputs = tf.reshape(outputs, [3, 3])
    return tf.transpose(outputs,perm=[1,0])


def rotation_vector_To_quaternion(rot_axis, rot_angle):
    cosV = math.cos(rot_angle / 2)
    sinV = math.sin(rot_angle / 2)

    rot_axis = normalized(rot_axis)

    q0 = rot_axis[0] * sinV
    q1 = rot_axis[1] * sinV
    q2 = rot_axis[2] * sinV
    q3 = cosV

    return np.array([q0, q1, q2, q3])


def rotation_vector_To_rotation_matrix_tf(rot_axis, rot_angle):
    # print(rot_axis.shape)
    cosV = tf.cos(rot_angle)
    sinV = tf.sin(rot_angle)

    output_list = []

    output_list.append(cosV + rot_axis[0] * rot_axis[0] * (1 - cosV))
    output_list.append(rot_axis[0] * rot_axis[1] * (1 - cosV) - rot_axis[2] * sinV)
    output_list.append(rot_axis[1] * sinV + rot_axis[0] * rot_axis[2] * (1 - cosV))

    output_list.append(rot_axis[2] * sinV + rot_axis[0] * rot_axis[1] * (1 - cosV))
    output_list.append(cosV + rot_axis[1] * rot_axis[1] * (1 - cosV))
    output_list.append(-rot_axis[0] * sinV + rot_axis[1] * rot_axis[2] * (1 - cosV))

    output_list.append(-rot_axis[1] * sinV + rot_axis[0] * rot_axis[2] * (1 - cosV))
    output_list.append(rot_axis[0] * sinV + rot_axis[1] * rot_axis[2] * (1 - cosV))
    output_list.append(cosV + rot_axis[2] * rot_axis[2] * (1 - cosV))

    outputs = tf.stack(output_list)

    return tf.reshape(outputs, [3, 3])


def batch_angle2mat(angles):
    batch_size, channel = angles.shape

    mats = np.zeros((batch_size, 3, 3), dtype=float)

    for i in range(batch_size):
        z, y, x = angles[i, :]
        mats[i, :, :] = euler_To_rotation_matrix(angles[i, :])

    return mats


def batch_angle2mat_tf(angles):
    batch_size, channel = angles.shape
    # import pdb;
    # pdb.set_trace()

    # mats = tf.zeros((batch_size, 3, 3), dtype=tf.float32)
    output_list = []
    for i in range(batch_size):
        # import pdb;pdb.set_trace()
        output_list.append(euler2mat_tf(angles[i, :]))
        # mats[i, :, :] = euler2mat_tf(angles[i,:])

    mats = tf.stack(output_list)

    # import pdb;pdb.set_trace()
    return mats


def batch_rotationVector2mat(angles):
    batch_size, channel = angles.shape
    mats = np.zeros((batch_size, 3, 3), dtype=float)
    for i in range(batch_size):
        mats[i, :, :] = rotation_vector_To_rotation_matrix(angles[i, 0:-1], angles[i, -1])
    return mats


def batch_rotationVector2mat_tf(angles):
    batch_size, channel = angles.shape
    # import pdb;
    # pdb.set_trace()

    # mats = tf.zeros((batch_size, 3, 3), dtype=tf.float32)
    output_list = []
    for i in range(batch_size):
        # import pdb;pdb.set_trace()
        angle = angles[i, 0:-1]
        assert (angle.shape[0] == 3)
        output_list.append(rotation_vector_To_rotation_matrix_tf(angles[i, 0:-1], angles[i, -1]))
        # mats[i, :, :] = euler2mat_tf(angles[i,:])

    mats = tf.stack(output_list)

    # import pdb;pdb.set_trace()
    return mats


def batch_quaternion2mat(angles):
    batch_size, channel = angles.shape
    mats = np.zeros((batch_size, 3, 3), dtype=float)
    for i in range(batch_size):
        mats[i, :, :] = quaternion_To_rotation_matrix(angles[i, :])
    return mats


def batch_quaternion_Mul_quaternion(batch_left, batch_right):
    # https://blog.csdn.net/shenshikexmu/article/details/53608224?locationNum=8&fps=1
    # transposed matrix
    batch_out = np.zeros(batch_left.shape)
    for i in range(batch_left.shape[0]):
        left = batch_left[i, :]
        right = batch_right[i, :]
        batch_out[i, :] = quaternion_Mul_quaternion(left, right)
    return batch_out

def quaternion_T(data):
    data_T = - data
    data_T[-1] = -data_T[-1]
    return data_T

def batch_quaternion_T(batch_data):
    num,_ = batch_data.shape
    batch_data_T = np.zeros(batch_data.shape)
    for i in range(num):
        batch_data_T[i,:] =  quaternion_T(batch_data[i,:])
    return batch_data_T

def batch_quaternion2mat_tf(angles):
    batch_size, channel = angles.shape
    # import pdb;
    # pdb.set_trace()

    # mats = tf.zeros((batch_size, 3, 3), dtype=tf.float32)
    output_list = []
    for i in range(batch_size):
        # import pdb;pdb.set_trace()
        angle = angles[i, 0:-1]
        assert (angle.shape[0] == 3)
        output_list.append(quaternion_To_rotation_matrix_tf(angles[i, :]))
        # mats[i, :, :] = euler2mat_tf(angles[i,:])

    mats = tf.stack(output_list)

    # import pdb;pdb.set_trace()
    return mats


def batch_quaternion_Mul_quaternion_tf(batch_left, batch_right):
    # https://blog.csdn.net/shenshikexmu/article/details/53608224?locationNum=8&fps=1
    # transposed matrix
    batch_size, channel = batch_left.shape

    output_list = []
    for i in range(batch_size):
        left = batch_left[i, :]
        right = batch_right[i, :]
        output_list.append(quaternion_Mul_quaternion_tf(left, right))

    quats = tf.stack(output_list)
    return quats#tf.reshape(quats, [batch_size, 4])



def quaternion_T_tf(data):
    output_list = []
    output_list.append(-data[0])
    output_list.append(-data[1])
    output_list.append(-data[2])
    output_list.append(data[3])
    data_T = tf.stack(output_list)
    return tf.reshape(data_T, [4])

def batch_quaternion_T_tf(batch_data):
    # https://blog.csdn.net/shenshikexmu/article/details/53608224?locationNum=8&fps=1
    # transposed matrix
    batch_size, channel = batch_data.shape

    output_list = []
    for i in range(batch_size):
        data_T = quaternion_T_tf(batch_data[i, :])
        output_list.append(data_T)

    quats = tf.stack(output_list)
    return quats#tf.reshape(quats, [batch_size, 4])


if __name__ == '__main__':
    # a = (-0.006576016845720566, 0.20515224329972243, 0.011860567926381188)
    # b = (0, 0.2056, 0)
    # rot_mat = cal_rotate_matrix(a, b)
    #
    # print(b)
    # print(np.array(a) * rot_mat)

   #  rotationVec = np.array([3, 4, 5, math.pi])
   #  rot_mat = rotation_vector_To_rotation_matrix(rotationVec[0:-1], rotationVec[-1])
   #
   #  quen = rotation_vector_To_quaternion(rotationVec[0:-1], rotationVec[-1])
   #  quenM = quaternion_To_rotation_matrix(quen)
   #
   #  #print(rot_mat)
   # # print(quenM)
   #
   #
   #
   #  rot1 = np.array([0, 1, 1, math.pi/4])
   #  rot2 = np.array([1, 0, 0, math.pi / 4])
   #
   #  quen1 = rotation_vector_To_quaternion(rot1[0:-1],rot1[-1])
   #  quen2 = rotation_vector_To_quaternion(rot2[0:-1],rot2[-1])
   #
   #  quen1_T = - quen1
   #  quen1_T[-1] = -quen1_T[-1]
   #
   #
   # # print(quen1,quen1_T)
   #
   #
   #  quen3 = quaternion_Mul_quaternion(quen1,quen2)
   #  quen4 = quaternion_Mul_quaternion(quen1_T,quen3)
   #  print(quen2, quen4)
   #
   #  matrix1 = quaternion_To_rotation_matrix(quen1)
   #  matrix2 = quaternion_To_rotation_matrix(quen2)
   #
   #  matrix4 = np.matmul(matrix1,matrix2)
   #  matrix3 = quaternion_To_rotation_matrix(quen3)

    #print(matrix4)
    #print(matrix3)

    rotationVec = np.array([0, 1, 0, math.pi/4])
    quen = rotation_vector_To_quaternion(rotationVec[0:-1], rotationVec[-1])
    quenM = quaternion_To_rotation_matrix(quen)
    quen_tf = tf.constant(quen,dtype=tf.float64)
    rotMa =  quaternion_To_rotation_matrix_tf(quen_tf)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(rotMa))
        print(quenM)

