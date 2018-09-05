import os
import time

import numpy as np
import tensorflow as tf
from scipy import spatial


def distance_point2edge(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = tf.expand_dims(points, axis=2)
    segment0 = tf.expand_dims(segment0, axis=1)
    segment1 = tf.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = tf.reduce_sum(w * v, axis=-1)
    c2 = tf.reduce_sum(v * v, axis=-1)

    # distance to the line
    distance0 = tf.reduce_sum(tf.pow(points - segment0, 2), axis=-1)
    distance1 = tf.reduce_sum(tf.pow(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = tf.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = tf.reduce_sum(tf.pow(points - segmentb, 2), axis=-1)
    dist = tf.where(c2 <= c1, distance1, distanceb)
    dist = tf.where(c1 <= 0, distance0, dist)
    return dist

def distance_point2edge_np(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = np.expand_dims(points, axis=2)
    segment0 = np.expand_dims(segment0, axis=1)
    segment1 = np.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = np.sum(w * v, axis=-1)
    c2 = np.sum(v * v, axis=-1)

    # distance to the line
    distance0 = np.sum(np.power(points - segment0, 2), axis=-1)
    distance1 = np.sum(np.power(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = np.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = np.sum(np.power(points - segmentb, 2), axis=-1)
    dist = np.where(c2 <= c1, distance1, distanceb)
    dist = np.where(c1 <= 0, distance0, dist)
    return dist

def distance_point2mesh(points, faces):
    # point (batch, nPoint,3)
    # tri   (batch, nTri,9)

    faces = tf.expand_dims(faces, axis=1)
    points = tf.expand_dims(points,axis=2)
    B  = faces[:, :, :, 0:3]
    E0 = faces[:, :, :, 3:6] - B
    E1 = faces[:, :, :, 6:9] - B

    D = B-points
    a = tf.reduce_sum(E0*E0,axis=-1)+1e-12
    b = tf.reduce_sum(E0*E1,axis=-1)+1e-12
    c = tf.reduce_sum(E1*E1,axis=-1)+1e-12
    d = tf.reduce_sum(E0*D, axis=-1)+1e-12
    e = tf.reduce_sum(E1*D, axis=-1)+1e-12
    f = tf.reduce_sum(D*D,  axis=-1)+1e-12

    det = a*c-b*b
    s = b*e -c*d
    t = b*d -a*e

    #region 4
    dist41  = tf.where(-d>=a, a+2*d+f, -d*d/a+f)
    dist422 = tf.where(-e>=c, c+2*e+f,-e*e/c+f)
    dist42 = tf.where(e >= 0, f, dist422)
    dist4 = tf.where(d < 0, dist41, dist42)

    #region 3
    dist3 = tf.where(e>=0,f, dist422)

    #region 5
    dist5 = tf.where(d>=0,f, dist41)

    #region 0
    ss = s/(det+1e-12)
    tt = t/(det+1e-12)
    dist0 = ss*(a*ss+b*tt + 2*d)+tt*(b*ss+c*tt+2*e)+f

    #region 2
    temp0 = b+d
    temp1 = c+e
    numer = temp1 -temp0
    denom = a - 2*b +c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist212 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist21 = tf.where(numer>=denom,a + 2*d +f,dist212)
    dist22 = tf.where(temp1<=0,c+2*e+f, tf.where(e>=0,f, -e*e/c+f))
    dist2 = tf.where(temp1>temp0,dist21,dist22)

    #region 6
    temp0 = b + e
    temp1 = a + d
    numer = temp1 -temp0
    denom = a-2*b+c
    tt = numer/(denom+1e-12)
    ss = 1 -tt
    dist612 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist61 = tf.where(numer>=denom,c+2*e+f,dist612)
    dist62 = tf.where(temp1<=0,a+2*d+f,tf.where(d>=0,f,-d*d/a+f))
    dist6 = tf.where(temp1>temp0, dist61,dist62)

    #region 1
    numer = c+e-b-d
    denom = a -2*b + c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist122 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist12 = tf.where(numer>denom, a + 2*d+f, dist122)
    dist1 = tf.where(numer<=0, c+2*e+f,dist12)


    dista = tf.where(s<0,tf.where(t<0,dist4,dist3),tf.where(t<0,dist5,dist0))
    distb = tf.where(s<0, dist2, tf.where(t<0,dist6,dist1))
    dist = tf.where(s+t<=det, dista, distb)
    dist = tf.maximum(dist,0.0)
    return dist

def distance_point2mesh_np(points, faces):
    # point (batch, nPoint,3)
    # tri   (batch, nTri,9)

    faces = np.expand_dims(faces, axis=1)
    points = np.expand_dims(points,axis=2)
    B  = faces[:, :, :, 0:3]
    E0 = faces[:, :, :, 3:6] - B
    E1 = faces[:, :, :, 6:9] - B

    D = B-points
    a = np.sum(E0*E0,axis=-1)+1e-12
    b = np.sum(E0*E1,axis=-1)+1e-12
    c = np.sum(E1*E1,axis=-1)+1e-12
    d = np.sum(E0*D, axis=-1)+1e-12
    e = np.sum(E1*D, axis=-1)+1e-12
    f = np.sum(D*D,  axis=-1)+1e-12

    det = a*c-b*b
    s = b*e -c*d
    t = b*d -a*e

    #region 4
    dist41  = np.where(-d>=a, a+2*d+f, -d*d/a+f)
    dist422 = np.where(-e>=c, c+2*e+f,-e*e/c+f)
    dist42 = np.where(e >= 0, f, dist422)
    dist4 = np.where(d < 0, dist41, dist42)

    #region 3
    dist3 = np.where(e>=0,f, dist422)

    #region 5
    dist5 = np.where(d>=0,f, dist41)

    #region 0
    ss = s/(det+1e-12)
    tt = t/(det+1e-12)
    dist0 = ss*(a*ss+b*tt + 2*d)+tt*(b*ss+c*tt+2*e)+f

    #region 2
    temp0 = b+d
    temp1 = c+e
    numer = temp1 -temp0
    denom = a - 2*b +c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist212 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist21 = np.where(numer>=denom,a + 2*d +f,dist212)
    dist22 = np.where(temp1<=0,c+2*e+f, np.where(e>=0,f, -e*e/c+f))
    dist2 = np.where(temp1>temp0,dist21,dist22)

    #region 6
    temp0 = b + e
    temp1 = a + d
    numer = temp1 -temp0
    denom = a-2*b+c
    tt = numer/(denom+1e-12)
    ss = 1 -tt
    dist612 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist61 = np.where(numer>=denom,c+2*e+f,dist612)
    dist62 = np.where(temp1<=0,a+2*d+f,np.where(d>=0,f,-d*d/a+f))
    dist6 = np.where(temp1>temp0, dist61,dist62)

    #region 1
    numer = c+e-b-d
    denom = a -2*b + c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist122 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist12 = np.where(numer>denom, a + 2*d+f, dist122)
    dist1 = np.where(numer<=0, c+2*e+f,dist12)


    dista = np.where(s<0,np.where(t<0,dist4,dist3),np.where(t<0,dist5,dist0))
    distb = np.where(s<0, dist2, np.where(t<0,dist6,dist1))
    dist = np.where(s+t<=det, dista, distb)
    # dist_min = np.reduce_min(dist,axis=2)
    # dist = tf.where(tf.is_nan(dist),tf.zeros_like(dist),dist)
    return dist


def projected_point(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = tf.expand_dims(points, axis=2)
    segment0 = tf.expand_dims(segment0, axis=1)
    segment1 = tf.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = tf.reduce_sum(w * v, axis=-1)
    c2 = tf.reduce_sum(v * v, axis=-1)

    # distance to the line
    distance0 = tf.reduce_sum(tf.pow(points - segment0, 2), axis=-1)
    distance1 = tf.reduce_sum(tf.pow(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = tf.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = tf.reduce_sum(tf.pow(points - segmentb, 2), axis=-1)

    projected_point0 = tf.tile(segment0, [1, 600, 1, 1])
    projected_point1 = tf.tile(segment1, [1, 600, 1, 1])
    projected_pointb = segmentb

    dist = tf.where(c2 <= c1, distance1, distanceb)
    dist = tf.where(c1 <= 0, distance0, dist)
    projected_point = tf.where(tf.tile(tf.expand_dims(c2 <= c1, axis=-1), [1, 1, 1, 3]), projected_point1,
                               projected_pointb)
    projected_point = tf.where(tf.tile(tf.expand_dims(c1 <= 0, axis=-1), [1, 1, 1, 3]), projected_point0,
                               projected_point)

    B = points.shape[0].value
    nPoint = points.shape[1].value
    idx0 = tf.tile(tf.reshape(tf.range(B), (B, 1)), [1, nPoint])
    idx1 = tf.tile(tf.reshape(tf.range(nPoint), (1, nPoint)), [B, 1])
    idx2 = tf.argmin(dist, axis=-1,output_type=tf.int32)
    projected_point = tf.gather_nd(projected_point, tf.stack([idx0, idx1, idx2], axis=-1))

    return projected_point

def projected_point_np(points, edges):
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = np.expand_dims(points, axis=2)
    segment0 = np.expand_dims(segment0, axis=1)
    segment1 = np.expand_dims(segment1, axis=1)

    v = segment1 - segment0
    w = points - segment0

    c1 = np.sum(w * v, axis=-1)
    c2 = np.sum(v * v, axis=-1)

    # distance to the line
    distance0 = np.sum(np.power(points - segment0, 2), axis=-1)
    distance1 = np.sum(np.power(points - segment1, 2), axis=-1)
    b = c1 / c2
    b = np.expand_dims(b, axis=-1)
    segmentb = segment0 + b * v
    distanceb = np.sum(np.power(points - segmentb, 2), axis=-1)

    dist = np.where(c2 <= c1, distance1, distanceb)
    dist = np.where(c1 <= 0, distance0, dist)

    projected_point = np.where(np.tile(np.expand_dims(c2 <= c1, axis=-1), [1, 1, 1, 3]), segment1, segmentb)
    projected_point = np.where(np.tile(np.expand_dims(c1 <= 0, axis=-1), [1, 1, 1, 3]), segment0, projected_point)

    B = points.shape[0]
    nPoint = points.shape[1]
    idx0 = np.tile(np.arange(B).reshape(B, 1), [1, nPoint])
    idx1 = np.tile(np.arange(nPoint).reshape(1, nPoint), [B, 1])
    idx2 = np.argmin(dist, axis=-1)
    projected_point = projected_point[idx0, idx1, idx2]

    return projected_point,np.min(dist,axis=-1)


def query_neighbor(pred_pts, sample_pts, radius=None):
    if np.isscalar(radius):
        radius = np.asarray([radius])
    radius = np.asarray(radius)
    pred_tree = spatial.cKDTree(pred_pts)
    sample_tree = spatial.cKDTree(sample_pts)
    counts = []
    for radi in radius:
        idx = sample_tree.query_ball_tree(pred_tree, r=radi)
        number = [len(item) for item in idx]
        counts.append(number)
    counts = np.asarray(counts)
    return counts


def resample4density(pred):
    pred_tree = spatial.cKDTree(pred[:, 0:3])
    dists, idx = pred_tree.query(pred[:, 0:3], k=30)
    avg = np.mean(dists, axis=1)
    prob = avg
    prob = prob / np.sum(prob)
    # cnts = query_neighbor(pred[:,0:3],pred[:,0:3],radius=[0.05,0.10,0.15])
    # cnts = cnts*1.0/np.reshape(np.asarray([1,4,9]),(3,1))
    # # cnts = np.sqrt(np.sum(cnts**2,axis=0))
    # # prob = 1.0 / (cnts + 1)
    # std = np.std(cnts, axis=0)
    # prob = std
    # prob = prob/np.sum(prob)
    idx = np.random.choice(len(pred), len(pred) / 4, replace=False, p=prob)
    # idx = np.random.permutation(np.arange(len(pred)))[:len(pred)/4]
    # f,ax= plt.subplots(2)
    # ax[0].plot(cnts)
    # ax[1].hist(idx,bins=100)
    # plt.show()
    select_pts = pred[idx]
    return select_pts


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data[:,:,3:] = 0
    jittered_data += batch_data
    return jittered_data