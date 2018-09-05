import os
import sys
import numpy as np
import h5py
from scipy.spatial.distance import euclidean
#sys.setrecursionlimit(1000000)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'ops'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))
#from ops import DataOperator
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
#from tf_ops.sampling import tf_sampling

import Tools3D


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]

def shuffle_single_points(points):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(points.shape[0])
    np.random.shuffle(idx)
    return points[idx, :]

def shuffle_batch_data(batch_data, batch_label):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxPxNxC array
        Output:
            BxPxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :, :], batch_label[:, idx]


def shuffle_batch_data2(batch_data, batch_label):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[0]).reshape((-1, 2))
    idx_T = idx[:, 1].tolist()
    idx_T.reverse()
    idx[:, 1] = idx_T
    idx = idx.reshape(-1)
    # np.random.shuffle(idx)
    if batch_label is not None:
        return batch_data[idx, ...], batch_label[idx, :]
    else:
        return batch_data[idx, ...]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def point_normalization(points):
    centroid = np.mean(points, axis=0, keepdims=True)
    furthest_distance = np.amax(np.sqrt(np.sum((points - centroid) ** 2, axis=-1)),
                                keepdims=True)
    points = (points - centroid) / furthest_distance
    return points


def point_sampling(point,sess,patch_size=1024, normalization=True):
    pointNum = point.shape[0]
    idx = np.arange(pointNum)

    if pointNum < patch_size:

        offset = patch_size - pointNum
        idx = np.concatenate([np.arange(pointNum), np.random.randint(0,pointNum, size=offset)], axis=0)
        np.random.shuffle(idx)

    elif pointNum > patch_size:

        #sample_data = np.expand_dims(point, axis=0)
        #sample_seed = tf_sampling.farthest_point_sample(patch_size, sample_data)
        # with tf.Session(config=config) as sess:
        #idx = sess.run(sample_seed)
        #idx = np.squeeze(idx)
        #idx = farthest_sampling(point, patch_size)
        np.random.shuffle(idx)
        idx = idx[:patch_size]

    point = point[idx, ...]
    if normalization:
        centroid = np.mean(point, axis=0, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((point - centroid) ** 2, axis=-1)),
                                    keepdims=True)
        point = (point - centroid) / furthest_distance

    return point




def point_erosion(points, ratio):
    holes = [0.05,0.1,0.15, 0.2,0.25, 0.3]

    N, _ = points.shape

    num = int(N * ratio)
    while (num > 0) & (points.shape[0] >= 1024):
        N, _ = points.shape

        idx = np.random.randint(0, points.shape[0])
        id = int(np.random.randint(0, len(holes), size=1))
        distance = np.sum(np.sqrt(np.square(points - points[idx])), axis=1)

        remain_idx = np.where(distance > holes[id])[0]
        points = points[remain_idx, ...]

        num = num - (N - points.shape[0])
    return points


def dis_point_list(point, point_list):
    return [euclidean(point, point_list[i]) for i in range(len(point_list))]

def dis_point_list_idx(idx, point_list):

    return [euclidean(point_list[idx], point_list[i]) for i in range(len(point_list))]

def farthest_point(points, point_list):
    ds_max = max(point_list)
    idx = point_list.index(ds_max)
    return points[idx]

def farthest_sampling_idx(points, K):
    N = points.shape[0]
    farthest_pts = np.zeros(K).astype(np.int32)
    farthest_pts[0] = 0

    ds_tmp = dis_point_list_idx(farthest_pts[0],points)

    for i in range(1,K):
        farthest_pts[i] = np.argmax(ds_tmp)
        ds_tmp2 = dis_point_list_idx(farthest_pts[i],points)
        ds_tmp = [min(ds_tmp[j],ds_tmp2[j]) for j in range(len(ds_tmp))]

    return farthest_pts


def farthest_point(points, point_list):
    ds_max = max(point_list)
    idx = point_list.index(ds_max)
    return points[idx]

def farthest_sampling(points, K):
    N = points.shape[0]
    farthest_pts = [0]*K
    P0 = points[0]
    farthest_pts[0] = P0
    ds_tmp = dis_point_list(P0,points)

    for i in range(1,K):
        farthest_pts[i] = farthest_point(points,ds_tmp)
        ds_tmp2 = dis_point_list(farthest_pts[i],points)
        ds_tmp = [min(ds_tmp[j],ds_tmp2[j]) for j in range(len(ds_tmp))]

    return np.array(farthest_pts)


def single_farthest_sampling(points,K):

    idx = np.zeros(K).astype(np.int32)

    DataOperator.farthest_point_samplin_func(np.squeeze(points),idx,K)

    return idx


def batch_farthest_sampling(points,K):
    if len(points.shape)==2:
        points = np.expand_dims(points,0)

    B,N,C = points.shape

    idx_nn = np.zeros([B,K]).astype(np.int32)
    points_sampled = np.zeros([B,K,C])

    for i in range(B):
        idx_nn[i, :] = single_farthest_sampling(points[i,...],K)
        #points_sampled[i,:,:] = points[i,idx_nn[i,:],:]

    return np.squeeze(idx_nn)


def get_pairwise_distance(batch_features):
    """Compute pairwise distance of a point cloud.

    Args:
      batch_features: numpy (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    og_batch_size = len(batch_features.shape)

    if og_batch_size == 2: #just two dimension
        batch_features = np.expand_dims(batch_features, axis=0)


    batch_features_transpose = np.transpose(batch_features, (0, 2, 1))

    batch_features_inner = batch_features@batch_features_transpose

    #print(np.max(batch_features_inner), np.min(batch_features_inner))


    batch_features_inner = -2 * batch_features_inner
    batch_features_square = np.sum(np.square(batch_features), axis=-1, keepdims=True)


    batch_features_square_tranpose = np.transpose(batch_features_square, (0, 2, 1))


    return batch_features_square + batch_features_inner + batch_features_square_tranpose




def pointcloud_simulation(batch_data,batch_data_numbers,sess=None, ratio=0.2, patch_size=1024):

    B, N, C = batch_data.shape
    batch_data_out = np.zeros((B, patch_size, C))
    for i in range(B):
        points = batch_data[i, :batch_data_numbers[i], :]
        points = shuffle_single_points(points)
        points = point_erosion(points, ratio=0.1)
        points = point_sampling(np.squeeze(points),patch_size=patch_size,sess=sess,normalization=True)
        #points = point_normalization(points)
        batch_data_out[i, ...] = points

    return batch_data_out


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, 0:3] += shifts[batch_index, :]
    return batch_data


def random_rotate_point_cloud(batch_data, rotation_base, hasnormal=False):
    B, N, C = batch_data.shape
    len, _ = rotation_base.shape
    # just random rotate
    seeds = np.random.randint(2, size=B)
    # just rotate smaller one
    seeds = np.array([1, 0]).reshape(1, 2).repeat(B, axis=0).reshape(-1)

    rotation_idx = np.random.randint(len, size=B)

    for i in range(B):
        points = np.squeeze(batch_data[i, :, :])
        if seeds[i] != 0:
            rotation_mat = Tools3D.quaternion_To_rotation_matrix(rotation_base[rotation_idx[i], :])
            # print(rotation_mat)
            batch_data[i, :, :3] = np.dot(points[:, :3], rotation_mat)
            if hasnormal:
                batch_data[i, :, 3:6] = np.dot(points[:, 3:6], rotation_mat)

    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename, pointNum=1024):
    f = h5py.File(h5_filename)
    data = f['data'][:pointNum]
    label = f['label'][:pointNum]
    print(data.shape)
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def load_fineTune_data(h5_filename):
    f = h5py.File(h5_filename)
    partial_data = f['partial_data']
    rotated_data = f['rotated_data']
    pred_quat = f['pred_quat']
    gt_quat = f['gt_quat']
    labels = f['labels']
    return partial_data, rotated_data, pred_quat, gt_quat, labels


# Write fine tune data to h5_filename
def save_fineTune_data(h5_filename, partial_data, rotated_data, pred_quat, gt_quat, labels, data_dtype='float32',
                       label_dtype='int32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'partial_data', data=partial_data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'rotated_data', data=rotated_data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'pred_quat', data=pred_quat,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'gt_quat', data=gt_quat,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'labels', data=labels,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()


if __name__ == '__main__':
    # outFile = 'E:/DeepLearning/PointCloud/Dataset/Data/test/einstein@normal_0_0_1.xyz'
    # batch_data = np.loadtxt('E:/DeepLearning/PointCloud/Dataset/Data/test/einstein@normal_1_0.xyz')
    #
    # batch_data = np.expand_dims(batch_data, axis=0)
    # ROTATION_DATABASE_FILE = os.path.join(ROOT_DIR,
    #                                       'E:/DeepLearning/PointCloud/Dataset/Data/test/quaternion_1000_20_train.xyz')
    #
    # ROTATION_DATABASE = np.loadtxt(ROTATION_DATABASE_FILE)
    #
    # idx = np.arange(ROTATION_DATABASE.shape[0])
    # np.random.shuffle(idx)
    # ROTATION_DATABASE = ROTATION_DATABASE[idx, ...]
    # batch_data_T = np.squeeze(random_rotate_point_cloud(batch_data, ROTATION_DATABASE, hasnormal=True))
    # print(batch_data_T.shape)
    # print(type(batch_data_T))
    # np.savetxt(outFile, batch_data_T, fmt="%f")

    # filename = os.path.join(ROOT_DIR, 'TuneNet/data/finetune_test.h5')
    # print(filename)
    # partial_data, rotated_data, pred_quat, gt_quat, labels = load_fineTune_data(filename)
    #
    # idx = np.arange(labels.shape[0])
    # np.random.shuffle(idx)
    # pred_quat = np.array(pred_quat)[idx, ...]
    # gt_quat = np.array(gt_quat)[idx, ...]
    #
    # print(partial_data.shape, rotated_data.shape)
    # print(pred_quat[100, :], gt_quat[100, :])
    from time import time
    K=5
    points = np.random.rand(100000,3)
    t0 = time()
    idx=farthest_sampling(points,K)
    print(idx)
    print(time()-t0)
    t0 = time()
    idx = farthest_sampling_idx(points,K)
    print(points[idx])

    print(time()-t0)
    t0 = time()
    #idx = DataOperator.farthest_sampling_idx(points, 10)

    index = np.zeros(K).astype(np.int32)
    DataOperator.farthest_point_samplin_func(points, index, K)
    print(points[index])
    print(time() - t0)

    # points = np.random.rand(16,10000,3)
    # t0 = time()
    # idx_nn=batch_farthest_sampling(points,K)#.astype(np.int32)
    # #print(points[:,idx_nn,:])
    # print(time() - t0)
    # #print(points[idx])
