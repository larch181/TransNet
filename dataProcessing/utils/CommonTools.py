#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import numpy as np


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


def Normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)

    #mu = np.average(data)
    #sigma = np.std(data)
    #return (data-mu)/(sigma)
    print(mx,mn)
    return (data-mn)/(mx-mn+1e-6)
    #return 1.0 / (1 + np.exp(-data))

    #return [(float(i) - mn) / (mx - mn) for i in data]


if __name__ == '__main__':
    batch_features = np.arange(12).reshape([3, 4])
    print(Normalize(batch_features))

    pairwise_distance = get_pairwise_distance(batch_features)

    print(pairwise_distance)
