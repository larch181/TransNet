#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import numpy as np
def cube_grid_points(number,bounding):
    '''
    :param number: 3 interger,sampling num along x,y,z
    :param bounding: 3*2 sampling bounding along x,y,z
    :return: sampling point
    '''
    l,m,n = number
    points = np.zeros((l*m*n,3))
    diff_x = (bounding[1]-bounding[0])/(l+1)
    diff_y = (bounding[3]-bounding[2])/(m+1)
    diff_z = (bounding[5]-bounding[4])/(n+1)

    x = np.arange(bounding[0] + diff_x, bounding[1], diff_x)
    y = np.arange(bounding[2] + diff_y, bounding[3], diff_y)
    z = np.arange(bounding[4] + diff_z, bounding[5], diff_z)

    interv = np.arange(1,n+1)**2
    interv = bounding[4]+interv/np.sum(interv) * (bounding[5]-bounding[4])
    print(interv)

    for i in range(n):

        for j in range(m):
            points[l * m * i + l * j:l * m * i + l * j + l, 0] = x
            points[l * m * i + l * j:l * m * i + l * j + l, 1] = y[j]
            points[l * m * i + l * j:l * m * i + l * j + l, 2] = z[i]

    return points

if __name__=="__main__":
    number = np.array([3,4,5])
    bounding = np.array([-1,1,-1,1,-1,1])
    points = cube_grid_points(number,bounding)
    print(points)