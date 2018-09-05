#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

from mayavi import mlab
import numpy as np
import sys
import os
from vispy.util.transforms import perspective, translate, rotate,ortho,scale

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
POINT_CLOUD_DIR = os.path.dirname(ROOT_DIR)

sys.path.append(BASE_DIR)

import cube_grid_points

number = np.array([4,4,8])
bounding = np.array([-1,1,-1,1,-1,1])
points = cube_grid_points.cube_grid_points(number,bounding)

viewMatrix = rotate(180,axis=[1,0,0])
viewMatrix = np.matrix(viewMatrix)
projMatrix = perspective(63.3, 1.245, 2, 10)
projMatrix = np.matrix(projMatrix)

print(projMatrix.I)

print('viewMatrix\n',viewMatrix)
print('projMatrix\n',projMatrix)

points2 = np.ones((points.shape[0],4))
points2[:,:-1] = points
unproj_pos = np.dot(points2,projMatrix.I)
view_pose = unproj_pos/unproj_pos[:,3]
view_pose[:,1:3] = -view_pose[:,1:3];
print(unproj_pos[5:10,:])
print(view_pose[5:10,:])
model_pos = np.dot(view_pose,viewMatrix.I)
model_pos = view_pose

filename = os.path.join(POINT_CLOUD_DIR,'Dataset/sample_transformation','transformation.xyz')
np.savetxt(filename,model_pos[:,:-1],fmt='%.6f')
print(model_pos)
model_pos = points

handler = mlab.points3d(model_pos[:,0],model_pos[:,1],model_pos[:,2])

mlab.axes(handler, color=(.7, .7, .7),ranges=(0, 1, 0, 1, 0, 1), xlabel='X', ylabel='Y',
            zlabel='Z',
            x_axis_visibility=True, z_axis_visibility=True)


mlab.show()