#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import trimesh
from trimesh import io
from trimesh import util


mesh = trimesh.load_mesh('data/einstein_normalized.obj')

minball = 1/mesh.bounding_sphere.primitive.radius
print(1/minball)
minball=1.0695013

mesh.vertices = (mesh.vertices - mesh.centroid)*minball
io.export.export_mesh(mesh,'data/2.obj')