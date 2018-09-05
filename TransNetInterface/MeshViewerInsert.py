#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: testskip
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# Abstract: show mesh primitive
# Keywords: cone, arrow, sphere, cylinder, qt
# -----------------------------------------------------------------------------

"""
Test the fps capability of Vispy with meshdata primitive
"""
try:
    from sip import setapi
    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass

#from PyQt4 import QtGui, QtCore
#import sys

import numpy as np
from vispy import app,io, gloo,scene
from vispy.util.transforms import perspective, translate, rotate,ortho,scale
from vispy.geometry import meshdata as md
from vispy import keys
import sys,os
import GLSLOperator
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR,'ops'))
from ops import DataOperator
DEFAULT_COLOR = (0, 1, 1, 1)
# -----------------------------------------------------------------------------


class MyMeshData(md.MeshData):
    """ Add to Meshdata class the capability to export good data for gloo """
    def __init__(self, vertices=None, faces=None, edges=None,
                 vertex_colors=None, face_colors=None):
        md.MeshData.__init__(self, vertices=None, faces=None, edges=None,
                             vertex_colors=None, face_colors=None)

    def get_glTriangles(self):
        """
        Build vertices for a colored mesh.
            V  is the vertices
            I1 is the indices for a filled mesh (use with GL_TRIANGLES)
            I2 is the indices for an outline mesh (use with GL_LINES)
        """
        vtype = [('a_position', np.float32, 3),
                 ('a_normal', np.float32, 3),
                 ('a_color', np.float32, 4)]
        vertices = self.get_vertices()
        normals = self.get_vertex_normals()
        faces = np.uint32(self.get_faces())

        edges = np.uint32(self.get_edges().reshape((-1)))
        colors = self.get_vertex_colors()

        nbrVerts = vertices.shape[0]
        V = np.zeros(nbrVerts, dtype=vtype)
        V[:]['a_position'] = vertices
        V[:]['a_normal'] = normals
        V[:]['a_color'] = colors

        return V, faces.reshape((-1)), edges.reshape((-1))
# -----------------------------------------------------------------------------

class Canvas(app.Canvas):

    def __init__(self,title='MeshViewer',size=[640,480],mesh_name='einstein'):
        app.Canvas.__init__(self)
        self.size = size
        self.location = [0,0]
        self.title = title
        self.mesh_name = mesh_name
        # fovy, zfar params
        self.program = GLSLOperator.create_program('glsl/vert_mesh.glsl','glsl/frag_mesh.glsl')
        # self.program['u_model'] = self.model
        # self.program['u_view'] = self.view
        self.default_model = translate((0, 0, 0))
        self.model = self.default_model
        GLSLOperator.set_default_MVP(self.program)
        self.view = translate((0, 0, -4.3))
        self.program['u_view'] = self.view
        self.program['u_model'] = self.model
        self._button = None
        self.init_data()
        self.theta=0
        self.phi=0

    def update_model(self):
        # self.theta += .5
        # self.phi += .5
        # self.model = np.dot(rotate(self.theta, (0, 0, 1)),
        #                     rotate(self.phi, (0, 1, 0)))
        # self.program['u_model'] = self.model
        self.update()


    def set_model(self,model):
        self.model = model
        self.program['u_model'] = self.model
        self.update()

    def apply_resize(self,viewpoint):
        gloo.set_viewport(*viewpoint)
        self.projection = perspective(42.74, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        # self.projection = ortho(left=-1,right=1,bottom=-1,top=1,znear=0.10,zfar=1000)

        self.program['u_projection'] = self.projection
    # ---------------------------------
    def draw(self,x,y,width,height):

        #gloo.set_clear_color('white')
        self.location = [x,y]
        self.size = [width,height]
        self.apply_resize([x,y,width,height])

        gloo.set_state(blend=False, depth_test=True,
                       polygon_offset_fill=True)
        self.program['u_color'] = 1, 1, 1, 1
        self.program.draw('triangles', self.filled_buf)

        # Outline
        gloo.set_state(blend=True, depth_test=True,
                       polygon_offset_fill=False)
        gloo.set_depth_mask(False)
        self.program['u_color'] = 0, 0, 0, 1
        self.program.draw('lines', self.outline_buf)
        gloo.set_depth_mask(True)

    def init_data(self,normalization=True):
        verts, faces, normals, nothin = io.read_mesh("data/%s_normalized6.obj"%(self.mesh_name))

        if normalization:
           centroid = np.mean(verts, axis=0, keepdims=True)
           furthest_distance = np.amax(np.sqrt(np.sum((verts - centroid) ** 2, axis=-1)), keepdims=True)
          # verts = (verts - centroid) / furthest_distance

        print('--furthest_distance,',furthest_distance)
        #verts = verts*1.3
        meshdata = md.MeshData(vertices=verts, faces=faces)
        self.mesh = MyMeshData()
        self.mesh.set_vertices(verts)
        self.mesh.set_faces(faces)
        colors = np.tile(DEFAULT_COLOR, (verts.shape[0], 1))
        self.mesh.set_vertex_colors(colors)
        vertices, filled, outline = self.mesh.get_glTriangles()
        self.set_data(vertices, filled, outline)

        self.faces = self.mesh.get_faces()
        # print(faces.shape)
        face_normals = self.mesh.get_face_normals()
        vertices = self.mesh.get_vertices()
        # print(type(vertices))
        self.vertice_wrapper = np.ones([vertices.shape[0], 4])
        self.vertice_wrapper[:, :-1] = verts

        self.face_normals_wrapper = np.ones([face_normals.shape[0], 4])
        self.face_normals_wrapper[:, :-1] = face_normals

    # ---------------------------------
    def set_data(self, vertices, filled, outline):
        self.filled_buf = gloo.IndexBuffer(filled)
        self.outline_buf = gloo.IndexBuffer(outline)
        self.vertices_buff = gloo.VertexBuffer(vertices)
        self.program.bind(self.vertices_buff)
        self.update()

    def get_visible_pixel(self,intrinsic,center):

        print('---------------center',center)
        #center = 10.0*center
        view_mat = translate((center[0],center[1],center[2]))
        scale_factor = 0.0725#25#0.01*6.5#.32#.375
        scale_mat = scale((scale_factor,scale_factor,scale_factor))

        model_view = np.matrix(self.model)
        model_view = np.dot(model_view,scale_mat)
        model_view = np.dot(model_view,view_mat)

        modified_normal = np.dot(self.face_normals_wrapper, (model_view.I).T)
        modified_vertices = np.dot(self.vertice_wrapper, model_view)
        modified_vertices = np.array(modified_vertices)

        idx = np.where(modified_normal[:, 2] > 1e-7)
        visible_faces = np.array(self.faces[idx[0],:]).reshape(-1)
        points = modified_vertices[visible_faces]
        out_pixel = np.zeros([points.shape[0], 2]).astype(np.int32)
        DataOperator.proj_point2pixel_func(points, intrinsic, out_pixel)

        return out_pixel

    def on_mouse_press(self, event):
        if event.button == 1:
            self._button = event
        else:
            self._button = None

    def on_mouse_release(self, event):
        self._button = None

    def on_mouse_move(self, event):
        import math
        if event.button == 1:
             dx,dy = self._button.pos - event.pos
             nx = -dy
             ny = -dx
             scale = max(math.sqrt(nx * nx + ny * ny),1e-6)
             nx = nx / scale
             ny = ny / scale
             angle = scale * 0.01 * 80 / 90.0
             self.model = np.dot(rotate(angle, (nx, ny, 0)),self.model )
             self.program['u_model'] = self.model
             self.update()

    def on_mouse_wheel(self, event):
        if event.delta[1] > 0:
            self.model = np.dot(scale([1.5,1.5,1.5] ), self.model)
        else:
            self.model = np.dot(scale([0.8, 0.8, 0.8]), self.model)
        self.program['u_model'] = self.model
        self.update()

    def on_key_press(self, event):
        if event.key == keys.ESCAPE:
            exit(0)

# -----------------------------------------------------------------------------

# Start Qt event loop unless running in interactive mode.

if __name__ == '__main__':
    c = Canvas()
    app.run()

