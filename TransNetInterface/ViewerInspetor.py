from vispy import gloo
from vispy import app,keys,util,visuals
from vispy.util.transforms import perspective, translate, rotate,ortho,scale
from vispy.visuals.transforms import STTransform, NullTransform
import numpy as np
import time
import os
import cv2
#vispy.app.use_app(backend_name="PyQt5", call_reuse=True)
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(BASE_PATH)
POINTCLOUD_PATH = os.path.dirname(ROOT_PATH)
sys.path.append(BASE_PATH)
sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH,'dataProcessing/utils'))
util.set_data_dir(BASE_PATH)
import Tools3D

PC_SAVE_PATH = os.path.join(POINTCLOUD_PATH,'Dataset/Data/partialface/real_scan/raw/')
from RotationPredictor import RotationPredictor
from MeshViewerInsert import Canvas as MeshViewerCanvas
from RealSenseOperator import RealSenseOperator
import GLSLOperator
#from Tracker import Tracker
from Tracker_YOLO import Tracker_YOLO as Tracker
# ------------------------------------------------------------ Canvas class ---
class Canvas(app.Canvas):

    def __init__(self,size=[640,480]):
        app.Canvas.__init__(self,title='PointCloudVisulize', keys='interactive', size=size)
        self.model_name = 'pwd'
        self.model_name = 'einstein'
        self.mesh_viewer = MeshViewerCanvas(mesh_name=self.model_name)
        self.predictor = RotationPredictor(model_name=self.model_name)
        self.program = GLSLOperator.create_program('glsl/vert_pc.glsl','glsl/frag_pc.glsl')
        self.program = GLSLOperator.set_default_MVP(self.program)
        self.view = translate((0, 0, -6.0))
        self.default_view = self.view.copy()
        self.is_reInit = False
        self.rotate_button = None
        self.scale_button = None
        self.default_angle = 0
        self.model = rotate(self.default_angle , axis=[0, 1, 0])

        self.model = np.dot(scale([2, 2, 2]),self.model)
        self.default_model = self.model.copy()
        self.program['u_model'] = self.model
        self.projection = np.eye(4, dtype=np.float32)
        self.program['u_linewidth'] = 1.0
        self.program['u_antialias'] = 1.0
        self.program['u_size'] = 1
        self._starttime = time.time()
        self.save_pc = False
        self.save_index = 0
        self.obj_scale_factor = 2
        self.image_scale_factor = 4

        self.rsOperator = RealSenseOperator()
        self.tracker = Tracker()
        self.update_pointcloud()

        gloo.set_state('translucent', clear_color='white')
        self.init_show_info()
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def init_show_info(self):
        l_pos = np.array([
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])

        self.cross_eye_line = visuals.LineVisual(pos=l_pos,
                                                 color=(1.0, 0.0, 0.0, 1),
                                                 method='gl')
        self.texts = [
            visuals.TextVisual('Left', bold=True, font_size=16,
                               color='r', pos=(0, 0)),
            visuals.TextVisual('Front', bold=True, font_size=16,
                               color='r', pos=(0, 0)),
            visuals.TextVisual('Top', bold=True, font_size=16,
                               color='r', pos=(0, 0)),
            visuals.TextVisual('Mesh', bold=True, font_size=16,
                               color='r', pos=(0, 0))
        ]
        for text in self.texts:
            text.transform = NullTransform()

    def on_key_press(self, event):
        if event.key == keys.F1:
            self.obj_scale_factor = self.obj_scale_factor+0.2
            self.obj_scale_factor = min(self.obj_scale_factor,4.0)
        if event.key == keys.F2:
            self.obj_scale_factor = self.obj_scale_factor-0.2
            self.obj_scale_factor = max(self.obj_scale_factor,1.0)
        # if event.key == keys.F1:
        #     self.is_reInit = True
        #     self.tracker.isNeedUpdate = False
        #     self.tracker.reinit_tracking(self.color_image)
        #     self.is_reInit = False
        #     self.tracker.isNeedUpdate = True
        #     print('yes')
        # if event.text == ' ':
        #     print('yes')
        #     #self.save_pc = True
        #     #self.save_index = self.save_index + 1
        #     # if self.timer.running:
        #     #     self.timer.stop()
        #     # else:
        #     #     self.timer.start()
        if event.key == keys.ENTER:
            print('yes')
            np.savetxt('data/points.xyz',self.points,fmt='%.6f')


    def update_tracking(self):
        if not self.tracker.isInited:
            self.tracker.init_tracking()
        else:
            self.tracker.tracking(self.rsOperator.color_image)
        if self.tracker.isNeedUpdate:
            self.points,self.points_center = self.rsOperator.extract_depth_point_with_seg(self.tracker.bbox)

    def show_info(self):
        intrinsic = np.array([389.284, 389.284, 320.569, 238.517])
        intrinsic = np.array([613.113, 613.205, 316.667, 243.441])
        out_pixel = self.mesh_viewer.get_visible_pixel(intrinsic,self.points_center)

        self.rsOperator.show_current_image(self.tracker.bbox,out_pixel,self.obj_scale_factor,self.image_scale_factor)

    def save_pointcloud(self,points,color_image):
        import math
        rotationVec =  np.array([0,0,1])
        rotationAngle = np.random.rand(1)*math.pi

        rotMatrix = Tools3D.rotation_vector_To_rotation_matrix(rotationVec, rotationAngle)

        points = np.matmul(points,rotMatrix)

        np.savetxt(os.path.join(PC_SAVE_PATH, 'real_%d.xyz' % (self.save_index)), points, fmt='%0.6f')
        cv2.imwrite(os.path.join(PC_SAVE_PATH, 'real_%d.jpg' % (self.save_index)), color_image)
        self.save_pc = False

    def predict_pointcloud(self):
        if self.mesh_viewer is not None and self.tracker.isNeedUpdate:
            transform_xyz,self.points = self.predictor.eval_one_frame(self.points)
            _transform = np.eye(4)
            _transform[:-1,:-1] = transform_xyz

            self.mesh_viewer.model = _transform
            self.mesh_viewer.program['u_model'] = self.mesh_viewer.model
            self.mesh_viewer.update()

    def update_pointcloud(self):
        self.rsOperator.capture_frame()
        self.update_tracking()
        self.predict_pointcloud()
        self.show_info()
        n = self.points.shape[0]  # points.shape[0]
        data = np.zeros(n, [('a_position', np.float32, 3),
                            ('a_texcoord', np.float32, 2),
                            ('a_bg_color', np.float32, 4),
                            ('a_fg_color', np.float32, 4),
                            ('a_size', np.float32, 1)])
        data['a_position'] = self.points
        color = np.repeat((np.array([0, 0, 0, 1]).reshape(1, 4)), n, axis=0)
        data['a_bg_color'] = np.repeat((np.array([0, 0, 0, 1]).reshape(1, 4)), n, axis=0)
        data['a_fg_color'] = color
        data['a_size'] = 1 * self.pixel_scale
        self.program.bind(gloo.VertexBuffer(data))
        self._starttime = time.time()

    def on_draw(self, event):

        if self.is_reInit is not True:
            self.update_pointcloud()

        width, height = self.physical_size[0] // 2, self.physical_size[1] // 2
        gloo.clear()
        gloo.set_clear_color('white')
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        gloo.set_viewport(*vp)

        self.cross_eye_line.draw()
        for i in range(len(self.texts)):
            self.texts[i].transforms.configure(canvas=self, viewport=vp)
            self.texts[i].pos = [width * (i // 2) + 40, height * (i % 2) + 90]
            self.texts[i].anchors = ('top', 'right')
            self.texts[i].draw()


        for i in range(3):
            self.default_view = self.view.copy()
            vp = (width * (i // 2), height * ((i + 1) // 2), width, height)
            gloo.set_viewport(*vp)
            self.program['u_view'] = np.dot(rotate(-90 * ((i + 1) // 2), axis=[i // 2, 1 - i // 2, 0]), self.view)
            self.program.draw('points')

        self.mesh_viewer.draw(width, 0, width, height)


    def on_timer(self, event):
        self.update()

    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(63.3, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection

    def on_resize(self, event):

        self.apply_zoom()

    def on_mouse_wheel(self, event):
        if event.delta[1] > 0:
            self.model = np.dot(scale([1.5,1.5,1.5] ), self.model)
        else:
            self.model = np.dot(scale([0.8, 0.8, 0.8]), self.model)
        self.program['u_model'] = self.model
        self.update()

    def on_mouse_press(self, event):
        if event.button == 1:
            self.rotate_button = event
        else:
            self.rotate_button = None

    def on_mouse_release(self, event):
        self.rotate_button = None
        self.rotate_button = None

    def on_mouse_move(self, event):
        import math
        if self.rotate_button != None and event.button == 1:
             dx,dy = event.pos-self.rotate_button.pos
             nx = dy
             ny = dx
             scale = max(math.sqrt(nx * nx + ny * ny),1e-6)
             nx = nx / scale
             ny = ny / scale
             angle = scale * 0.03 * 60 / 90.0
             self.model = np.dot(rotate(angle, (nx, ny, 0)),self.model )
             self.program['u_model'] = self.model
             self.update()

    def on_mouse_double_click(self,event):
        self.model = self.default_model.copy()
        self.program['u_model'] = self.model
        self.update()




if __name__ == '__main__':
    c = Canvas()
    app.run()
