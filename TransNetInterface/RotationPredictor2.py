'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import numpy as np
import tensorflow as tf
import os
import sys
from datetime import datetime
from time import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
POINTCLOUD_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'partialNet'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing/utils'))

import pc_util
import Tools3D
import PartialNet
import PointCloudOperator

POINT_NUM = 5000# 4096

class RotationPredictor:
    def __init__(self,model_name='einstein'):
        self.model_name = model_name
        filename = os.path.join(POINTCLOUD_DIR, 'Dataset/%s/%s_gt_4096.xyz'%(model_name,model_name))
        filename2 = os.path.join(POINTCLOUD_DIR, 'Dataset/%s/%s_gt_20480.xyz'%(model_name,model_name))
        self.pointcloud_gt = np.loadtxt(filename)
        self.pointcloud_gt_big =np.loadtxt(filename2)
        self.init_network()
        self.startime = time()
        self.predict_trans = None#np.eye(3)
    def init_network(self):
        self.model_path = os.path.join(ROOT_DIR,'log/PartialNetNew2/model.ckpt')
        print(self.model_path)
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(0)):
                 pointclouds_pl = tf.placeholder(tf.float32, shape=(1, None, 3))
                 pointclouds_pl_big = tf.placeholder(tf.float32, shape=(1, None, 3))
                 is_training_pl = tf.placeholder(tf.bool, shape=())
                 pred_angle = PartialNet.get_model(pointclouds_pl, is_training_pl)
                 #pred_angle = PartialNet.get_model(pointclouds_pl,pointclouds_pl_big, is_training_pl)

                 saver = tf.train.Saver()
            # Create a session
            config = tf.ConfigProto()
            # config.gpu_options.visible_device_list = '1'
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            # config.gpu_options.per_process_gpu_memory_fraction = 0.5
            self.sess = tf.Session(config=config)

             # Restore variables from disk.
            saver.restore(self.sess, self.model_path)
            print('Model restored.')

            self.ops = {'pointclouds_pl'   : pointclouds_pl,
                        'pointclouds_pl_big':pointclouds_pl_big,
                        'is_training_pl'   : is_training_pl,
                        'pred_angle'       : pred_angle}


    def eval_one_frame(self,points,isSampled=False,isNormalized=False,isSaving=False):
        is_training = False

        PC_PATH = os.path.join(POINTCLOUD_DIR, 'Dataset/%s/partialface/real_scan/'%(self.model_name))
        OUTPUT_FOLDER = os.path.join(PC_PATH, 'pred')
        #PointCloudOperator.get_pairwise_distance(points)
        if isSampled:
            if self.predict_trans is not None:
                point_temp = np.dot(points.copy(),np.eye(3))#self.predict_trans.T)
            else:
                point_temp = np.dot(points.copy(),np.eye(3))
            point_temp_max=np.max(point_temp,axis=0)
            point_temp_min=np.min(point_temp,axis=0)

            #print('point_temp_max:',point_temp_max)
            _clip_ = np.where((point_temp[:, 1] < point_temp_max[1]*0.9))
            points = points[_clip_]

           #  interv = int(points.shape[0]/1024)
           #  print(points.shape[0],interv)
           #  idx = np.arange(points.shape[0])
           #
           #  idx_n = [i for i in idx if i % interv == 0]
           #
            # idx_n = np.arange(points.shape[0])
            # np.random.shuffle(idx_n)
            #
            # if points.shape[0]<POINT_NUM:
            #     offset = POINT_NUM - points.shape[0]
            #     idx_n = np.concatenate([np.arange(points.shape[0]), np.random.randint(0, points.shape[0], size=offset)], axis=0)
            #     np.random.shuffle(idx_n)
            #
            #     #idx_n = np.random.randint(0,points.shape[0],size=POINT_NUM)
            #
            #
            # idx_n = idx_n[:POINT_NUM]
            #
            # np.random.shuffle(idx_n)
            #points = points[idx_n,...]

        if isNormalized:
            centroid = np.mean(points, axis=0, keepdims=True)
            print('------------centroid:',centroid)
            furthest_distance = np.amax(np.sqrt(np.sum((points - centroid) ** 2, axis=-1)),
                                        keepdims=True)
            points = (points - centroid) / furthest_distance
            distance = np.sqrt(np.sum(points** 2, axis=-1))
            med_distance = np.median(distance)
            max_distance = np.max(distance)
            scale = max_distance/med_distance * 0.8
            print('med_distance:',med_distance,'--max_distance:',max_distance)
            _clip_ = np.where(distance < scale*med_distance)
            points = points[_clip_]

            centroid = np.mean(points, axis=0, keepdims=True)
            print('------------centroid:', centroid)
            furthest_distance = np.amax(np.sqrt(np.sum((points - centroid) ** 2, axis=-1)),
                                        keepdims=True) * 0.89
            points = (points - centroid) / furthest_distance


        pointclouds_pl = np.expand_dims(points,axis=0)
       # pointclouds_pl[:,:,1:3] =  -pointclouds_pl[:,:,1:3]
        #pointclouds_pl[:,:,2] =  -pointclouds_pl[:,:,2]

        pointcloud_gt_big = np.expand_dims(self.pointcloud_gt_big,axis=0)

        feed_dict = {self.ops['pointclouds_pl']   : pointclouds_pl,
                     self.ops['pointclouds_pl_big']: pointcloud_gt_big,
                     self.ops['is_training_pl']   : is_training, }
        # loss_val, pred_angle = sess.run([ops['loss'], ops['pred_angle']], feed_dict=feed_dict)
        pred_angle = self.sess.run([self.ops['pred_angle']], feed_dict=feed_dict)
        pred_angle = np.squeeze(pred_angle)
        pred_angle = pred_angle[:4]
        print(pred_angle)
        transform_xyz = Tools3D.quaternion_To_rotation_matrix(pred_angle)
        transform_xyz = np.array(transform_xyz)

        if isSaving and (time() - self.startime>2.5):
            self.startime = time()
            point_cloud_transformed = np.matmul(self.pointcloud_gt, transform_xyz)

            point_input = np.squeeze(pointclouds_pl)
            points_gt = np.squeeze(self.pointcloud_gt)
            points_aligned = np.squeeze(point_cloud_transformed)

            info = 'Nothing'
            filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            pc_filename = os.path.join(OUTPUT_FOLDER, '%s.xyz' %(filename))
            np.savetxt(pc_filename, points, fmt='%0.6f')

            img_filename = os.path.join(OUTPUT_FOLDER, '%s.png' %(filename))
            #img_filename = os.path.join(OUTPUT_FOLDER, '1.png')
            pc_util.point_cloud_three_points(point_input, points_aligned, points_aligned, img_filename, info)
        # if self.predict_trans is not None:
        #     transform_xyz = 0.2*transform_xyz + 0.8*self.predict_trans
        self.predict_trans = transform_xyz
        return transform_xyz,points

if __name__ == "__main__":
    print('pid: %s' % (str(os.getpid())))
    pred = RotationPredictor()

    filename = os.path.join(POINTCLOUD_DIR, 'Dataset/Data/einstein_gt_s.xyz')
    pointcloud_gt = np.loadtxt(filename)

    pred.eval_one_frame(pointcloud_gt)
