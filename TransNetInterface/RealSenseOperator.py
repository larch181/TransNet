#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import pyrealsense2 as rs
import numpy as np
import os
import cv2
import sys
from time import time
import pcl

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
sys.path.append(os.path.join(BASE_PATH, 'ops/TransNetInterface/ops'))
import DataAnalyze
from SaliencyCut import SaliencyCut

class RealSenseOperator:

    def __init__(self,size=[640,480]):
        self.pipe = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipe.start(config)
        # Declare pointcloud object, for calculating pointclouds and texture mappings
        self.pc = rs.pointcloud()
        # We want the points object to be persistent so we can display the last cloud when a frame drops
        self.points = rs.points()
        self.init_sensor()
        print('Sensor Initilization Done!')
        self.extract_instrics()
        self.saliency_cut = SaliencyCut()

    def init_sensor(self):
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        #depth_sensor.set_option(rs.option.depth_units, 0.0001)
        self.depth_scale = depth_sensor.get_depth_scale()

        # range2 = depth_sensor.get_option_range(rs.option.visual_preset)
        # preset = range2.max-1
        # print('motion_range:',preset)

        preset = 4 #3 High Accuracy / 4 High Density
        depth_sensor.set_option(rs.option.visual_preset, preset)
        preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset, preset)
        print(preset_name)
        power_range = depth_sensor.get_option_range(rs.option.laser_power)
        print('power_range:',power_range)
        depth_sensor.set_option(rs.option.laser_power,power_range.max)

        laser_power = depth_sensor.get_option(rs.option.laser_power)
        print(laser_power)
        #infrared depth_sensor.set_option(rs.option.)
        #depth_sensor.set_option(rs.option.emitter_enabled,True)

        print("Depth Scale is: ", self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters_f = 0.43  # 1 meter
        clipping_distance_in_meters_n = 0.25  # 1 meter

        clipping_distance_f = clipping_distance_in_meters_f / self.depth_scale
        clipping_distance_n = clipping_distance_in_meters_n / self.depth_scale

        print("Depth Distance is: ", np.array([clipping_distance_f, clipping_distance_n]))
        self.clipping_distance = np.array([clipping_distance_f, clipping_distance_n])



    def fetch_frame(self):
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Wait for the next set of frames from the camera
        frames = self.pipe.wait_for_frames()
        # Align the depth frame to color frame
        self.depth2 = frames.get_depth_frame()
        frames = align.process(frames)
        # Fetch color and depth frames
        self.depth = frames.get_depth_frame()
        self.color = frames.get_color_frame()
        # Convert images to numpy arrays
        self.depth_data = np.asanyarray(self.depth.get_data())
        self.depth_image2 = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(self.depth2.get_data()), alpha=0.03), cv2.COLORMAP_JET)
        self.depth_image = cv2.applyColorMap(cv2.convertScaleAbs( self.depth_data, alpha=0.03), cv2.COLORMAP_JET)
        self.color_image = np.asanyarray(self.color.get_data())

    def extract_instrics(self):
        self.fetch_frame()

        # Intrinsics & Extrinsics
        self.depth_intrin = self.depth.profile.as_video_stream_profile().intrinsics
        self.color_intrin = self.color.profile.as_video_stream_profile().intrinsics
        self.depth_to_color_extrin = self.depth.profile.get_extrinsics_to(self.color.profile)
        self.color_to_depth_extrin = self.color.profile.get_extrinsics_to(self.depth.profile)

        print('depth_intrin\n', self.depth_intrin)
        print('color_intrin\n', self.color_intrin)
        print('depth_to_color_extrin\n', self.depth_to_color_extrin.rotation)
        print('color_to_depth_extrin\n', self.color_to_depth_extrin.rotation)
        print('deptp fov\n', rs.rs2_fov(self.depth_intrin))
        print('color fov\n', rs.rs2_fov(self.color_intrin))

    def capture_frame(self):

        self.fetch_frame()
        # Wait for a coherent pair of frames: depth and color
        if not self.depth or not self.color:
            return None

        #vertices, tex_coords = self.extract_pointcloud()

    def get_image_location(self,point):

        intrinsic = np.array([389.284, 389.284, 320.569, 238.517])
        intrinsic = np.array([613.113, 613.205, 316.667, 243.441])

        x = intrinsic[0]* (-point[0] / point[2]) + intrinsic[2]
        y = intrinsic[1]* (point[1] / point[2]) + intrinsic[3]

        return np.array([x,y]).astype(np.int32)

    def find_contour(self,src):
        """src为原图"""
        ROI = np.zeros(src.shape, np.uint8)  # 感兴趣区域ROI
        proimage = src.copy()  # 复制原图
        """提取轮廓"""
        proimage = cv2.cvtColor(proimage, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
        proimage = cv2.adaptiveThreshold(proimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 7)
        proimage, contours, hierarchy = cv2.findContours(proimage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # 提取所有的轮廓
        """ROI提取"""
        cv2.drawContours(ROI, contours, 1, (255, 255, 255), -1)  # ROI区域填充白色，轮廓ID1
        ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
        ROI = cv2.adaptiveThreshold(ROI, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 7)  # 自适应阈值化
        #imgroi = cv2.bitwise_and(ROI, proimage)  # 图像交运算 ，获取的是原图处理——提取轮廓后的ROI
        imgroi = cv2.bitwise_and(src,src,mask=ROI)
        # imgroi = ROI & src 无需灰度+阈值，获取的是原图中的ROI
        return imgroi

    def pixel_segment2(self, img, bbox):
        height, width, _ = img.shape
        factor = 0.05
        x1 = max(int(bbox[0] - factor * bbox[2] / 2), 0)
        y1 = max(int(bbox[1] - factor * bbox[3] / 2), 0)
        x2 = min(int(bbox[0] + bbox[2] + factor * bbox[2] / 2), height)
        y2 = min(int(bbox[1] + bbox[3] + factor * bbox[3] / 2), width)

        roi_img = img[y1:y2, x1:x2, :].copy()


        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv2.watershed(roi_img, markers)
        roi_img[markers == -1] = [255, 0, 0]
        mask2 = np.zeros(roi_img.shape[:2], np.uint8)
        mask2[markers == -1] = 1
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y1:y2, x1:x2] = mask2
        img = img * mask[:, :, np.newaxis]
        return img

    def pixel_segment(self,img,bbox):

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        height,width,_ = img.shape
        factor = 0.05
        x1 = max(int(bbox[0] - factor * bbox[2] / 2), 0)
        y1 = max(int(bbox[1] - factor * bbox[3] / 2), 0)
        x2 = min(int(bbox[0]+bbox[2] + factor * bbox[2] / 2), height)
        y2 = min(int(bbox[1]+bbox[3] + factor * bbox[3] / 2), width)

        roi_img = img[y1:y2,x1:x2,:]
        rect = (1,1,roi_img.shape[1]-1,roi_img.shape[0]-1)
        mask = np.zeros(roi_img.shape[:2], np.uint8)
        cv2.grabCut(roi_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y1:y2,x1:x2] = mask2
        img = img * mask[:, :, np.newaxis]
        return img
        #return img
        #cv2.imshow(img)
        #cv2.grabCut(img,cv2.GC_PR_FGD,bbox,np.zeros([1,65]),10,cv2.GC_INIT_WITH_RECT)


    def show_current_image(self, bbox, out_pixel, obj_scale_factor = 2, image_scale_factor=4):

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        overlay_image = self.color_image.copy()
        height,width,_ = overlay_image.shape



        print('-----------------------',np.max(self.color_image))
        # self.pixel_segment2(overlay_image.copy(),bbox)

        cv2.rectangle(self.color_image, p1, p2, (255, 0, 0), 2, 1)

        cv2.rectangle(self.depth_image, p1, p2, (255, 255, 255), 2, 1)
        cv2.circle(self.depth_image, p1, 20, (255, 0, 0), 2, 1)
        cv2.circle(self.depth_image, p2, 20, (0, 255, 0), 2, 1)

        # Stack both images horizontally
        images = np.hstack((self.color_image,  self.depth_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        center = self.get_image_location(self.centroid)
        cv2.circle(overlay_image, (center[0],center[1]), 20, (0, 255, 0), 2, 1)

        interv = 3
        for i in range(int(out_pixel.shape[0] / interv)):
            cv2.line(overlay_image, (out_pixel[interv * i, 0], out_pixel[interv * i, 1]),
                     (out_pixel[interv * i + 1, 0], out_pixel[interv * i + 1, 1]), [255, 255, 255], 1, lineType=cv2.LINE_AA)
            cv2.line(overlay_image, (out_pixel[interv * i, 0], out_pixel[interv * i, 1]),
                     (out_pixel[interv * i + 2, 0], out_pixel[interv * i + 2, 1]), [255, 255, 255], 1, lineType=cv2.LINE_AA)
            cv2.line(overlay_image, (out_pixel[interv * i+1, 0], out_pixel[interv * i+1, 1]),
                     (out_pixel[interv * i + 2, 0], out_pixel[interv * i + 2, 1]), [255, 255, 255], 1, lineType=cv2.LINE_AA)


        print(overlay_image.shape)
        x = obj_scale_factor*((p1[0]+p2[0])/2 - width/2)
        y = obj_scale_factor*((p1[1]+p2[1])/2 - height/2)

        M = np.float32([[obj_scale_factor, 0, -(obj_scale_factor - 1) * width / 2-x], [0, obj_scale_factor, -(obj_scale_factor - 1) * height / 2-y]])
        overlay_image = cv2.warpAffine(overlay_image, M, (overlay_image.shape[1],overlay_image.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE )
        overlay_image = cv2.resize(overlay_image, (image_scale_factor * overlay_image.shape[1], image_scale_factor * overlay_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow('overlay_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('overlay_image', overlay_image)

        cv2.namedWindow('ROI image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('ROI image',self.ROI_image)

    def extract_pointcloud(self, normalized=False):
        # Tell pointcloud object to map to this color frame
        self.pc.map_to(self.color)
        # Generate the pointcloud and texture mappings
        self.points = self.pc.calculate(self.depth)

        vertices = np.asfortranarray(self.points.get_vertices())
        depth_image = np.asanyarray(self.depth.get_data()).reshape(-1)
        tex_coords = np.asfortranarray(self.points.get_texture_coordinates())

        point_removed = np.where((depth_image < self.clipping_distance[0]) & (depth_image > self.clipping_distance[1]))

        _vertices = vertices[point_removed]
        _tex_coords = tex_coords[point_removed]

        vertices = np.zeros((_vertices.shape[0], 3), dtype=np.float32)
        tex_coords = np.zeros((_vertices.shape[0], 2), dtype=np.float32)
        count = DataAnalyze.fetch_data_vert_frag(_vertices.astype(tuple), _tex_coords.astype(tuple), vertices,
                                                 tex_coords)

        # if normalized:
        #     centroid = np.mean(vertices[:, :-1], axis=0, keepdims=True)
        #     vertices[:, :-1] = vertices[:, :-1] - centroid
        if normalized:
            centroid = np.mean(vertices, axis=0, keepdims=True)
            furthest_distance = np.amax(np.sqrt(np.sum((vertices - centroid) ** 2, axis=-1)), keepdims=True)

            vertices = (vertices - centroid) / furthest_distance
        return vertices, tex_coords

    def remove_plane_point(self,points):
        print('remove_plane_point------------')
        cloud = pcl.PointCloud(points.astype(np.float32))
        print(cloud.size)
        fil = cloud.make_passthrough_filter()
        fil.set_filter_field_name("z")
        fil.set_filter_limits(0.3, 0.5)
        cloud_filtered = fil.filter()
        print(cloud_filtered.size)
        seg = cloud_filtered.make_segmenter_normals(ksearch=5)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_normal_distance_weight(0.1)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.02)
        indices, model = seg.segment()
        cloud_remained = cloud_filtered.extract(indices, negative=True)
        return np.asarray(cloud_remained)


    def extract_depth_point_with_seg(self, color_pixel):
        self.ROI_image,bbox = self.saliency_cut.segment(self.color_image.copy(), color_pixel)

        left_top_depth_pixel = [color_pixel[0], color_pixel[1]]
        right_down_depth_pixel = [color_pixel[0] + color_pixel[2], color_pixel[1] + color_pixel[3]]

        lx = int(bbox[0])
        ly = int(bbox[1])
        rx = int(bbox[2])
        ry = int(bbox[3])

        lx = int(left_top_depth_pixel[0])
        ly = int(left_top_depth_pixel[1])
        rx = int(right_down_depth_pixel[0])
        ry = int(right_down_depth_pixel[1])

        num = abs((rx - lx) * (ry - ly))
        points = np.zeros([num, 3]).astype(np.float32)
        index = 0

        for i in range(lx, rx):
            for j in range(ly, ry):
                if self.ROI_image[j,i] == 255:
                    scale = self.depth.get_distance(i, j)
                    if scale > 0.2 and scale < 0.5:
                        point = rs.rs2_deproject_pixel_to_point(self.color_intrin, [i, j], scale)
                        # point = rs.rs2_transform_point_to_point(self.depth_to_color_extrin, point)
                        points[index, 0] = point[0]
                        points[index, 1] = point[1]
                        points[index, 2] = point[2]
                        index = index + 1

        points = points[0:index, :]
        # points = self.remove_plane_point(points)
        points[:, 1:3] = -points[:, 1:3]

        print('max,min:', np.max(points[:, 0]), np.min(points[:, 0]))
        print('max,min:', np.max(points[:, 1]), np.min(points[:, 1]))

        idx = np.where(points[:, 1] > np.min(points[:, 1]) / 1.1)
        points = points[idx[0], ...]

        centroid = np.mean(points, axis=0, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((points - centroid) ** 2, axis=-1)), keepdims=True)
        print('furthest_distance:', furthest_distance)

        points = (points - centroid) / furthest_distance
        self.centroid = np.squeeze(centroid)

        # self.centroid = rs.rs2_transform_point_to_point(self.color_to_depth_extrin,  [self.centroid[0],self.centroid[1],self.centroid[2]])

        # centroid[2] = 1.47*centroid[2]
        return points, self.centroid

    def extract_depth_point(self, color_pixel):

        left_top_depth_pixel = [color_pixel[0], color_pixel[1]]
        right_down_depth_pixel = [color_pixel[0] + color_pixel[2], color_pixel[1] + color_pixel[3]]

        # color_pixel1 = [color_pixel[0], color_pixel[1]]
        # color_pixel2 = [color_pixel[0] + color_pixel[2], color_pixel[1] + color_pixel[3]]
        # self.depth_to_color_extrin = self.depth.profile.get_extrinsics_to(self.color.profile)
        # self.color_to_depth_extrin = self.color.profile.get_extrinsics_to(self.depth.profile)
        # left_top_color_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, color_pixel1, 1.0)
        # left_top_depth_point = rs.rs2_transform_point_to_point(self.color_to_depth_extrin, left_top_color_point)
        # left_top_depth_pixel = rs.rs2_project_point_to_pixel(self.depth_intrin, left_top_depth_point)
        #
        # right_down_color_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, color_pixel2, 1.0)
        # right_down_depth_point = rs.rs2_transform_point_to_point(self.color_to_depth_extrin, right_down_color_point)
        # right_down_depth_pixel = rs.rs2_project_point_to_pixel(self.depth_intrin, right_down_depth_point)


        height, width = self.depth_data.shape
        print(height,width)
        lx = int(left_top_depth_pixel[0])
        ly = int(left_top_depth_pixel[1])
        rx = int(right_down_depth_pixel[0])
        ry = int(right_down_depth_pixel[1])

        depth_info = self.depth_data*self.depth_scale
        scale_max = np.max(depth_info[ly:ry,lx:rx])
        scale_mean = np.median(depth_info[ly:ry,lx:rx])
        print('----------------------')

        # x = int((rx+lx)/2)
        # y = int((ry+ly)/2)
        # scale = self.depth.get_distance(x,y)
        # self.centroid = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x, y], scale)
        # self.centroid = rs.rs2_transform_point_to_point(self.depth_to_color_extrin,  self.centroid)
        #
        # self.centroid = np.array(self.centroid)
        # self.centroid[1:3] = -self.centroid[1:3]

        num = abs((rx - lx) * (ry - ly))
        points = np.zeros([num, 3]).astype(np.float32)
        index = 0

        for i in range(lx, rx):
            for j in range(ly, ry):
                scale = self.depth.get_distance(i,j)
                if scale > 0.2 and scale < 0.5:
                    point = rs.rs2_deproject_pixel_to_point(self.color_intrin, [i, j], scale)
                    #point = rs.rs2_transform_point_to_point(self.depth_to_color_extrin, point)
                    points[index, 0] = point[0]
                    points[index, 1] = point[1]
                    points[index, 2] = point[2]
                    index = index + 1

        points = points[0:index, :]
        #points = self.remove_plane_point(points)
        points[:, 1:3] = -points[:, 1:3]

        print('max,min:', np.max(points[:, 0]), np.min(points[:, 0]))
        print('max,min:', np.max(points[:, 1]), np.min(points[:, 1]))

        idx = np.where(points[:, 1]>np.min(points[:, 1])/1.1)
        points = points[idx[0],...]

        centroid = np.mean(points, axis=0, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((points - centroid) ** 2, axis=-1)), keepdims=True)
        print('furthest_distance:', furthest_distance)

        points = (points - centroid) / furthest_distance
        self.centroid = np.squeeze(centroid)


        #self.centroid = rs.rs2_transform_point_to_point(self.color_to_depth_extrin,  [self.centroid[0],self.centroid[1],self.centroid[2]])

        #centroid[2] = 1.47*centroid[2]
        return points, self.centroid

    def extract_based_pixel(self, color_pixel):
        self.depth_to_color_extrin = self.depth.profile.get_extrinsics_to(self.color.profile)
        self.color_to_depth_extrin = self.color.profile.get_extrinsics_to(self.depth.profile)

        # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # depth = depth_frame.get_distance(j, i)
        # depth_point = rs.rs2_deproject_pixel_to_point(
        #     depth_intrin, [j, i], depth)
        # text = "%.5lf, %.5lf, %.5lf\n" % (
        #     depth_point[0], depth_point[1], depth_point[2])

        color_pixel1 = [color_pixel[0], color_pixel[1]]
        color_pixel2 = [color_pixel[0] + color_pixel[2], color_pixel[1] + color_pixel[3]]


        left_top_color_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, color_pixel1, 1.0)
        left_top_depth_point = rs.rs2_transform_point_to_point(self.color_to_depth_extrin, left_top_color_point)
        left_top_depth_pixel = rs.rs2_project_point_to_pixel(self.depth_intrin, left_top_depth_point)

        right_down_color_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, color_pixel2, 1.0)
        right_down_depth_point = rs.rs2_transform_point_to_point(self.color_to_depth_extrin, right_down_color_point)
        right_down_depth_pixel = rs.rs2_project_point_to_pixel(self.depth_intrin, right_down_depth_point)

        depth_image = np.asanyarray(self.depth.get_data()).T
        height, width = depth_image.shape
        lx = int(left_top_depth_pixel[0])
        ly = int(left_top_depth_pixel[1])
        rx = int(right_down_depth_pixel[0])
        ry = int(right_down_depth_pixel[1])

        depth_bbox = [left_top_depth_pixel[0], left_top_depth_pixel[1], right_down_depth_pixel[0],
                      right_down_depth_pixel[1]]
        print(color_pixel)
        print(depth_bbox)
        self.show_depth_bounding(depth_bbox)


        scale1 = depth_image[lx,ly]
        scale2 = depth_image[rx,ry]

        scale_max = np.max(depth_image[lx:rx,ly:ry])
        scale_mean = np.median(depth_image[lx:rx,ly:ry])
        #scale_mean = np.mean(depth_image[lx:rx,ly:ry])
        #rs.depth_frame.get_distance(x, y)
        high = max(scale1,scale2)
        low = min(scale1, scale2)
        num = (rx-lx)*(ry-ly)
        points = np.zeros([num,3]).astype(np.float32)
        index=0
        for i in range(lx,rx):
            for j in range(ly,ry):
                scale = depth_image[i, j]
                if scale > 0.01 and scale < scale_mean *1.1:
                    point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [i,j], scale)
                    points[index,0] = point[0]
                    points[index,1] = point[1]
                    points[index,2] = point[2]

                    index = index + 1

        points= points[0:index,:]*depth_scale
        #print(point.shape)
        # centroid = np.mean(points, axis=0, keepdims=True)
        # furthest_distance = np.amax(np.sqrt(np.sum((points - centroid) ** 2, axis=-1)), keepdims=True)
        #
        # points = (points - centroid) / furthest_distance
        # points[:, 1:3] = -points[:, 1:3]

        return points

    def extract_based_pixel2(self, color_pixel):
        self.depth_to_color_extrin = self.depth.profile.get_extrinsics_to(self.color.profile)
        self.color_to_depth_extrin = self.color.profile.get_extrinsics_to(self.depth.profile)

        # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        color_pixel1 = [color_pixel[0], color_pixel[1]]
        color_pixel2 = [color_pixel[0] + color_pixel[2], color_pixel[1] + color_pixel[3]]

        left_top_color_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, color_pixel1, 1.0)
        left_top_depth_point = rs.rs2_transform_point_to_point(self.color_to_depth_extrin, left_top_color_point)
        left_top_depth_pixel = rs.rs2_project_point_to_pixel(self.depth_intrin, left_top_depth_point)

        right_down_color_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, color_pixel2, 1.0)
        right_down_depth_point = rs.rs2_transform_point_to_point(self.color_to_depth_extrin, right_down_color_point)
        right_down_depth_pixel = rs.rs2_project_point_to_pixel(self.depth_intrin, right_down_depth_point)

        depth_image = np.asanyarray(self.depth.get_data())
        height, width = depth_image.shape
        lx = int(left_top_depth_pixel[0])
        ly = int(left_top_depth_pixel[1])
        rx = int(right_down_depth_pixel[0])
        ry = int(right_down_depth_pixel[1])

        index_max = np.argmax(depth_image[lx:rx + 1, ly:ry + 1])
        index_min = np.argmin(depth_image[lx:rx + 1, ly:ry + 1])
        print('index_max', index_max)
        print('index_min', index_min)
        index_max = lx * width + ly + index_max
        index_min = lx * width + ly + index_min

        pixel_max= [int(index_max / width), int(index_max % width)]
        pixel_min = [int(index_min / width), int(index_min % width)]
        scale1 = depth_image[lx, ly]
        scale2 = depth_image[rx, ry]
        scale3 = depth_image[pixel_max[0], pixel_max[1]]
        scale4 = depth_image[pixel_min[0], pixel_min[1]]

        print('scale1', scale1)
        print('scale2', scale2)
        print('scale3', scale3)
        print('scale4', scale4)
        point1 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, left_top_depth_pixel, scale1)
        point2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, right_down_depth_pixel, scale2)
        point3 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, pixel_max, scale3)
        point4 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, pixel_min, scale4)

        l1 = min(point1[0], point2[0])* depth_scale
        r1 = max(point1[0], point2[0])* depth_scale
        l2 = min(point1[1], point2[1])* depth_scale
        r2 = max(point1[1], point2[1])* depth_scale
        l3 = min(point1[2], point2[2])* depth_scale
        r3 = max(point1[2], point2[2])* depth_scale

        bounding_cube = [l1,r1,l2,r2,l3,r3]


        depth_bbox = [left_top_depth_pixel[0], left_top_depth_pixel[1], right_down_depth_pixel[0],
                      right_down_depth_pixel[1]]
        print(depth_bbox)
        self.show_depth_bounding(depth_bbox)

        # return [left_top_depth_pixel[0],left_top_depth_pixel[1],right_down_depth_pixel[0],right_down_depth_pixel[1]]

        return bounding_cube
        # print(left_top_color_point)
        # print(color_point)
        # print(color_pixel)
        # exit(0)
