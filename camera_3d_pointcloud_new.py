#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : openpose_for_image_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-29 21:50:17
"""
import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import os
import open3d as o3d
from struct import pack
import png

######################### Detection ##########################
# load the COCO class names
with open('object_detection_coco.txt', 'r') as f: 
    class_names = f.read().split('\n')
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
# load the DNN model
model = cv.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco.txt', framework='TensorFlow')

# distance threshold
close_threshold = 1.0  # units: meter
far_threshold = 2.5  # units: meter

def Target_Detection(image, depth_frame, intrinsics):
    image_height, image_width, _ = image.shape
    # create blob from image
    blob = cv.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
    model.setInput(blob)
    output = model.forward()
    # loop over each of the detections
    for detection in output[0, 0, :, :]:
        # extract the confidence of the detection
        confidence = detection[2]
        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        if confidence > .4:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = class_names[int(class_id) - 1]
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            # calculate the distance from the center point
            center_x = int((box_x + box_width) / 2)
            center_y = int((box_y + box_height) / 2)
            depth_value = depth_frame.get_distance(center_x, center_y)
            # calculate horizontal distance (left/right)
            depth_point = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth_value)
            horizontal_distance = depth_point[0]
            # judge the kind of range
            if depth_value < close_threshold:
                obstacle_type = "Close"
            elif depth_value < far_threshold:
                obstacle_type = "Far"
            else:
                obstacle_type = "No"
            # draw a rectangle around each detected object
            cv.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
            # put the class name text on the detected object
            cv.putText(image, f"{class_name} {obstacle_type} {depth_value:.2f}m, {horizontal_distance:.2f}m", (int(box_x), int(box_y - 5)), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

def depth_to_point_cloud(depth_image, intrinsics, depth_scale):
    h, w = depth_image.shape
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    depth_3d = np.zeros((h, w, 3))
    depth_3d[:, :, 2] = depth_image * depth_scale
    depth_3d[:, :, 0] = (np.arange(w) - cx) * depth_3d[:, :, 2] / fx
    depth_3d[:, :, 1] = (np.arange(h)[:, None] - cy) * depth_3d[:, :, 2] / fy
    return depth_3d


def detect_and_remove_ground_plane(point_cloud):
    # con90vert pointcloud to Open3D pointcloud object
    point_cloud=point_cloud.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # use RANSAC detect ground
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.03,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # extract ground inlier_cloud and delete
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    return outlier_cloud

def filter_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.005)
    cl, ind = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    outlier_cloud = pcd_downsampled.select_by_index(ind)
    
    return np.asarray(outlier_cloud.points)

def euclidean_clustering(point_cloud, eps=0.02, min_points=10):
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        clusters.append(point_cloud.select_by_index(indices))
    return clusters

def draw_bounding_boxes(image, clusters, intrinsics):
    for cluster in clusters:
        points = np.asarray(cluster.points)
        if len(points) == 0:
            continue
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        
        bbox_corners = [
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
        ]
        bbox_corners = np.array(bbox_corners)
        
        bbox_corners_2d = []
        for point in bbox_corners:
            px, py = rs.rs2_project_point_to_pixel(intrinsics, point[:3])
            bbox_corners_2d.append((int(px), int(py)))
        
        # Draw bounding box
        for i in range(4):
            cv.line(image, bbox_corners_2d[i], bbox_corners_2d[(i+1)%4], (0, 255, 0), 2)
            cv.line(image, bbox_corners_2d[i+4], bbox_corners_2d[(i+1)%4 + 4], (0, 255, 0), 2)
            cv.line(image, bbox_corners_2d[i], bbox_corners_2d[i+4], (0, 255, 0), 2)
    
    return image

if __name__ == '__main__':
    capture = cv.VideoCapture(2)
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # get camera parameters
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    
    # build open3D visualization 
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    state = True
    while True:
        start = time.time()
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        action = cv.waitKey(10)
        color_frame = Target_Detection(color_image, aligned_depth_frame, intrinsics)

        cv.putText(color_frame, "Detection", (240, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        if action == ord('q') or action == ord('Q'):
            break
        # convert depth maps to 3D point clouds
        point_cloud = depth_to_point_cloud(depth_image, intrinsics, depth_scale)
        point_cloud=point_cloud.reshape(-1,3)
        # remove the ground 
        nonplane = detect_and_remove_ground_plane(point_cloud)
        nonplane_filtered = filter_point_cloud(np.asarray(nonplane.points))
        # pass the outlier
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(nonplane_filtered)

        # Update Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_nonplane = o3d.geometry.PointCloud()
        pcd_nonplane.points = o3d.utility.Vector3dVector(np.array(nonplane.points))

        # save the ply 3D file
        o3d.io.write_point_cloud("output.ply", pcd)
        o3d.io.write_point_cloud("nonplane.ply", pcd_nonplane)
        o3d.io.write_point_cloud("nonplane_filtered.ply", pcd_filtered)
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        depth_image_normalized = cv.normalize(depth_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
        depth_image_colored = cv.applyColorMap(depth_image_normalized, cv.COLORMAP_JET)
        cv.imshow('depth_frame', depth_image_colored)
        end = time.time()
        fps = 1 / (end - start)
        text = "FPS : " + str(int(fps))
        cv.putText(color_frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 200), 1)
        cv.imshow('color_frame', color_frame)
        cv.imwrite("frame.jpg",color_frame)
        with open("depth.jpg","wb") as f:
            write = png.Writer(width = depth_image.shape[1],height= depth_image.shape[0],bitdepth=16, greyscale=True)
            gray2list = depth_image.tolist()
            write.write(f,gray2list)

    capture.release()
    cv.destroyAllWindows()