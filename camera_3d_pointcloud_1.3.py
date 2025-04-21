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

######################### Detection ##########################
# load the COCO class names
with open('object_detection_coco.txt', 'r') as f: 
    class_names = f.read().split('\n')
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
# load the DNN model
model = cv.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco.txt', framework='TensorFlow')

######################### openpose ##########################
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

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

def openpose(frame):
    frameHeight, frameWidth = frame.shape[:2]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert (len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > 0.2 else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    return frame

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

def visualize_point_cloud(point_cloud):
    points = point_cloud.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    capture = cv.VideoCapture(2)
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
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
        action = cv.waitKey(10) & 0xFF
        if state:
            color_frame = Target_Detection(color_image, aligned_depth_frame, intrinsics)
            cv.putText(color_frame, "Detection", (240, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        else:
            color_frame = openpose(color_image)
            cv.putText(color_frame, "Openpose", (240, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        if action == ord('q') or action == ord('Q'):
            break
        if action == ord('f') or action == ord('F'):
            state = not state

        # convert depth maps to 3D point clouds
        point_cloud = depth_to_point_cloud(depth_image, intrinsics, depth_scale)
        # Save point cloud if 's' key is pressed
        if action == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"point_cloud_{timestamp}.npy"
            np.save(filename, point_cloud)
            print(f"Point cloud saved as '{filename}' in {os.getcwd()}")
            visualize_point_cloud(point_cloud)
            
        end = time.time()
        fps = 1 / (end - start)
        text = "FPS : " + str(int(fps))
        cv.putText(color_frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 200), 1)
        cv.imshow('frame', color_frame)

    capture.release()
    cv.destroyAllWindows()
