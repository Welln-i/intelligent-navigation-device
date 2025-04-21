import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from datetime import datetime
import png
from ultralytics import YOLO
import serial
import pyttsx3
from playsound import playsound
# abroadst the voice, 1.5s per time and serial have problem

# Load the COCO class names
with open('object_detection_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# Get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the DNN model
model = YOLO('yolov8n.pt')

# Distance thresholds
close_threshold = 1.0  # units: meter
far_threshold = 2.5  # units: meter
max_distance = 3.0  # units: meter

def Target_Detection(image, depth_frame, intrinsics):
    image_height, image_width, _ = image.shape
    # Perform inference
    results = model(image)
    for result in results:
        boxes = result.boxes # Detected bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
            conf = box.conf[0] # Confidence score
            if conf > 0.8:
                label = model.names[int(box.cls[0])] # Class label
                center_x = int((x1 + x2)/2)
                center_y = int((y1 + y2)/2)
                depth_value = depth_frame.get_distance(center_x, center_y)
                depth_point = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth_value)
                horizontal_distance = depth_point[0]
                # Judge the kind of range
                if depth_value < close_threshold:
                    obstacle_type = "Close"
                elif depth_value < far_threshold:
                    obstacle_type = "Far"
                else:
                    obstacle_type = "No"
                # Draw bounding box
                cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Put label and confidence score
                cv.putText(image, f'{conf:.2f} {label} {depth_value} {horizontal_distance:.2f}',
                        (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print(label)
                if abs(horizontal_distance) < 0.35 and depth_value < 2.0:
                    if label == 'person':
                        playsound('./audio_bags/person.wav')
                    elif label == 'chair':
                        playsound('./audio_bags/chair.wav')
                    elif label == 'bottle':
                        playsound('./audio_bags/bottle.wav')
                    elif label == 'cup':
                        playsound('./audio_bags/cup.wav')
                    elif label == 'mouse':
                        playsound('./audio_bags/mouse.wav')
    return image

def depth_to_point_cloud(depth_image, intrinsics, depth_scale, max_distance):
    h, w = depth_image.shape
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    depth_3d = np.zeros((h, w, 3))
    depth_3d[:, :, 2] = depth_image * depth_scale
    depth_3d[:, :, 0] = (np.arange(w) - cx) * depth_3d[:, :, 2] / fx
    depth_3d[:, :, 1] = (np.arange(h)[:, None] - cy) * depth_3d[:, :, 2] / fy
    # Apply mask to filter points within max_distance in the z direction
    mask = depth_3d[:, :, 2] < max_distance
    depth_3d = depth_3d[mask]
    return depth_3d

def filter_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.02)
    cl, ind = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    outlier_cloud = pcd_downsampled.select_by_index(ind)
    return np.asarray(outlier_cloud.points)

def detect_and_remove_ground_plane(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)

    normal_vector = plane_model[:3]
    gravity_vector = np.array([0, 1, 0]) 
    angle = np.arccos(np.dot(normal_vector, gravity_vector) / (np.linalg.norm(normal_vector) * np.linalg.norm(gravity_vector)))
    angle_degrees = np.degrees(angle)

    if angle_degrees > 15.0:
        # print(f"Detected plane's normal vector is not aligned with gravity direction. Angle: {angle_degrees:.2f} degrees")
        return pcd
    else:
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return outlier_cloud

def euclidean_clustering(point_cloud, eps=0.10, min_points=60):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    # if labels.size == 0:
    #     return []
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        clusters.append(pcd.select_by_index(indices))
    return clusters

def find_chosen_clusters(clusters, distance_threshold=2.5, x_range=1.5, min_points=100):
    valid_clusters = []
    cluster_count = 0
    for cluster in clusters:
        points = np.asarray(cluster.points)
        if len(points) >= min_points:
            # Check distance threshold
            distances = np.linalg.norm(points, axis=1)
            if np.all(distances < distance_threshold):
                # # Check x range
                # x_values = points[:, 0]
                # if np.all(np.abs(x_values) < x_range):
                valid_clusters.append(points)
                cluster_count += 1
    return valid_clusters, cluster_count

def get_information_obstacle(valid_clusters, k):
    center_points = []
    center_position_count = 0
    flag_midian = 0
    flag_left = 0
    flag_right = 0
    width = 0
    if k == 0:
        print("it's ok")
    else:
        for cluster in valid_clusters:
            sorted_cluster = cluster[np.argsort(cluster[:, 0])]
            closest_point = sorted_cluster[0]
            farthest_point = sorted_cluster[-1]
            if (closest_point[2] <= 2.0 or farthest_point[2] <= 2.0):#0.36right#-0.32left
                center_position_count += 1
                # if(width < abs(closest_point[0] - farthest_point[0])):
                width = abs(closest_point[0] - farthest_point[0])
                center_position = (closest_point[0] + farthest_point[0])/2
                # most dangerous point
                if(closest_point[0]>-0.2):#-0.2 is location of user as for camera
                    judge_position = closest_point[0]
                elif(farthest_point[0]<-0.2):
                    judge_position = farthest_point[0]
                else:
                    judge_position = center_position
                # dirction    
                if(judge_position > 0.1 and judge_position < 0.4):
                    flag_right = 1
                elif(judge_position < -0.38 and judge_position > -0.68):
                    flag_left = 1
                elif(judge_position >= -0.38 and judge_position <= 0.1):
                    flag_midian = 1
                else:
                    print('')
                center_points.append(closest_point)
                print("total",k,"NO.",center_position_count,":",closest_point,farthest_point,judge_position)
    pin = flag_left*1 + flag_midian*2 +flag_right*4
    return pin, width, center_points

if __name__ == '__main__':
    print("hello, world")
    ser = serial.Serial('COM4', 9600, timeout=1)
    if ser.is_open:
        print("串口已打开")
    else:
        print("串口未打开")
    engine = pyttsx3.init()
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    flag_left = 0
    flag_right = 0
    flag_midian = 0
    try:
        while True:
            start = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            print("initial",time.time() - start)
            color_frame = Target_Detection(color_image, aligned_depth_frame, intrinsics)
            cv.putText(color_frame, "Detection", (240, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            print("imagetime",time.time() - start)
            point_cloud = depth_to_point_cloud(depth_image, intrinsics, depth_scale, 3.0)
            point_cloud=point_cloud.reshape(-1,3)
            nonplane_filtered = filter_point_cloud(np.asarray(point_cloud))
            nonplane = detect_and_remove_ground_plane(nonplane_filtered)
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(nonplane_filtered)
            pcd_nonplane = o3d.geometry.PointCloud()
            pcd_nonplane.points = o3d.utility.Vector3dVector(np.asarray(nonplane.points))
            print("nonplane",time.time() - start)
            clusters = euclidean_clustering(np.asarray(nonplane.points), eps=0.10, min_points=60)
            print("cluster",time.time() - start)
            valid_clusters, k = find_chosen_clusters(clusters, distance_threshold=2.5, x_range=1.5, min_points=100)
            pin, width, center_points = get_information_obstacle(valid_clusters, k)
            # o3d.io.write_point_cloud("pcd_filtered.ply", pcd_filtered)
            # o3d.io.write_point_cloud("nonplane.ply", pcd_nonplane)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd_nonplane = o3d.geometry.PointCloud()
            pcd_nonplane.points = o3d.utility.Vector3dVector(np.array(nonplane.points))
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            end = time.time()
            print("ending",time.time() - start)
            fps = 1 / (end - start)
            text = "FPS : " + str(int(fps))
            cv.putText(color_frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 200), 1)
            cv.imshow('color_frame', color_frame)
            command = f"{pin}\n"
            if ser:
                ser.write(command.encode())
            cv.imwrite("frame.jpg", color_frame)
    finally:
        cv.destroyAllWindows()
        pipeline.stop()
        vis.destroy_window()
        if ser.is_open:
            ser.close()
            print("串口已关闭")