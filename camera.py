import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from datetime import datetime
import png

# Load the COCO class names
with open('object_detection_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# Get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the DNN model
model = cv.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco.txt', framework='TensorFlow')

# Distance thresholds
close_threshold = 1.0  # units: meter
far_threshold = 2.5  # units: meter
max_distance = 3.0  # units: meter

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

    if angle_degrees > 20.0:
        # print(f"Detected plane's normal vector is not aligned with gravity direction. Angle: {angle_degrees:.2f} degrees")
        return pcd
    else:
        a = 1
        # print(f"Detected ground plane with normal vector within {angle_degrees:.2f} degrees of gravity direction")
    # print(f"Plane equation: {plane_model}")
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return outlier_cloud

def euclidean_clustering(point_cloud, eps=0.10, min_points=60):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        clusters.append(pcd.select_by_index(indices))
    return clusters

def find_chosen_clusters(clusters, distance_threshold=3.0, x_range=1.5, min_points=100):
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


if __name__ == '__main__':
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
        action = cv.waitKey(10)
        color_frame = Target_Detection(color_image, aligned_depth_frame, intrinsics)
        cv.putText(color_frame, "Detection", (240, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        if action == ord('q') or action == ord('Q'):
            break
        print(time.time() - start)
        point_cloud = depth_to_point_cloud(depth_image, intrinsics, depth_scale, max_distance)
        print("depth-point",time.time() - start)
        point_cloud=point_cloud.reshape(-1,3)
        nonplane_filtered = filter_point_cloud(np.asarray(point_cloud))
        print("filter",time.time() - start)
        nonplane = detect_and_remove_ground_plane(nonplane_filtered)
        print("remove_plane",time.time() - start)
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(nonplane_filtered)
        print(time.time() - start)
        clusters = euclidean_clustering(np.asarray(nonplane.points), eps=0.10, min_points=60)
        print("euclidean_clustering",time.time() - start)
        valid_clusters, k = find_chosen_clusters(clusters, distance_threshold=3.0, x_range=1.5, min_points=100)
        print("find cluster",time.time() - start)
        center_points = []
        center_position_count = 0
        if k > 0:
            for cluster in valid_clusters:
                center_position_count += 1
                center_point = cluster[len(cluster) // 2]
                center_points.append(center_point)
                print("total",k,"NO.",center_position_count,":",center_point)
                center_pcd = o3d.geometry.PointCloud()
                center_pcd.points = o3d.utility.Vector3dVector(np.array(center_points))
                center_pcd.paint_uniform_color([1, 0, 0])  # Red color for center points
                o3d.io.write_point_cloud("center_pcd.ply", center_pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_nonplane = o3d.geometry.PointCloud()
        pcd_nonplane.points = o3d.utility.Vector3dVector(np.array(nonplane.points))
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        end = time.time()
        print(time.time() - start)
        fps = 1 / (end - start)
        text = "FPS : " + str(int(fps))
        cv.putText(color_frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 200), 1)
        cv.imshow('color_frame', color_frame)
        # cv.imwrite("frame.jpg", color_frame)
    cv.destroyAllWindows()
    pipeline.stop()
    vis.destroy_window()