import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from datetime import datetime
import png
from ultralytics import YOLO

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
            if conf > 0.6:
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
    return image

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
        end = time.time()
        fps = 1 / (end - start)
        text = "FPS : " + str(int(fps))
        cv.putText(color_frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 200), 1)
        cv.imshow('color_frame', color_frame)
    cv.destroyAllWindows()
    pipeline.stop()
   