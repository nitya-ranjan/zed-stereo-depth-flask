#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import os
from datetime import datetime

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use NEURAL depth mode for best quality
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeters for depth measurements
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 resolution
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Error: {repr(err)}. Exit program.")
        exit(1)

    # Display camera info
    cam_info = zed.get_camera_information()
    print(f"ZED Camera Serial: {cam_info.serial_number}")
    print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
    print(f"FPS: {cam_info.camera_configuration.fps}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"zed_capture_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving data to: {output_dir}")
    print("Capturing 10 frames...")

    # Create sl.Mat objects to hold image and depth data
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    
    # Create runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    frame_count = 0
    distances = []
    
    while frame_count < 10:
        # Grab an image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Convert to OpenCV format
            rgb_image = image.get_data()[:, :, :3].copy()
            rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
            depth_map = depth.get_data()
            
            # Normalize depth for visualization
            depth_normalized = np.nan_to_num(depth_map, nan=0.0)
            depth_colored = cv2.normalize(depth_normalized, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

            # Get distance at center of image
            center_x = image.get_width() // 2
            center_y = image.get_height() // 2
            err, point_cloud_value = point_cloud.get_value(center_x, center_y)
            
            distance_mm = None
            if not err and math.isfinite(point_cloud_value[2]):
                distance_mm = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
                distances.append(distance_mm)
                print(f"Frame {frame_count}: Distance at center = {distance_mm:.1f}mm")
            else:
                print(f"Frame {frame_count}: Distance measurement not available")

            # Save images
            cv2.imwrite(os.path.join(output_dir, f"rgb_{frame_count:03d}.jpg"), rgb_image)
            cv2.imwrite(os.path.join(output_dir, f"depth_{frame_count:03d}.jpg"), depth_colored)
            
            # Save raw depth data as numpy array
            np.save(os.path.join(output_dir, f"depth_raw_{frame_count:03d}.npy"), depth_map)
            
            frame_count += 1

    # Save summary data
    summary = {
        'camera_serial': cam_info.serial_number,
        'resolution': f"{cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}",
        'fps': cam_info.camera_configuration.fps,
        'frames_captured': frame_count,
        'distances_mm': distances,
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_dir, 'capture_summary.txt'), 'w') as f:
        f.write("ZED Camera Capture Summary\n")
        f.write("=" * 30 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        
        if distances:
            f.write(f"\nDistance Statistics:\n")
            f.write(f"Min: {min(distances):.1f}mm\n")
            f.write(f"Max: {max(distances):.1f}mm\n")
            f.write(f"Avg: {np.mean(distances):.1f}mm\n")

    print(f"\nCapture complete! Data saved in: {output_dir}")
    print(f"Files created:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size} bytes)")

    # Clean up
    zed.close()
    print("Camera closed successfully")

if __name__ == "__main__":
    main()