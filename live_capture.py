#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math

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
    print("Press 'q' to quit, 's' to save current frame")

    # Create sl.Mat objects to hold image and depth data
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    
    # Create runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    frame_count = 0
    
    while True:
        # Grab an image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Convert to OpenCV format for display
            rgb_image = image.get_data()[:, :, :3].copy()  # Drop alpha channel and make contiguous
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
            
            distance_text = "Distance: N/A"
            if not err and math.isfinite(point_cloud_value[2]):
                distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
                distance_text = f"Distance: {distance:.1f}mm"

            # Add text overlay
            cv2.putText(rgb_image, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(rgb_image, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(depth_colored, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw crosshair at center
            cv2.line(rgb_image, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)
            cv2.line(rgb_image, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
            cv2.line(depth_colored, (center_x-20, center_y), (center_x+20, center_y), (255, 255, 255), 2)
            cv2.line(depth_colored, (center_x, center_y-20), (center_x, center_y+20), (255, 255, 255), 2)

            # Display images
            cv2.imshow("ZED RGB", rgb_image)
            cv2.imshow("ZED Depth", depth_colored)
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"rgb_frame_{frame_count}.jpg", rgb_image)
                cv2.imwrite(f"depth_frame_{frame_count}.jpg", depth_colored)
                print(f"Saved frame {frame_count}")

    # Clean up
    cv2.destroyAllWindows()
    zed.close()
    print("Camera closed successfully")

if __name__ == "__main__":
    main()