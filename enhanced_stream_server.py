#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import threading
import time
from flask import Flask, Response
import sys

class ZEDStreamer:
    def __init__(self):
        self.zed = None
        self.streaming = False
        self.current_rgb = None
        self.current_depth = None
        self.distance_text = "Distance: N/A"
        self.debug_info = ""
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.depth_stats = {"valid_pixels": 0, "total_pixels": 0}
        
    def initialize_camera(self):
        # Create a Camera object
        self.zed = sl.Camera()

        # Create InitParameters object with better settings for depth
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Best quality depth mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 15
        
        # Important depth settings
        init_params.depth_minimum_distance = 200  # 20cm minimum
        init_params.depth_maximum_distance = 20000  # 20m maximum
        init_params.enable_right_side_measure = False
        
        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {repr(err)}")
            return False

        cam_info = self.zed.get_camera_information()
        print(f"ZED Camera initialized - Serial: {cam_info.serial_number}")
        print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
        print(f"Depth range: 200mm to 20000mm")
        return True

    def get_depth_at_point(self, point_cloud, depth_map, x, y, radius=5):
        """Get depth measurement with fallback strategies"""
        height, width = depth_map.shape
        
        # Strategy 1: Try point cloud at exact center
        err, point_cloud_value = point_cloud.get_value(x, y)
        if not err and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]) and point_cloud_value[2] > 0:
            distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
            return distance, f"Point cloud center: {distance:.1f}mm"
        
        # Strategy 2: Try depth map at exact center
        if 0 <= x < width and 0 <= y < height:
            depth_value = depth_map[y, x]
            if math.isfinite(depth_value) and depth_value > 0:
                return depth_value, f"Depth map center: {depth_value:.1f}mm"
        
        # Strategy 3: Average around the center point
        valid_depths = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    depth_val = depth_map[py, px]
                    if math.isfinite(depth_val) and depth_val > 200:  # At least 20cm
                        valid_depths.append(depth_val)
        
        if valid_depths:
            avg_depth = np.mean(valid_depths)
            return avg_depth, f"Average ({len(valid_depths)} pixels): {avg_depth:.1f}mm"
        
        # Strategy 4: Try point cloud around the center
        for dy in range(-radius, radius + 1, 2):
            for dx in range(-radius, radius + 1, 2):
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    err, pc_val = point_cloud.get_value(px, py)
                    if not err and len(pc_val) >= 3 and math.isfinite(pc_val[2]) and pc_val[2] > 0:
                        distance = math.sqrt(sum(coord * coord for coord in pc_val[:3]))
                        return distance, f"Point cloud nearby ({dx},{dy}): {distance:.1f}mm"
        
        return None, "No valid depth found"

    def capture_loop(self):
        if not self.zed:
            return
            
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        
        # Set runtime parameters for better depth
        runtime_parameters.confidence_threshold = 50  # Lower threshold
        runtime_parameters.texture_confidence_threshold = 100

        while self.streaming:
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve images
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                # Convert to OpenCV format
                rgb_image = image.get_data()[:, :, :3].copy()
                rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
                depth_map = depth.get_data()

                # Get image dimensions
                height, width = rgb_image.shape[:2]
                center_x = width // 2
                center_y = height // 2

                # Enhanced depth processing
                depth_normalized = np.nan_to_num(depth_map, nan=0.0)
                
                # Calculate depth statistics
                valid_depth_mask = (depth_normalized > 200) & (depth_normalized < 20000)
                valid_pixels = np.sum(valid_depth_mask)
                total_pixels = depth_normalized.size
                self.depth_stats = {"valid_pixels": valid_pixels, "total_pixels": total_pixels}
                
                # Create better depth visualization
                depth_for_display = depth_normalized.copy()
                depth_for_display[depth_for_display > 10000] = 10000  # Cap at 10m for better visualization
                depth_colored = cv2.normalize(depth_for_display, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

                # Get enhanced depth measurement
                distance, depth_method = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
                
                if distance:
                    self.distance_text = f"Distance: {distance:.1f}mm"
                    self.debug_info = depth_method
                else:
                    self.distance_text = "Distance: N/A"
                    self.debug_info = depth_method

                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Add comprehensive overlays
                cv2.putText(rgb_image, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(rgb_image, f"FPS: {self.fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(rgb_image, f"Valid depth: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(rgb_image, self.debug_info, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.putText(depth_colored, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(depth_colored, f"Valid: {100*valid_pixels/total_pixels:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw enhanced crosshairs
                cv2.line(rgb_image, (center_x-30, center_y), (center_x+30, center_y), (0, 255, 0), 2)
                cv2.line(rgb_image, (center_x, center_y-30), (center_x, center_y+30), (0, 255, 0), 2)
                cv2.circle(rgb_image, (center_x, center_y), 10, (0, 255, 0), 2)
                
                cv2.line(depth_colored, (center_x-30, center_y), (center_x+30, center_y), (255, 255, 255), 2)
                cv2.line(depth_colored, (center_x, center_y-30), (center_x, center_y+30), (255, 255, 255), 2)
                cv2.circle(depth_colored, (center_x, center_y), 10, (255, 255, 255), 2)

                # Store current frames
                self.current_rgb = rgb_image
                self.current_depth = depth_colored
                self.frame_count += 1

            time.sleep(0.066)  # ~15 FPS

    def start_streaming(self):
        if self.initialize_camera():
            self.streaming = True
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            return True
        return False

    def stop_streaming(self):
        self.streaming = False
        if self.zed:
            self.zed.close()

    def get_frame_rgb(self):
        if self.current_rgb is not None:
            _, buffer = cv2.imencode('.jpg', self.current_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
        return None

    def get_frame_depth(self):
        if self.current_depth is not None:
            _, buffer = cv2.imencode('.jpg', self.current_depth, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
        return None

# Flask app
app = Flask(__name__)
streamer = ZEDStreamer()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ZED Camera Stream - Enhanced</title>
    <style>
        body { 
            margin: 0; 
            padding: 20px; 
            background-color: #1a1a1a; 
            color: white; 
            font-family: Arial, sans-serif;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto;
        }
        .stream-container {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .stream-box {
            border: 2px solid #333;
            border-radius: 8px;
            padding: 10px;
            background-color: #2a2a2a;
        }
        .stream-box h3 {
            margin: 0 0 10px 0;
            text-align: center;
        }
        img { 
            max-width: 640px; 
            height: auto; 
            border-radius: 4px;
        }
        .info {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #333;
            border-radius: 8px;
        }
        .tips {
            background-color: #2a4a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .tips h4 {
            margin-top: 0;
            color: #90EE90;
        }
        @media (max-width: 1300px) {
            .stream-container { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="info">
            <h1>ZED Stereo Camera - Enhanced Stream</h1>
            <p>RGB Feed shows color image with enhanced depth measurement</p>
            <p>Depth Feed shows depth heatmap (blue=close, red=far)</p>
        </div>
        
        <div class="stream-container">
            <div class="stream-box">
                <h3>RGB Camera Feed</h3>
                <img src="/video_rgb" alt="RGB Stream">
            </div>
            
            <div class="stream-box">
                <h3>Depth Map</h3>
                <img src="/video_depth" alt="Depth Stream">
            </div>
        </div>
        
        <div class="tips">
            <h4>Tips for Better Depth Sensing:</h4>
            <ul>
                <li><strong>Distance:</strong> Point camera at objects 0.2m - 20m away</li>
                <li><strong>Lighting:</strong> Ensure good lighting on the scene</li>
                <li><strong>Texture:</strong> Plain walls may not provide depth - try textured surfaces</li>
                <li><strong>Movement:</strong> Keep camera relatively still for best results</li>
                <li><strong>Calibration:</strong> Wave the camera gently if you see "N/A" consistently</li>
            </ul>
        </div>
        
        <script>
            setInterval(function() {
                var timestamp = new Date().getTime();
                document.querySelector('img[src^="/video_rgb"]').src = '/video_rgb?' + timestamp;
                document.querySelector('img[src^="/video_depth"]').src = '/video_depth?' + timestamp;
            }, 100);
        </script>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML_TEMPLATE

def generate_rgb():
    while True:
        frame = streamer.get_frame_rgb()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.066)

def generate_depth():
    while True:
        frame = streamer.get_frame_depth()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.066)

@app.route('/video_rgb')
def video_rgb():
    return Response(generate_rgb(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_depth')
def video_depth():
    return Response(generate_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Enhanced ZED Camera Stream Server...")
    if streamer.start_streaming():
        print("Camera initialized successfully!")
        print("Enhanced depth measurement active!")
        print("Starting web server on port 5000...")
        print("Access the stream at: http://192.168.1.196:5000")
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            streamer.stop_streaming()
    else:
        print("Failed to initialize camera")