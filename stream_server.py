#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import threading
import time
from flask import Flask, Response, render_template_string
import base64
from io import BytesIO

class ZEDStreamer:
    def __init__(self):
        self.zed = None
        self.streaming = False
        self.current_rgb = None
        self.current_depth = None
        self.distance_text = "Distance: N/A"
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def initialize_camera(self):
        # Create a Camera object
        self.zed = sl.Camera()

        # Create InitParameters object
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 15  # Lower FPS for streaming

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {repr(err)}")
            return False

        cam_info = self.zed.get_camera_information()
        print(f"ZED Camera initialized - Serial: {cam_info.serial_number}")
        print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
        return True

    def capture_loop(self):
        if not self.zed:
            return
            
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

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

                # Process depth
                depth_normalized = np.nan_to_num(depth_map, nan=0.0)
                depth_colored = cv2.normalize(depth_normalized, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

                # Get center distance
                center_x = image.get_width() // 2
                center_y = image.get_height() // 2
                err, point_cloud_value = point_cloud.get_value(center_x, center_y)
                
                if not err and math.isfinite(point_cloud_value[2]):
                    distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
                    self.distance_text = f"Distance: {distance:.1f}mm"
                else:
                    self.distance_text = "Distance: N/A"

                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Add overlays
                cv2.putText(rgb_image, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(rgb_image, f"FPS: {self.fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(depth_colored, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Draw crosshairs
                cv2.line(rgb_image, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)
                cv2.line(rgb_image, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
                cv2.line(depth_colored, (center_x-20, center_y), (center_x+20, center_y), (255, 255, 255), 2)
                cv2.line(depth_colored, (center_x, center_y-20), (center_x, center_y+20), (255, 255, 255), 2)

                # Store current frames
                self.current_rgb = rgb_image
                self.current_depth = depth_colored
                self.frame_count += 1

            time.sleep(0.033)  # ~30 FPS max

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
    <title>ZED Camera Stream</title>
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
        @media (max-width: 1300px) {
            .stream-container { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="info">
            <h1>ZED Stereo Camera Live Stream</h1>
            <p>RGB Feed shows the color image with distance measurement at center</p>
            <p>Depth Feed shows depth information as a colored heatmap</p>
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
        
        <script>
            // Auto-refresh images every 100ms
            setInterval(function() {
                document.querySelector('img[src="/video_rgb"]').src = '/video_rgb?' + new Date().getTime();
                document.querySelector('img[src="/video_depth"]').src = '/video_depth?' + new Date().getTime();
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
        time.sleep(0.033)

def generate_depth():
    while True:
        frame = streamer.get_frame_depth()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video_rgb')
def video_rgb():
    return Response(generate_rgb(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_depth')
def video_depth():
    return Response(generate_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting ZED Camera Stream Server...")
    if streamer.start_streaming():
        print("Camera initialized successfully!")
        print("Starting web server on port 5000...")
        print("Access the stream at: http://[device-ip]:5000")
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            streamer.stop_streaming()
    else:
        print("Failed to initialize camera")