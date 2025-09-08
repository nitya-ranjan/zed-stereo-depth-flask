#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import threading
import time
from flask import Flask, Response
import urllib.request
import os

class YOLODetector:
    def __init__(self):
        self.net = None
        self.output_layers = None
        self.classes = None
        self.colors = None
        self.initialized = False
        
    def download_yolo_files(self):
        """Download YOLOv4-tiny files if not present"""
        base_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/"
        weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
        
        files_to_download = [
            ("yolov4-tiny.cfg", "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"),
            ("coco.names", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
        ]
        
        print("Checking YOLO files...")
        
        # Download config and names files
        for filename, url in files_to_download:
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"✓ Downloaded {filename}")
                except Exception as e:
                    print(f"✗ Failed to download {filename}: {e}")
                    return False
        
        # Check for weights file
        if not os.path.exists("yolov4-tiny.weights"):
            print("YOLOv4-tiny weights not found.")
            print("Please download from: https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights")
            print("For now, running without object detection...")
            return False
            
        return True
    
    def initialize(self):
        """Initialize YOLO detector"""
        if not self.download_yolo_files():
            return False
            
        try:
            # Load YOLO
            self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate colors for each class
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            self.initialized = True
            print("✓ YOLO detector initialized successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize YOLO: {e}")
            return False
    
    def detect(self, image):
        """Detect objects in image"""
        if not self.initialized:
            return []
        
        height, width, channels = image.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.3:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                label = self.classes[class_id]
                color = self.colors[class_id]
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'class_id': class_id,
                    'label': label,
                    'color': color
                })
        
        return detections

class ZEDOpenCVStreamer:
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
        self.detection_stats = {"objects": 0}
        
        # Initialize YOLO detector
        self.yolo = YOLODetector()
        self.yolo_enabled = False
        
    def initialize_camera(self):
        # Create a Camera object
        self.zed = sl.Camera()

        # Create InitParameters object
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 15
        init_params.depth_minimum_distance = 200
        init_params.depth_maximum_distance = 20000

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {repr(err)}")
            return False

        # Try to initialize YOLO
        print("Initializing YOLO detector...")
        self.yolo_enabled = self.yolo.initialize()

        cam_info = self.zed.get_camera_information()
        print(f"ZED Camera initialized - Serial: {cam_info.serial_number}")
        print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
        print(f"YOLO Detection: {'Enabled' if self.yolo_enabled else 'Disabled'}")
        return True

    def get_depth_at_point(self, point_cloud, depth_map, x, y, radius=5):
        """Get depth measurement with fallback strategies"""
        height, width = depth_map.shape
        
        # Try point cloud first
        err, point_cloud_value = point_cloud.get_value(x, y)
        if not err and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]) and point_cloud_value[2] > 0:
            distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
            return distance, f"Point cloud: {distance:.1f}mm"
        
        # Try depth map
        if 0 <= x < width and 0 <= y < height:
            depth_value = depth_map[y, x]
            if math.isfinite(depth_value) and depth_value > 0:
                return depth_value, f"Depth map: {depth_value:.1f}mm"
        
        # Try area average
        valid_depths = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    depth_val = depth_map[py, px]
                    if math.isfinite(depth_val) and depth_val > 200:
                        valid_depths.append(depth_val)
        
        if valid_depths:
            avg_depth = np.mean(valid_depths)
            return avg_depth, f"Average: {avg_depth:.1f}mm"
        
        return None, "No valid depth"

    def get_depth_at_bbox(self, point_cloud, depth_map, bbox):
        """Get depth measurement at center of bounding box"""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        distance, _ = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
        return distance

    def draw_detections(self, image, detections, point_cloud, depth_map):
        """Draw detected objects with depth information"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            label = detection['label']
            color = detection['color']
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Get depth at object center
            depth_distance = self.get_depth_at_bbox(point_cloud, depth_map, detection['bbox'])
            
            # Create label text
            label_text = f"{label} {confidence*100:.0f}%"
            if depth_distance:
                distance_text = f"{depth_distance:.0f}mm"
            else:
                distance_text = "N/A"
            
            # Draw label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x, y-25), (x + max(label_size[0], 80) + 10, y), color, -1)
            
            # Draw texts
            cv2.putText(image, label_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, distance_text, (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(image, (center_x, center_y), 5, color, -1)
        
        return len(detections)

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

                # Get image dimensions
                height, width = rgb_image.shape[:2]
                center_x = width // 2
                center_y = height // 2

                # Process depth
                depth_normalized = np.nan_to_num(depth_map, nan=0.0)
                valid_depth_mask = (depth_normalized > 200) & (depth_normalized < 20000)
                valid_pixels = np.sum(valid_depth_mask)
                total_pixels = depth_normalized.size
                self.depth_stats = {"valid_pixels": valid_pixels, "total_pixels": total_pixels}
                
                # Create depth visualization
                depth_for_display = depth_normalized.copy()
                depth_for_display[depth_for_display > 10000] = 10000
                depth_colored = cv2.normalize(depth_for_display, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

                # Get center depth
                distance, depth_method = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
                if distance:
                    self.distance_text = f"Center: {distance:.1f}mm"
                else:
                    self.distance_text = "Center: N/A"

                # Run YOLO detection
                object_count = 0
                if self.yolo_enabled:
                    detections = self.yolo.detect(rgb_image)
                    object_count = self.draw_detections(rgb_image, detections, point_cloud, depth_map)
                    
                    # Draw detection boxes on depth map too
                    for detection in detections:
                        x, y, w, h = detection['bbox']
                        cv2.rectangle(depth_colored, (x, y), (x + w, y + h), (255, 255, 255), 2)

                self.detection_stats["objects"] = object_count

                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Add overlays
                status_color = (0, 255, 0) if self.yolo_enabled else (0, 255, 255)
                cv2.putText(rgb_image, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(rgb_image, f"FPS: {self.fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                if self.yolo_enabled:
                    cv2.putText(rgb_image, f"Objects: {object_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                else:
                    cv2.putText(rgb_image, "YOLO: Disabled", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                cv2.putText(rgb_image, f"Valid depth: {100*valid_pixels/total_pixels:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                cv2.putText(depth_colored, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if self.yolo_enabled:
                    cv2.putText(depth_colored, f"Objects: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Draw crosshairs
                cv2.line(rgb_image, (center_x-30, center_y), (center_x+30, center_y), (0, 255, 255), 2)
                cv2.line(rgb_image, (center_x, center_y-30), (center_x, center_y+30), (0, 255, 255), 2)
                cv2.circle(rgb_image, (center_x, center_y), 10, (0, 255, 255), 2)
                
                cv2.line(depth_colored, (center_x-30, center_y), (center_x+30, center_y), (255, 255, 255), 2)
                cv2.line(depth_colored, (center_x, center_y-30), (center_x, center_y+30), (255, 255, 255), 2)
                cv2.circle(depth_colored, (center_x, center_y), 10, (255, 255, 255), 2)

                # Store frames
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
streamer = ZEDOpenCVStreamer()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ZED + OpenCV YOLO Detection</title>
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
        .features {
            background-color: #2a4a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .features h4 {
            margin-top: 0;
            color: #90EE90;
        }
        .warning {
            background-color: #4a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .warning h4 {
            margin-top: 0;
            color: #ff6b6b;
        }
        @media (max-width: 1300px) {
            .stream-container { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="info">
            <h1>🎯 ZED Camera + OpenCV YOLO</h1>
            <p>Real-time object detection using OpenCV DNN with depth information</p>
        </div>
        
        <div class="stream-container">
            <div class="stream-box">
                <h3>📹 RGB + YOLO Detection</h3>
                <img src="/video_rgb" alt="RGB with YOLO">
            </div>
            
            <div class="stream-box">
                <h3>📏 Depth Map + Objects</h3>
                <img src="/video_depth" alt="Depth with Objects">
            </div>
        </div>
        
        <div class="warning">
            <h4>⚠️ YOLO Model Setup Required:</h4>
            <p>If you see "YOLO: Disabled", download YOLOv4-tiny weights:</p>
            <code>wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights</code>
        </div>
        
        <div class="features">
            <h4>🔍 Features:</h4>
            <ul>
                <li><strong>OpenCV YOLO:</strong> YOLOv4-tiny for fast detection</li>
                <li><strong>80 Object Classes:</strong> People, cars, animals, etc.</li>
                <li><strong>3D Distance:</strong> Shows distance to each detected object</li>
                <li><strong>Real-time:</strong> ~15 FPS detection with depth mapping</li>
                <li><strong>Compatible:</strong> Works with all ZED camera models</li>
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
    print("Starting ZED + OpenCV YOLO Detection Server...")
    if streamer.start_streaming():
        print("System ready!")
        print("Access the stream at: http://192.168.1.196:5000")
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            streamer.stop_streaming()
    else:
        print("Failed to initialize camera")