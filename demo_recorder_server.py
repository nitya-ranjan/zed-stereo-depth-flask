#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import threading
import time
from flask import Flask, Response, jsonify, request
import urllib.request
import os
import json
import csv
from datetime import datetime

class YOLODetector:
    def __init__(self):
        self.net = None
        self.output_layers = None
        self.classes = None
        self.colors = None
        self.initialized = False
        
    def initialize(self):
        """Initialize YOLO detector"""
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
                    'color': color,
                    'center': [x + w//2, y + h//2]
                })
        
        return detections

class ZEDDemoRecorder:
    def __init__(self):
        self.zed = None
        self.streaming = False
        self.recording = False
        self.current_rgb = None
        self.current_depth = None
        self.distance_text = "Distance: N/A"
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.depth_stats = {"valid_pixels": 0, "total_pixels": 0}
        self.detection_stats = {"objects": 0}
        
        # Recording variables
        self.video_writer_rgb = None
        self.video_writer_depth = None
        self.video_writer_combined = None
        self.recording_start_time = None
        self.recording_session_id = None
        
        # Logging variables
        self.detection_log = []
        self.session_stats = {
            'total_detections': 0,
            'vehicle_detections': 0,
            'person_detections': 0,
            'detection_history': []
        }
        
        # Initialize YOLO detector
        self.yolo = YOLODetector()
        self.yolo_enabled = self.yolo.initialize()
        
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

        cam_info = self.zed.get_camera_information()
        print(f"ZED Camera initialized - Serial: {cam_info.serial_number}")
        print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
        print(f"YOLO Detection: {'Enabled' if self.yolo_enabled else 'Disabled'}")
        return True

    def start_recording(self, session_name="demo"):
        """Start video recording and logging"""
        if self.recording:
            return False
            
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_session_id = f"{session_name}_{timestamp}"
        session_dir = f"recordings/{self.recording_session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Get camera info for video dimensions
        cam_info = self.zed.get_camera_information()
        w = cam_info.camera_configuration.resolution.width
        h = cam_info.camera_configuration.resolution.height
        
        self.video_writer_rgb = cv2.VideoWriter(
            f"{session_dir}/rgb_detections.mp4", fourcc, 15.0, (w, h))
        self.video_writer_depth = cv2.VideoWriter(
            f"{session_dir}/depth_map.mp4", fourcc, 15.0, (w, h))
        self.video_writer_combined = cv2.VideoWriter(
            f"{session_dir}/combined_view.mp4", fourcc, 15.0, (w*2, h))
        
        # Reset logging variables
        self.detection_log = []
        self.session_stats = {
            'session_id': self.recording_session_id,
            'start_time': datetime.now().isoformat(),
            'total_detections': 0,
            'vehicle_detections': 0,
            'person_detections': 0,
            'detection_history': [],
            'session_dir': session_dir
        }
        
        self.recording = True
        self.recording_start_time = time.time()
        
        print(f"🔴 Recording started: {self.recording_session_id}")
        return True
    
    def stop_recording(self):
        """Stop recording and save logs"""
        if not self.recording:
            return False
            
        self.recording = False
        
        # Close video writers
        if self.video_writer_rgb:
            self.video_writer_rgb.release()
        if self.video_writer_depth:
            self.video_writer_depth.release()
        if self.video_writer_combined:
            self.video_writer_combined.release()
        
        # Save detection logs
        session_dir = self.session_stats['session_dir']
        
        # Save detailed detection log as CSV
        if self.detection_log:
            with open(f"{session_dir}/detections.csv", 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'frame_number', 'object_id', 'label', 'confidence', 
                             'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'distance_mm', 'center_x', 'center_y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.detection_log)
        
        # Save session summary as JSON
        self.session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats['duration_seconds'] = time.time() - self.recording_start_time
        self.session_stats['total_frames'] = len(self.detection_log)
        
        with open(f"{session_dir}/session_summary.json", 'w') as f:
            json.dump(self.session_stats, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(session_dir)
        
        print(f"⏹️  Recording stopped: {self.recording_session_id}")
        print(f"📁 Files saved to: {session_dir}")
        return True

    def generate_summary_report(self, session_dir):
        """Generate a human-readable summary report"""
        report_lines = [
            f"ZED Camera Detection Demo Report",
            f"=" * 40,
            f"Session ID: {self.session_stats['session_id']}",
            f"Duration: {self.session_stats['duration_seconds']:.1f} seconds",
            f"Total Detections: {self.session_stats['total_detections']}",
            f"Vehicle Detections: {self.session_stats['vehicle_detections']}",
            f"Person Detections: {self.session_stats['person_detections']}",
            f"",
            f"Files Generated:",
            f"- rgb_detections.mp4 (RGB video with detection overlays)",
            f"- depth_map.mp4 (Depth visualization)",
            f"- combined_view.mp4 (Side-by-side view)",
            f"- detections.csv (Detailed detection data)",
            f"- session_summary.json (Session metadata)",
            f"",
            f"Detection Breakdown:"
        ]
        
        # Count detections by class
        detection_counts = {}
        for detection in self.detection_log:
            label = detection['label']
            detection_counts[label] = detection_counts.get(label, 0) + 1
        
        for label, count in sorted(detection_counts.items()):
            report_lines.append(f"- {label}: {count}")
        
        with open(f"{session_dir}/README.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def get_depth_at_point(self, point_cloud, depth_map, x, y, radius=5):
        """Get depth measurement with fallback strategies"""
        height, width = depth_map.shape
        
        # Try point cloud first
        err, point_cloud_value = point_cloud.get_value(x, y)
        if not err and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]) and point_cloud_value[2] > 0:
            distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
            return distance
        
        # Try depth map
        if 0 <= x < width and 0 <= y < height:
            depth_value = depth_map[y, x]
            if math.isfinite(depth_value) and depth_value > 0:
                return depth_value
        
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
            return np.mean(valid_depths)
        
        return None

    def log_detection(self, detection, frame_number, timestamp, distance):
        """Log detection data"""
        if not self.recording:
            return
            
        x, y, w, h = detection['bbox']
        
        log_entry = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'object_id': len(self.detection_log),
            'label': detection['label'],
            'confidence': detection['confidence'],
            'bbox_x': x,
            'bbox_y': y,
            'bbox_w': w,
            'bbox_h': h,
            'distance_mm': distance if distance else 0,
            'center_x': detection['center'][0],
            'center_y': detection['center'][1]
        }
        
        self.detection_log.append(log_entry)
        
        # Update session stats
        self.session_stats['total_detections'] += 1
        if detection['label'] in ['car', 'truck', 'bus', 'motorbike']:
            self.session_stats['vehicle_detections'] += 1
        elif detection['label'] == 'person':
            self.session_stats['person_detections'] += 1

    def draw_detections(self, image, detections, point_cloud, depth_map, frame_number):
        """Draw detected objects with depth information and logging"""
        current_timestamp = datetime.now().isoformat()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            label = detection['label']
            color = detection['color']
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Get depth at object center
            center_x, center_y = detection['center']
            distance = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
            
            # Log this detection
            self.log_detection(detection, frame_number, current_timestamp, distance)
            
            # Create label text
            label_text = f"{label} {confidence*100:.0f}%"
            if distance:
                distance_text = f"{distance:.0f}mm"
            else:
                distance_text = "N/A"
            
            # Draw label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x, y-25), (x + max(label_size[0], 80) + 10, y), color, -1)
            
            # Draw texts
            cv2.putText(image, label_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, distance_text, (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(image, (center_x, center_y), 5, color, -1)
        
        return len(detections)

    def capture_loop(self):
        if not self.zed:
            return
            
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        
        recorded_frames = 0

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
                distance = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
                if distance:
                    self.distance_text = f"Center: {distance:.1f}mm"
                else:
                    self.distance_text = "Center: N/A"

                # Run YOLO detection
                object_count = 0
                if self.yolo_enabled:
                    detections = self.yolo.detect(rgb_image)
                    object_count = self.draw_detections(rgb_image, detections, point_cloud, depth_map, self.frame_count)
                    
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
                recording_color = (0, 0, 255) if self.recording else status_color
                
                cv2.putText(rgb_image, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, recording_color, 2)
                cv2.putText(rgb_image, f"FPS: {self.fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, recording_color, 2)
                
                if self.yolo_enabled:
                    cv2.putText(rgb_image, f"Objects: {object_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, recording_color, 2)
                    
                cv2.putText(rgb_image, f"Valid depth: {100*valid_pixels/total_pixels:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, recording_color, 2)
                
                # Recording indicator
                if self.recording:
                    cv2.putText(rgb_image, "🔴 RECORDING", (width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(depth_colored, "🔴 REC", (width-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
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

                # Record frames if recording is active
                if self.recording:
                    if self.video_writer_rgb:
                        self.video_writer_rgb.write(rgb_image)
                    if self.video_writer_depth:
                        self.video_writer_depth.write(depth_colored)
                    if self.video_writer_combined:
                        combined = np.hstack((rgb_image, depth_colored))
                        self.video_writer_combined.write(combined)
                    recorded_frames += 1

                # Store frames for streaming
                self.current_rgb = rgb_image
                self.current_depth = depth_colored
                self.frame_count += 1

            time.sleep(0.066)  # ~15 FPS

    def start_streaming(self):
        if self.initialize_camera():
            self.streaming = True
            os.makedirs("recordings", exist_ok=True)
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            return True
        return False

    def stop_streaming(self):
        if self.recording:
            self.stop_recording()
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

    def get_recording_status(self):
        return {
            'recording': self.recording,
            'session_id': self.recording_session_id,
            'duration': time.time() - self.recording_start_time if self.recording else 0,
            'total_detections': self.session_stats['total_detections'],
            'vehicle_detections': self.session_stats['vehicle_detections'],
            'person_detections': self.session_stats['person_detections']
        }

# Flask app
app = Flask(__name__)
recorder = ZEDDemoRecorder()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ZED Demo Recorder</title>
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
        .controls {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background-color: #333;
            border-radius: 8px;
        }
        .controls button {
            padding: 12px 24px;
            margin: 0 10px;
            font-size: 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .record-btn { background-color: #ff4444; color: white; }
        .stop-btn { background-color: #888; color: white; }
        .stop-btn:hover { background-color: #666; }
        .record-btn:hover { background-color: #cc3333; }
        .status {
            padding: 10px;
            margin: 10px 0;
            background-color: #2a2a2a;
            border-radius: 4px;
            font-family: monospace;
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
        @media (max-width: 1300px) {
            .stream-container { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="info">
            <h1>🎬 ZED Demo Recorder</h1>
            <p>Record professional demos with YOLO detection and depth analysis</p>
        </div>
        
        <div class="controls">
            <input type="text" id="sessionName" placeholder="demo_session" style="padding: 12px; margin: 0 10px; border-radius: 4px; border: 1px solid #555; background: #2a2a2a; color: white;">
            <button class="record-btn" onclick="startRecording()">🔴 Start Recording</button>
            <button class="stop-btn" onclick="stopRecording()">⏹️ Stop Recording</button>
            
            <div class="status" id="status">
                Status: Ready to record
            </div>
        </div>
        
        <div class="stream-container">
            <div class="stream-box">
                <h3>🎯 RGB + YOLO Detection</h3>
                <img src="/video_rgb" alt="RGB with YOLO">
            </div>
            
            <div class="stream-box">
                <h3>📏 Depth Map</h3>
                <img src="/video_depth" alt="Depth Map">
            </div>
        </div>
        
        <div class="features">
            <h4>📹 Recording Features:</h4>
            <ul>
                <li><strong>Multi-format Output:</strong> RGB detection video, depth map video, combined view</li>
                <li><strong>Detection Logging:</strong> CSV export with all detection data and distances</li>
                <li><strong>Session Analytics:</strong> JSON summary with statistics</li>
                <li><strong>Professional Quality:</strong> MP4 videos ready for demos</li>
                <li><strong>Real-time Indicators:</strong> Recording status shown on video feed</li>
            </ul>
        </div>
        
        <script>
            function startRecording() {
                const sessionName = document.getElementById('sessionName').value || 'demo';
                fetch('/start_recording', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_name: sessionName})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.querySelector('.record-btn').style.backgroundColor = '#cc3333';
                        document.querySelector('.record-btn').innerHTML = '🔴 Recording...';
                    }
                });
            }
            
            function stopRecording() {
                fetch('/stop_recording', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    document.querySelector('.record-btn').style.backgroundColor = '#ff4444';
                    document.querySelector('.record-btn').innerHTML = '🔴 Start Recording';
                    if (data.session_dir) {
                        alert(`Recording saved to: ${data.session_dir}`);
                    }
                });
            }
            
            function updateStatus() {
                fetch('/recording_status')
                .then(response => response.json())
                .then(data => {
                    const status = document.getElementById('status');
                    if (data.recording) {
                        status.innerHTML = `🔴 Recording: ${data.session_id}<br>` +
                                         `Duration: ${data.duration.toFixed(1)}s | ` +
                                         `Detections: ${data.total_detections} | ` +
                                         `Vehicles: ${data.vehicle_detections} | ` +
                                         `People: ${data.person_detections}`;
                    } else {
                        status.innerHTML = 'Status: Ready to record';
                    }
                });
            }
            
            setInterval(function() {
                var timestamp = new Date().getTime();
                document.querySelector('img[src^="/video_rgb"]').src = '/video_rgb?' + timestamp;
                document.querySelector('img[src^="/video_depth"]').src = '/video_depth?' + timestamp;
                updateStatus();
            }, 100);
        </script>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data = request.json
    session_name = data.get('session_name', 'demo')
    success = recorder.start_recording(session_name)
    return jsonify({'success': success})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    success = recorder.stop_recording()
    return jsonify({
        'success': success,
        'session_dir': recorder.session_stats.get('session_dir', '') if success else ''
    })

@app.route('/recording_status')
def recording_status():
    return jsonify(recorder.get_recording_status())

def generate_rgb():
    while True:
        frame = recorder.get_frame_rgb()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.066)

def generate_depth():
    while True:
        frame = recorder.get_frame_depth()
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
    print("Starting ZED Demo Recorder Server...")
    if recorder.start_streaming():
        print("System ready!")
        print("Access the recorder at: http://192.168.1.196:5000")
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            recorder.stop_streaming()
    else:
        print("Failed to initialize camera")