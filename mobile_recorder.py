#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import threading
import time
from flask import Flask, Response, jsonify, request, send_file
import os
import json
import csv
import glob
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
            self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
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
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
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

class ZEDMobileRecorder:
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
        self.detection_log = []
        self.session_stats = {
            'total_detections': 0,
            'vehicle_detections': 0,
            'person_detections': 0,
            'detection_history': []
        }
        
        # Performance optimization
        self.jpeg_quality = 70  # Mobile-optimized quality
        
        self.yolo = YOLODetector()
        self.yolo_enabled = self.yolo.initialize()
        
    def initialize_camera(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_minimum_distance = 200
        init_params.depth_maximum_distance = 20000

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
        if self.recording:
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_session_id = f"{session_name}_{timestamp}"
        session_dir = f"recordings/{self.recording_session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cam_info = self.zed.get_camera_information()
        w = cam_info.camera_configuration.resolution.width
        h = cam_info.camera_configuration.resolution.height
        
        self.video_writer_rgb = cv2.VideoWriter(
            f"{session_dir}/rgb_detections.mp4", fourcc, 30.0, (w, h))
        self.video_writer_depth = cv2.VideoWriter(
            f"{session_dir}/depth_map.mp4", fourcc, 30.0, (w, h))
        self.video_writer_combined = cv2.VideoWriter(
            f"{session_dir}/combined_view.mp4", fourcc, 30.0, (w*2, h))
        
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
        if not self.recording:
            return False
            
        self.recording = False
        
        if self.video_writer_rgb:
            self.video_writer_rgb.release()
        if self.video_writer_depth:
            self.video_writer_depth.release()
        if self.video_writer_combined:
            self.video_writer_combined.release()
        
        session_dir = self.session_stats['session_dir']
        
        if self.detection_log:
            with open(f"{session_dir}/detections.csv", 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'frame_number', 'object_id', 'label', 'confidence', 
                             'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'distance_mm', 'center_x', 'center_y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.detection_log)
        
        self.session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats['duration_seconds'] = time.time() - self.recording_start_time
        self.session_stats['total_frames'] = len(self.detection_log)
        
        with open(f"{session_dir}/session_summary.json", 'w') as f:
            json.dump(self.session_stats, f, indent=2)
        
        self.generate_summary_report(session_dir)
        print(f"⏹️  Recording stopped: {self.recording_session_id}")
        return True

    def generate_summary_report(self, session_dir):
        report_lines = [
            f"ZED Camera Detection Demo Report",
            f"=" * 40,
            f"Session ID: {self.session_stats['session_id']}",
            f"Duration: {self.session_stats['duration_seconds']:.1f} seconds",
            f"Total Detections: {self.session_stats['total_detections']}",
            f"Vehicle Detections: {self.session_stats['vehicle_detections']}",
            f"Person Detections: {self.session_stats['person_detections']}",
        ]
        
        with open(f"{session_dir}/README.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def get_depth_at_point(self, point_cloud, depth_map, x, y, radius=3):
        height, width = depth_map.shape
        
        err, point_cloud_value = point_cloud.get_value(x, y)
        if not err and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]) and point_cloud_value[2] > 0:
            distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
            return distance
        
        if 0 <= x < width and 0 <= y < height:
            depth_value = depth_map[y, x]
            if math.isfinite(depth_value) and depth_value > 0:
                return depth_value
        
        return None

    def log_detection(self, detection, frame_number, timestamp, distance):
        if not self.recording:
            return
            
        x, y, w, h = detection['bbox']
        log_entry = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'object_id': len(self.detection_log),
            'label': detection['label'],
            'confidence': detection['confidence'],
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'distance_mm': distance if distance else 0,
            'center_x': detection['center'][0],
            'center_y': detection['center'][1]
        }
        
        self.detection_log.append(log_entry)
        self.session_stats['total_detections'] += 1
        
        if detection['label'] in ['car', 'truck', 'bus', 'motorbike']:
            self.session_stats['vehicle_detections'] += 1
        elif detection['label'] == 'person':
            self.session_stats['person_detections'] += 1

    def draw_detections(self, image, detections, point_cloud, depth_map, frame_number):
        current_timestamp = datetime.now().isoformat()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            label = detection['label']
            color = detection['color']
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            center_x, center_y = detection['center']
            distance = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
            
            self.log_detection(detection, frame_number, current_timestamp, distance)
            
            label_text = f"{label} {confidence*100:.0f}%"
            distance_text = f"{distance:.0f}mm" if distance else "N/A"
            
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x, y-25), (x + max(label_size[0], 80) + 10, y), color, -1)
            cv2.putText(image, label_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, distance_text, (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                rgb_image = image.get_data()[:, :, :3].copy()
                rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
                depth_map = depth.get_data()

                height, width = rgb_image.shape[:2]
                center_x = width // 2
                center_y = height // 2

                # Process depth
                depth_normalized = np.nan_to_num(depth_map, nan=0.0)
                valid_depth_mask = (depth_normalized > 200) & (depth_normalized < 20000)
                valid_pixels = np.sum(valid_depth_mask)
                total_pixels = depth_normalized.size
                self.depth_stats = {"valid_pixels": valid_pixels, "total_pixels": total_pixels}
                
                depth_for_display = depth_normalized.copy()
                depth_for_display[depth_for_display > 10000] = 10000
                depth_colored = cv2.normalize(depth_for_display, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

                # Get center depth
                distance = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
                self.distance_text = f"Center: {distance:.1f}mm" if distance else "Center: N/A"

                # YOLO detection (every other frame)
                object_count = 0
                if self.yolo_enabled and (self.frame_count % 2 == 0):
                    detections = self.yolo.detect(rgb_image)
                    object_count = self.draw_detections(rgb_image, detections, point_cloud, depth_map, self.frame_count)
                    
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
                
                cv2.putText(rgb_image, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
                cv2.putText(rgb_image, f"FPS: {self.fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
                
                if self.yolo_enabled:
                    cv2.putText(rgb_image, f"Objects: {object_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
                    
                cv2.putText(rgb_image, f"Valid depth: {100*valid_pixels/total_pixels:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, recording_color, 2)
                
                # Recording indicator
                if self.recording:
                    cv2.putText(rgb_image, "[REC]", (width-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(depth_colored, "[REC]", (width-80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(depth_colored, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if self.yolo_enabled:
                    cv2.putText(depth_colored, f"Objects: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw crosshairs
                cv2.line(rgb_image, (center_x-30, center_y), (center_x+30, center_y), (0, 255, 255), 2)
                cv2.line(rgb_image, (center_x, center_y-30), (center_x, center_y+30), (0, 255, 255), 2)
                cv2.circle(rgb_image, (center_x, center_y), 10, (0, 255, 255), 2)
                
                cv2.line(depth_colored, (center_x-30, center_y), (center_x+30, center_y), (255, 255, 255), 2)
                cv2.line(depth_colored, (center_x, center_y-30), (center_x, center_y+30), (255, 255, 255), 2)
                cv2.circle(depth_colored, (center_x, center_y), 10, (255, 255, 255), 2)

                # Record if active
                if self.recording:
                    if self.video_writer_rgb:
                        self.video_writer_rgb.write(rgb_image)
                    if self.video_writer_depth:
                        self.video_writer_depth.write(depth_colored)
                    if self.video_writer_combined:
                        combined = np.hstack((rgb_image, depth_colored))
                        self.video_writer_combined.write(combined)

                self.current_rgb = rgb_image
                self.current_depth = depth_colored
                self.frame_count += 1

            time.sleep(0.03)  # 33ms = ~30 FPS max

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
            _, buffer = cv2.imencode('.jpg', self.current_rgb, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            return buffer.tobytes()
        return None

    def get_frame_depth(self):
        if self.current_depth is not None:
            _, buffer = cv2.imencode('.jpg', self.current_depth, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
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
    
    def get_recorded_sessions(self):
        """Get list of all recorded sessions"""
        sessions = []
        if os.path.exists("recordings"):
            for session_dir in os.listdir("recordings"):
                session_path = os.path.join("recordings", session_dir)
                if os.path.isdir(session_path):
                    summary_path = os.path.join(session_path, "session_summary.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                            sessions.append({
                                'id': session_dir,
                                'name': session_dir,
                                'duration': summary.get('duration_seconds', 0),
                                'detections': summary.get('total_detections', 0),
                                'vehicles': summary.get('vehicle_detections', 0),
                                'start_time': summary.get('start_time', ''),
                                'path': session_path
                            })
        return sorted(sessions, key=lambda x: x['start_time'], reverse=True)

# Flask app
app = Flask(__name__)
recorder = ZEDMobileRecorder()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ZED Mobile Recorder</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            margin: 0; 
            padding: 10px; 
            background-color: #1a1a1a; 
            color: white; 
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        .container { 
            max-width: 100%; 
            margin: 0 auto;
        }
        .controls {
            text-align: center;
            margin-bottom: 15px;
            padding: 15px;
            background-color: #333;
            border-radius: 8px;
        }
        .controls button {
            padding: 10px 20px;
            margin: 5px;
            font-size: 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input {
            padding: 10px;
            margin: 5px;
            border-radius: 4px;
            border: 1px solid #555;
            background: #2a2a2a;
            color: white;
            width: 150px;
        }
        .record-btn { background-color: #ff4444; color: white; }
        .stop-btn { background-color: #888; color: white; }
        .view-btn { background-color: #4CAF50; color: white; }
        .status {
            padding: 8px;
            margin: 8px 0;
            background-color: #2a2a2a;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            text-align: left;
        }
        .tabs {
            display: flex;
            background-color: #333;
            border-radius: 8px 8px 0 0;
            overflow: hidden;
        }
        .tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            background-color: #333;
            color: #ccc;
        }
        .tab.active {
            background-color: #555;
            color: white;
        }
        .tab-content {
            background-color: #2a2a2a;
            border-radius: 0 0 8px 8px;
            padding: 15px;
            min-height: 400px;
        }
        .tab-pane {
            display: none;
        }
        .tab-pane.active {
            display: block;
        }
        .stream-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .stream-box {
            border: 2px solid #333;
            border-radius: 8px;
            padding: 10px;
            background-color: #1a1a1a;
            flex: 1;
            min-width: 300px;
            max-width: 450px;
        }
        .stream-box h3 {
            margin: 0 0 10px 0;
            text-align: center;
            font-size: 16px;
        }
        img { 
            width: 100%;
            height: auto; 
            border-radius: 4px;
        }
        .sessions-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .session-item {
            background-color: #333;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
        }
        .session-item:hover {
            background-color: #444;
        }
        .session-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .session-stats {
            font-size: 12px;
            color: #ccc;
        }
        .video-player {
            width: 100%;
            max-width: 800px;
            margin: 10px auto;
            display: block;
            border-radius: 8px;
        }
        .player-controls {
            text-align: center;
            margin: 10px 0;
        }
        .player-controls button {
            padding: 8px 16px;
            margin: 5px;
            background-color: #555;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        @media (max-width: 768px) {
            .stream-container { flex-direction: column; }
            .stream-box { max-width: 100%; }
            .controls button { width: 100%; margin: 2px 0; }
            input { width: 100%; margin: 2px 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="controls">
            <h1>📱 ZED Mobile Recorder</h1>
            <input type="text" id="sessionName" placeholder="session_name">
            <button class="record-btn" onclick="startRecording()">🔴 Record</button>
            <button class="stop-btn" onclick="stopRecording()">⏹️ Stop</button>
            <button onclick="alert('Test button works!')">Test JS</button>
            
            <div class="status" id="status">
                Status: Ready to record
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="console.log('Live tab clicked'); switchTab('live')">📹 Live Feed</div>
            <div class="tab" onclick="console.log('Recordings tab clicked'); switchTab('recordings')">📁 Recordings</div>
        </div>
        
        <div class="tab-content">
            <div class="tab-pane active" id="live-pane">
                <div class="stream-container">
                    <div class="stream-box">
                        <h3>🎯 RGB + Detection</h3>
                        <img src="/video_rgb" alt="RGB Stream">
                    </div>
                    
                    <div class="stream-box">
                        <h3>📏 Depth Map</h3>
                        <img src="/video_depth" alt="Depth Stream">
                    </div>
                </div>
            </div>
            
            <div class="tab-pane" id="recordings-pane">
                <div id="sessionsList">
                    <p>Loading recordings...</p>
                </div>
                <div id="videoPlayer" style="display: none;">
                    <h3 id="playerTitle">Recording Player</h3>
                    <div class="player-controls">
                        <button onclick="playVideo('rgb')">▶️ RGB Detection</button>
                        <button onclick="playVideo('depth')">▶️ Depth Map</button>
                        <button onclick="playVideo('combined')">▶️ Combined View</button>
                        <button onclick="closePlayer()">❌ Close</button>
                    </div>
                    <video id="videoElement" class="video-player" controls>
                        Your browser does not support video playback.
                    </video>
                </div>
            </div>
        </div>
        
        <script>
            // Global error handler
            window.onerror = function(msg, url, line) {
                console.error('JavaScript Error:', msg, 'at line', line);
                return false;
            };

            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Page loaded, JavaScript working');
                // Test buttons exist
                const recordBtn = document.querySelector('.record-btn');
                const stopBtn = document.querySelector('.stop-btn');
                console.log('Record button found:', recordBtn ? 'YES' : 'NO');
                console.log('Stop button found:', stopBtn ? 'YES' : 'NO');
                
                // Check if functions exist  
                setTimeout(function() {
                    console.log('startRecording function exists:', typeof startRecording);
                    console.log('stopRecording function exists:', typeof stopRecording);
                    console.log('switchTab function exists:', typeof switchTab);
                    console.log('loadRecordings function exists:', typeof loadRecordings);
                }, 100);
                
                // Add click listeners as backup
                if (recordBtn) {
                    recordBtn.addEventListener('click', function(e) {
                        console.log('Record button clicked via addEventListener');
                        e.preventDefault();
                        if (typeof startRecording === 'function') {
                            startRecording();
                        } else {
                            alert('startRecording function not found!');
                        }
                    });
                }
                if (stopBtn) {
                    stopBtn.addEventListener('click', function(e) {
                        console.log('Stop button clicked via addEventListener');
                        e.preventDefault();
                        if (typeof stopRecording === 'function') {
                            stopRecording();
                        } else {
                            alert('stopRecording function not found!');
                        }
                    });
                }
            });
            
            let currentSession = null;
            
            function switchTab(tabName) {
                // Hide all panes
                document.querySelectorAll('.tab-pane').forEach(pane => {
                    pane.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected pane
                document.getElementById(tabName + '-pane').classList.add('active');
                event.target.classList.add('active');
                
                if (tabName === 'recordings') {
                    loadRecordings();
                }
            }
            
            function startRecording() {
                const sessionName = document.getElementById('sessionName').value || 'mobile_demo';
                console.log('Starting recording:', sessionName);
                fetch('/start_recording', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_name: sessionName})
                })
                .then(response => {
                    console.log('Start recording response:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('Start recording data:', data);
                    if (data.success) {
                        document.querySelector('.record-btn').style.backgroundColor = '#cc3333';
                        document.querySelector('.record-btn').innerHTML = '🔴 Recording...';
                        document.querySelector('.record-btn').disabled = true;
                        alert('Recording started successfully!');
                    } else {
                        alert('Failed to start recording');
                    }
                })
                .catch(error => {
                    console.error('Recording error:', error);
                    alert('Error starting recording: ' + error);
                });
            }
            
            function stopRecording() {
                console.log('Stopping recording');
                fetch('/stop_recording', {method: 'POST'})
                .then(response => {
                    console.log('Stop recording response:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('Stop recording data:', data);
                    document.querySelector('.record-btn').style.backgroundColor = '#ff4444';
                    document.querySelector('.record-btn').innerHTML = '🔴 Record';
                    document.querySelector('.record-btn').disabled = false;
                    if (data.success) {
                        loadRecordings();
                        alert('Recording saved! Check Recordings tab to view.');
                    } else {
                        alert('Failed to stop recording');
                    }
                })
                .catch(error => {
                    console.error('Stop recording error:', error);
                    alert('Error stopping recording: ' + error);
                });
            }
            
            function updateStatus() {
                fetch('/recording_status')
                .then(response => response.json())
                .then(data => {
                    const status = document.getElementById('status');
                    if (data.recording) {
                        status.innerHTML = '🔴 Recording: ' + data.session_id + '<br>' +
                                         'Duration: ' + data.duration.toFixed(1) + 's | ' +
                                         'Detections: ' + data.total_detections + ' | ' +
                                         'Vehicles: ' + data.vehicle_detections;
                    } else {
                        status.innerHTML = 'Status: Ready to record';
                    }
                });
            }
            
            function loadRecordings() {
                console.log('Loading recordings...');
                fetch('/get_sessions')
                .then(response => {
                    console.log('Get sessions response:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('Sessions data:', data);
                    const sessionsList = document.getElementById('sessionsList');
                    if (!data.sessions || data.sessions.length === 0) {
                        sessionsList.innerHTML = '<p>No recordings found. Start recording to create sessions!</p>';
                    } else {
                        sessionsList.innerHTML = data.sessions.map(session => 
                            '<div class="session-item" onclick="selectSession(\'' + session.id + '\')"><div class="session-name">' + session.name + '</div><div class="session-stats">Duration: ' + session.duration.toFixed(1) + 's | Detections: ' + session.detections + ' | Vehicles: ' + session.vehicles + '<br>Recorded: ' + new Date(session.start_time).toLocaleString() + '</div></div>'
                        ).join('');
                    }
                })
                .catch(error => {
                    console.error('Load recordings error:', error);
                    document.getElementById('sessionsList').innerHTML = '<p>Error loading recordings: ' + error + '</p>';
                });
            }
            
            function selectSession(sessionId) {
                currentSession = sessionId;
                document.getElementById('videoPlayer').style.display = 'block';
                document.getElementById('playerTitle').textContent = 'Playing: ' + sessionId;
                document.getElementById('sessionsList').style.display = 'none';
            }
            
            function playVideo(type) {
                if (!currentSession) {
                    alert('No session selected');
                    return;
                }
                
                console.log('Playing video:', type, 'for session:', currentSession);
                const video = document.getElementById('videoElement');
                let filename;
                switch(type) {
                    case 'rgb': filename = 'rgb_detections.mp4'; break;
                    case 'depth': filename = 'depth_map.mp4'; break;
                    case 'combined': filename = 'combined_view.mp4'; break;
                    default: 
                        alert('Invalid video type: ' + type);
                        return;
                }
                
                const videoUrl = '/video_file/' + currentSession + '/' + filename;
                console.log('Loading video:', videoUrl);
                video.src = videoUrl;
                video.load();
                
                video.onerror = function() {
                    alert('Error loading video: ' + filename);
                };
            }
            
            function closePlayer() {
                document.getElementById('videoPlayer').style.display = 'none';
                document.getElementById('sessionsList').style.display = 'block';
                currentSession = null;
            }
            
            // Refresh live feed and status
            setInterval(function() {
                if (document.getElementById('live-pane').classList.contains('active')) {
                    var timestamp = new Date().getTime();
                    document.querySelector('img[src^="/video_rgb"]').src = '/video_rgb?' + timestamp;
                    document.querySelector('img[src^="/video_depth"]').src = '/video_depth?' + timestamp;
                }
                updateStatus();
            }, 250); // Mobile-optimized refresh rate
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
    session_name = data.get('session_name', 'mobile_demo')
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

@app.route('/get_sessions')
def get_sessions():
    sessions = recorder.get_recorded_sessions()
    return jsonify({'sessions': sessions})

@app.route('/video_file/<session_id>/<filename>')
def video_file(session_id, filename):
    """Serve recorded video files"""
    file_path = os.path.join("recordings", session_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "File not found", 404

def generate_rgb():
    while True:
        frame = recorder.get_frame_rgb()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

def generate_depth():
    while True:
        frame = recorder.get_frame_depth()
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
    print("Starting ZED Mobile Recorder Server...")
    if recorder.start_streaming():
        print("System ready!")
        print("Access the mobile interface at: http://192.168.1.196:5000")
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            recorder.stop_streaming()
    else:
        print("Failed to initialize camera")