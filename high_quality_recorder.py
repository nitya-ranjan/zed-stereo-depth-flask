#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import threading
import time
from flask import Flask, Response, jsonify, request
import os
import json
import csv
from datetime import datetime

class HighQualityYOLO:
    def __init__(self):
        self.net = None
        self.output_layers = None
        self.classes = None
        self.colors = None
        self.initialized = False
        
    def initialize(self):
        """Initialize high-quality YOLO detector"""
        try:
            # Try full YOLOv4 first (better accuracy)
            if os.path.exists("yolov4.weights") and os.path.exists("yolov4.cfg"):
                print("🎯 Loading YOLOv4 (full model) for maximum accuracy...")
                self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
                print("✅ YOLOv4 full model loaded successfully!")
            else:
                print("⚠️  YOLOv4 full model not found, using YOLOv4-tiny...")
                self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
                print("✅ YOLOv4-tiny model loaded")
            
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            self.initialized = True
            
            # Use GPU if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("🚀 CUDA acceleration enabled for YOLO!")
            
            return True
        except Exception as e:
            print(f"❌ Failed to initialize YOLO: {e}")
            return False
    
    def detect(self, image, confidence_threshold=0.5, nms_threshold=0.4):
        """Detect objects with higher accuracy settings"""
        if not self.initialized:
            return []
        
        height, width, channels = image.shape
        
        # Use higher resolution for better accuracy
        blob_size = 608  # Higher resolution for better detection (vs 416)
        blob = cv2.dnn.blobFromImage(image, 0.00392, (blob_size, blob_size), (0, 0, 0), True, crop=False)
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
                
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
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

class HighQualityZEDRecorder:
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
        
        # High quality recording settings
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
        
        # High quality settings
        self.video_fps = 30.0  # Full 30 FPS recording
        self.jpeg_quality = 95  # High quality for web streaming
        self.yolo_every_n_frames = 1  # Run YOLO on every frame for accuracy
        
        self.yolo = HighQualityYOLO()
        self.yolo_enabled = self.yolo.initialize()
        
    def initialize_camera(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        
        # High quality camera settings
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Highest quality depth
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Could use HD1080 for even higher quality
        init_params.camera_fps = 30
        init_params.depth_minimum_distance = 300  # Better minimum distance
        init_params.depth_maximum_distance = 15000  # Optimized for vehicle detection

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"❌ Camera Open Error: {repr(err)}")
            return False

        cam_info = self.zed.get_camera_information()
        print(f"🎥 ZED Camera initialized - Serial: {cam_info.serial_number}")
        print(f"📏 Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
        print(f"🎯 YOLO Detection: {'Enabled' if self.yolo_enabled else 'Disabled'}")
        print(f"📊 Depth Mode: ULTRA (highest quality)")
        return True

    def start_recording(self, session_name="high_quality_demo"):
        if self.recording:
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_session_id = f"{session_name}_{timestamp}"
        session_dir = f"recordings/{self.recording_session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # High quality video codec and settings
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Better codec
        cam_info = self.zed.get_camera_information()
        w = cam_info.camera_configuration.resolution.width
        h = cam_info.camera_configuration.resolution.height
        
        # High quality video writers with full FPS
        self.video_writer_rgb = cv2.VideoWriter(
            f"{session_dir}/rgb_detections.mp4", fourcc, self.video_fps, (w, h))
        self.video_writer_depth = cv2.VideoWriter(
            f"{session_dir}/depth_map.mp4", fourcc, self.video_fps, (w, h))
        self.video_writer_combined = cv2.VideoWriter(
            f"{session_dir}/combined_view.mp4", fourcc, self.video_fps, (w*2, h))
        
        self.detection_log = []
        self.session_stats = {
            'session_id': self.recording_session_id,
            'start_time': datetime.now().isoformat(),
            'total_detections': 0,
            'vehicle_detections': 0,
            'person_detections': 0,
            'detection_history': [],
            'session_dir': session_dir,
            'recording_fps': self.video_fps,
            'yolo_frequency': 'Every frame',
            'quality_mode': 'High Quality'
        }
        
        self.recording = True
        self.recording_start_time = time.time()
        print(f"🔴 High Quality Recording started: {self.recording_session_id}")
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
        self.session_stats['actual_fps'] = self.session_stats['total_frames'] / self.session_stats['duration_seconds'] if self.session_stats['duration_seconds'] > 0 else 0
        
        with open(f"{session_dir}/session_summary.json", 'w') as f:
            json.dump(self.session_stats, f, indent=2)
        
        self.generate_summary_report(session_dir)
        print(f"⏹️  High Quality Recording stopped: {self.recording_session_id}")
        return True

    def generate_summary_report(self, session_dir):
        report_lines = [
            f"ZED High Quality Recording Report",
            f"=" * 40,
            f"Session ID: {self.session_stats['session_id']}",
            f"Duration: {self.session_stats['duration_seconds']:.1f} seconds",
            f"Recorded FPS: {self.session_stats['recording_fps']}",
            f"Actual FPS: {self.session_stats['actual_fps']:.1f}",
            f"Quality Mode: {self.session_stats['quality_mode']}",
            f"Total Detections: {self.session_stats['total_detections']}",
            f"Vehicle Detections: {self.session_stats['vehicle_detections']}",
            f"Person Detections: {self.session_stats['person_detections']}",
            f"",
            f"Files Generated:",
            f"- rgb_detections.mp4 (RGB video with detection boxes)",
            f"- depth_map.mp4 (High quality depth visualization)",
            f"- combined_view.mp4 (Side-by-side RGB + depth)",
            f"- detections.csv (Frame-by-frame detection data)",
            f"- session_summary.json (Complete session metadata)",
        ]
        
        with open(f"{session_dir}/README.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def get_accurate_depth(self, point_cloud, depth_map, x, y, radius=5):
        """Enhanced depth measurement with better accuracy"""
        height, width = depth_map.shape
        
        # Try point cloud first
        err, point_cloud_value = point_cloud.get_value(x, y)
        if not err and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]) and point_cloud_value[2] > 0:
            distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
            return distance
        
        # Enhanced depth map sampling with averaging
        if 0 <= x < width and 0 <= y < height:
            # Sample multiple points around the center for better accuracy
            valid_depths = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        depth_value = depth_map[py, px]
                        if math.isfinite(depth_value) and depth_value > 0:
                            valid_depths.append(depth_value)
            
            if valid_depths:
                # Return median for better accuracy
                return np.median(valid_depths)
        
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
            
            # Enhanced visualization
            thickness = 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            center_x, center_y = detection['center']
            distance = self.get_accurate_depth(point_cloud, depth_map, center_x, center_y)
            
            self.log_detection(detection, frame_number, current_timestamp, distance)
            
            # Enhanced text display
            label_text = f"{label} {confidence*100:.1f}%"
            distance_text = f"{distance:.0f}mm" if distance else "N/A"
            
            # Better text background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x, y-30), (x + max(label_size[0], 80) + 10, y), color, -1)
            cv2.putText(image, label_text, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, distance_text, (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Enhanced center point
            cv2.circle(image, (center_x, center_y), 6, color, -1)
            cv2.circle(image, (center_x, center_y), 8, (255, 255, 255), 2)
        
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

                # Enhanced depth processing
                depth_normalized = np.nan_to_num(depth_map, nan=0.0)
                valid_depth_mask = (depth_normalized > 300) & (depth_normalized < 15000)
                valid_pixels = np.sum(valid_depth_mask)
                total_pixels = depth_normalized.size
                
                # Better depth visualization
                depth_for_display = depth_normalized.copy()
                depth_for_display[depth_for_display > 10000] = 10000
                depth_colored = cv2.normalize(depth_for_display, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_TURBO)  # Better colormap

                # Enhanced center depth measurement
                distance = self.get_accurate_depth(point_cloud, depth_map, center_x, center_y)
                self.distance_text = f"Center: {distance:.1f}mm" if distance else "Center: N/A"

                # High accuracy YOLO detection (every frame)
                object_count = 0
                if self.yolo_enabled and (self.frame_count % self.yolo_every_n_frames == 0):
                    detections = self.yolo.detect(rgb_image, confidence_threshold=0.5, nms_threshold=0.4)
                    object_count = self.draw_detections(rgb_image, detections, point_cloud, depth_map, self.frame_count)
                    
                    # Draw detection boxes on depth map too
                    for detection in detections:
                        x, y, w, h = detection['bbox']
                        cv2.rectangle(depth_colored, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # Calculate accurate FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Enhanced overlays
                status_color = (0, 255, 0) if self.yolo_enabled else (0, 255, 255)
                recording_color = (0, 0, 255) if self.recording else status_color
                
                cv2.putText(rgb_image, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
                cv2.putText(rgb_image, f"FPS: {self.fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
                
                if self.yolo_enabled:
                    cv2.putText(rgb_image, f"Objects: {object_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
                    cv2.putText(rgb_image, "HIGH QUALITY MODE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                cv2.putText(rgb_image, f"Valid depth: {100*valid_pixels/total_pixels:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, recording_color, 2)
                
                # Recording indicator
                if self.recording:
                    cv2.putText(rgb_image, "[HQ REC]", (width-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(depth_colored, "[HQ REC]", (width-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Enhanced depth overlays
                cv2.putText(depth_colored, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if self.yolo_enabled:
                    cv2.putText(depth_colored, f"Objects: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Enhanced crosshairs
                cv2.line(rgb_image, (center_x-40, center_y), (center_x+40, center_y), (0, 255, 255), 3)
                cv2.line(rgb_image, (center_x, center_y-40), (center_x, center_y+40), (0, 255, 255), 3)
                cv2.circle(rgb_image, (center_x, center_y), 15, (0, 255, 255), 3)
                
                cv2.line(depth_colored, (center_x-40, center_y), (center_x+40, center_y), (255, 255, 255), 3)
                cv2.line(depth_colored, (center_x, center_y-40), (center_x, center_y+40), (255, 255, 255), 3)
                cv2.circle(depth_colored, (center_x, center_y), 15, (255, 255, 255), 3)

                # Record at full 30 FPS if active
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

            # Precise timing for 30 FPS
            time.sleep(0.033)

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
            'person_detections': self.session_stats['person_detections'],
            'quality_mode': 'High Quality'
        }

# Flask app
app = Flask(__name__)
recorder = HighQualityZEDRecorder()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ZED High Quality Recorder</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            margin: 0; 
            padding: 20px; 
            background-color: #0f0f0f; 
            color: white; 
            font-family: Arial, sans-serif;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff88;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .controls {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            border-radius: 10px;
            border: 2px solid #00ff88;
        }
        .controls button {
            padding: 12px 25px;
            margin: 10px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
        }
        input {
            padding: 12px;
            margin: 10px;
            border-radius: 6px;
            border: 2px solid #00ff88;
            background: #1a1a1a;
            color: white;
            font-size: 16px;
            width: 200px;
        }
        .record-btn { 
            background: linear-gradient(135deg, #ff4444, #cc3333);
            color: white; 
        }
        .record-btn:hover { 
            background: linear-gradient(135deg, #ff6666, #ff4444);
            transform: scale(1.05);
        }
        .stop-btn { 
            background: linear-gradient(135deg, #888, #666);
            color: white; 
        }
        .stop-btn:hover { 
            background: linear-gradient(135deg, #aaa, #888);
            transform: scale(1.05);
        }
        .status {
            padding: 15px;
            margin: 15px 0;
            background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            border-left: 5px solid #00ff88;
            font-size: 14px;
        }
        .stream-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .stream-box {
            border: 3px solid #00ff88;
            border-radius: 10px;
            padding: 15px;
            background: linear-gradient(135deg, #1a1a1a, #0f0f0f);
            flex: 1;
            min-width: 400px;
            max-width: 650px;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }
        .stream-box h3 {
            margin: 0 0 15px 0;
            text-align: center;
            color: #00ff88;
            font-size: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        img { 
            width: 100%;
            height: auto; 
            border-radius: 8px;
            border: 2px solid #333;
        }
        .quality-badge {
            display: inline-block;
            background: linear-gradient(135deg, #00ff88, #00cc66);
            color: black;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
            margin-left: 10px;
        }
        @media (max-width: 1000px) {
            .stream-container { flex-direction: column; }
            .stream-box { max-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 ZED High Quality Recorder<span class="quality-badge">HQ MODE</span></h1>
        
        <div class="controls">
            <input type="text" id="sessionName" placeholder="Enter session name" value="hq_demo">
            <br>
            <button class="record-btn" onclick="startRecording()">🔴 Start HQ Recording</button>
            <button class="stop-btn" onclick="stopRecording()">⏹️ Stop Recording</button>
            
            <div class="status" id="status">
                🟢 High Quality Mode Active - Ready to record at 30 FPS
            </div>
        </div>
        
        <div class="stream-container">
            <div class="stream-box">
                <h3>🎯 RGB + Enhanced Detection</h3>
                <img src="/video_rgb" alt="RGB Stream" id="rgbStream">
            </div>
            
            <div class="stream-box">
                <h3>📏 Ultra-Quality Depth</h3>
                <img src="/video_depth" alt="Depth Stream" id="depthStream">
            </div>
        </div>
        
        <script>
            function startRecording() {
                const sessionName = document.getElementById('sessionName').value || 'hq_demo';
                fetch('/start_recording', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_name: sessionName})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.querySelector('.record-btn').style.background = 'linear-gradient(135deg, #cc0000, #990000)';
                        document.querySelector('.record-btn').innerHTML = '🔴 Recording HQ...';
                    }
                });
            }
            
            function stopRecording() {
                fetch('/stop_recording', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    document.querySelector('.record-btn').style.background = 'linear-gradient(135deg, #ff4444, #cc3333)';
                    document.querySelector('.record-btn').innerHTML = '🔴 Start HQ Recording';
                    if (data.success) {
                        alert('High Quality Recording saved successfully!');
                    }
                });
            }
            
            function updateStatus() {
                fetch('/recording_status')
                .then(response => response.json())
                .then(data => {
                    const status = document.getElementById('status');
                    if (data.recording) {
                        status.innerHTML = '🔴 HIGH QUALITY RECORDING: ' + data.session_id + '<br>' +
                                         'Duration: ' + data.duration.toFixed(1) + 's | ' +
                                         'Detections: ' + data.total_detections + ' | ' +
                                         'Vehicles: ' + data.vehicle_detections + ' | ' +
                                         'Mode: ' + data.quality_mode;
                    } else {
                        status.innerHTML = '🟢 High Quality Mode Active - Ready to record at 30 FPS';
                    }
                });
            }
            
            // Refresh streams and status
            setInterval(function() {
                var timestamp = new Date().getTime();
                document.getElementById('rgbStream').src = '/video_rgb?' + timestamp;
                document.getElementById('depthStream').src = '/video_depth?' + timestamp;
                updateStatus();
            }, 150); // Faster refresh for high quality
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
    session_name = data.get('session_name', 'hq_demo')
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
    print("🚀 Starting ZED High Quality Recorder Server...")
    if recorder.start_streaming():
        print("✅ System ready!")
        print("🌐 Access the HIGH QUALITY recorder at: http://100.79.213.91:5000")
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            recorder.stop_streaming()
    else:
        print("❌ Failed to initialize camera")