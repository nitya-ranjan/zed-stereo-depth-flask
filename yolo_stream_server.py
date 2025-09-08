#!/usr/bin/env python3

import pyzed.sl as sl
import cv2
import numpy as np
import math
import threading
import time
from flask import Flask, Response
import sys

class ZEDYOLOStreamer:
    def __init__(self):
        self.zed = None
        self.streaming = False
        self.current_rgb = None
        self.current_depth = None
        self.current_detection = None
        self.distance_text = "Distance: N/A"
        self.debug_info = ""
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.depth_stats = {"valid_pixels": 0, "total_pixels": 0}
        self.detection_stats = {"objects": 0, "fps": 0}
        
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

        # Enable positional tracking (required for object detection)
        positional_tracking_params = sl.PositionalTrackingParameters()
        err = self.zed.enable_positional_tracking(positional_tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Positional Tracking Error: {repr(err)}")
            return False

        # Setup object detection
        obj_detection_params = sl.ObjectDetectionParameters()
        obj_detection_params.enable_tracking = True
        obj_detection_params.enable_segmentation = False  # Faster without segmentation
        obj_detection_params.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
        obj_detection_params.filtering_mode = sl.OBJECT_FILTERING_MODE.NMS3D
        
        print("Loading YOLO object detection model...")
        err = self.zed.enable_object_detection(obj_detection_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Object Detection Error: {repr(err)}")
            # Continue without object detection if it fails
            self.object_detection_enabled = False
        else:
            self.object_detection_enabled = True
            print("YOLO object detection enabled successfully!")

        cam_info = self.zed.get_camera_information()
        print(f"ZED Camera initialized - Serial: {cam_info.serial_number}")
        print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
        return True

    def get_depth_at_point(self, point_cloud, depth_map, x, y, radius=5):
        """Get depth measurement with fallback strategies"""
        height, width = depth_map.shape
        
        # Strategy 1: Try point cloud at exact center
        err, point_cloud_value = point_cloud.get_value(x, y)
        if not err and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]) and point_cloud_value[2] > 0:
            distance = math.sqrt(sum(coord * coord for coord in point_cloud_value[:3]))
            return distance, f"Point cloud: {distance:.1f}mm"
        
        # Strategy 2: Try depth map at exact center
        if 0 <= x < width and 0 <= y < height:
            depth_value = depth_map[y, x]
            if math.isfinite(depth_value) and depth_value > 0:
                return depth_value, f"Depth map: {depth_value:.1f}mm"
        
        # Strategy 3: Average around the center point
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

    def draw_objects(self, image, objects, point_cloud):
        """Draw detected objects with 3D information"""
        object_count = 0
        
        if not objects.is_new:
            return object_count
            
        for obj in objects.object_list:
            if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                object_count += 1
                
                # Get 2D bounding box
                bbox = obj.bounding_box_2d
                if len(bbox) >= 4:
                    # Convert to integer coordinates
                    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Get object label and confidence
                    label = str(obj.label).split('.')[-1]  # Remove enum prefix
                    confidence = obj.confidence
                    
                    # Get 3D position
                    position = obj.position
                    distance_3d = math.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
                    
                    # Create label text
                    label_text = f"{label} {confidence:.0f}%"
                    distance_text = f"{distance_3d:.1f}mm"
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(image, (x1, y1-25), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                    
                    # Draw texts
                    cv2.putText(image, label_text, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(image, distance_text, (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        return object_count

    def capture_loop(self):
        if not self.zed:
            return
            
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()
        objects = sl.Objects()
        runtime_parameters = sl.RuntimeParameters()
        
        # Object detection runtime parameters
        if self.object_detection_enabled:
            obj_runtime_params = sl.ObjectDetectionRuntimeParameters()
            obj_runtime_params.detection_confidence_threshold = 40
            obj_runtime_params.object_confidence_threshold = 30

        while self.streaming:
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve images
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                
                # Retrieve objects if detection is enabled
                if self.object_detection_enabled:
                    self.zed.retrieve_objects(objects, obj_runtime_params)

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

                # Get center depth measurement
                distance, depth_method = self.get_depth_at_point(point_cloud, depth_map, center_x, center_y)
                if distance:
                    self.distance_text = f"Center: {distance:.1f}mm"
                else:
                    self.distance_text = "Center: N/A"

                # Draw detected objects
                object_count = 0
                if self.object_detection_enabled:
                    object_count = self.draw_objects(rgb_image, objects, point_cloud)
                    
                    # Also draw objects on depth map for reference
                    if objects.is_new:
                        for obj in objects.object_list:
                            if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                                bbox = obj.bounding_box_2d
                                if len(bbox) >= 4:
                                    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                                    cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (255, 255, 255), 2)

                self.detection_stats["objects"] = object_count

                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Add comprehensive overlays
                cv2.putText(rgb_image, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(rgb_image, f"FPS: {self.fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(rgb_image, f"Objects: {object_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(rgb_image, f"Valid depth: {100*valid_pixels/total_pixels:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(depth_colored, self.distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(depth_colored, f"Objects: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Draw crosshairs at center
                cv2.line(rgb_image, (center_x-30, center_y), (center_x+30, center_y), (0, 255, 255), 2)
                cv2.line(rgb_image, (center_x, center_y-30), (center_x, center_y+30), (0, 255, 255), 2)
                cv2.circle(rgb_image, (center_x, center_y), 10, (0, 255, 255), 2)
                
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
            if self.object_detection_enabled:
                self.zed.disable_object_detection()
            self.zed.disable_positional_tracking()
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
streamer = ZEDYOLOStreamer()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ZED Camera + YOLO Detection</title>
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
        @media (max-width: 1300px) {
            .stream-container { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="info">
            <h1>🤖 ZED Camera + YOLO Object Detection</h1>
            <p>Real-time object detection with 3D depth information</p>
            <p>Green boxes show detected objects with distance measurements</p>
        </div>
        
        <div class="stream-container">
            <div class="stream-box">
                <h3>🎯 RGB + Object Detection</h3>
                <img src="/video_rgb" alt="RGB with YOLO">
            </div>
            
            <div class="stream-box">
                <h3>📏 Depth Map + Objects</h3>
                <img src="/video_depth" alt="Depth with Objects">
            </div>
        </div>
        
        <div class="features">
            <h4>🔍 Detection Features:</h4>
            <ul>
                <li><strong>Object Detection:</strong> YOLO model detects people, cars, etc.</li>
                <li><strong>3D Distance:</strong> Shows distance to each detected object</li>
                <li><strong>Object Tracking:</strong> Maintains IDs across frames</li>
                <li><strong>Real-time:</strong> ~15 FPS detection with depth mapping</li>
                <li><strong>Confidence:</strong> Shows detection confidence percentage</li>
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
    print("Starting ZED Camera + YOLO Detection Server...")
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