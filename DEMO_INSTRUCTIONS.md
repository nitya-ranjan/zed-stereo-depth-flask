# 🎥 ZED Camera Demo - Live Instructions

## Quick Start (30 seconds)

1. **SSH into the device:**
   ```bash
   ssh nvidia@100.79.213.91
   ```

2. **Go to demo directory:**
   ```bash
   cd /home/nvidia/zed
   ```

3. **Start the demo:**
   ```bash
   ./start_demo.sh
   ```

4. **Access from your laptop:**
   ```bash
   ssh -L 5000:localhost:5000 nvidia@100.79.213.91
   ```
   Then open: **http://localhost:5000**

## What You'll Demo

### Live Streaming Features
- **RGB Camera**: Live video with YOLO object detection
- **Depth Map**: Real-time depth sensing (blue=close, red=far)
- **Object Detection**: Bounding boxes with confidence percentages
- **Distance Measurements**: Real-time distance to detected objects in mm

### Recording Features
- **Session Recording**: Enter session name → click "REC Start Recording"
- **Live Status**: Shows recording time and detection counts
- **Stop & Save**: Click "STOP Recording" → files saved automatically

### Demo Flow Suggestions
1. **Show live streams** - point camera at various objects
2. **Demonstrate distance measurement** - move closer/farther from objects
3. **Start recording** - name it "live_demo_[timestamp]"
4. **Show object detection** - cars, people, etc. with distance
5. **Stop recording** - explain the saved files

## Recorded Files Created
Each recording session creates:
- `rgb_detections.mp4` - Video with detection boxes and distances
- `depth_map.mp4` - Depth visualization video
- `combined_view.mp4` - Side-by-side RGB + depth
- `detections.csv` - Frame-by-frame detection data
- `session_summary.json` - Session statistics

## Troubleshooting

### If server won't start:
```bash
pkill -f optimized_recorder.py
./start_demo.sh
```

### If camera not found:
```bash
ls /dev/video*  # Should show video devices
```

### If browser can't connect:
- Check SSH port forwarding is active
- Verify server shows "System ready!" message

## Stop Demo
- Press **Ctrl+C** in the terminal running the demo
- Or run: `pkill -f optimized_recorder.py`

## Technical Details
- **Camera**: ZED Stereo Camera (Serial: 3585)
- **Resolution**: 1280x720 @ 30fps
- **Detection**: YOLOv4-tiny (80 object classes)
- **Depth Range**: 200mm to 20,000mm (20cm to 20m)
- **Server**: Flask on port 5000

---
✅ **Ready for live demo!**