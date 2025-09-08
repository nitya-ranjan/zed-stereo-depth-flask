# 🎥 ZED Camera Recording System

## Quick Start

### 🌐 Web Interface (Recommended)
- **URL**: http://100.79.213.91:5000 (Tailscale) or http://192.168.1.196:5000 (local)
- **Features**: Live RGB + depth streaming, YOLO vehicle detection
- **Recording**: Use Start/Stop buttons (buttons work!)
- **Mobile-optimized**: Works great on phone via SSH

### 📁 Download Recordings

Copy the download script to your local machine:
```bash
scp nvidia@100.79.213.91:/home/nvidia/zed/download_recordings.sh ./
chmod +x download_recordings.sh
```

Then use it from your local machine:
```bash
# List available recordings
./download_recordings.sh list

# Download specific recording
./download_recordings.sh download try5_20250908_002833

# Download all recordings
./download_recordings.sh all

# Get help
./download_recordings.sh help
```

## 🚗 In-Car Usage

1. **SSH to Jetson via Tailscale**:
   ```bash
   ssh nvidia@100.79.213.91
   ```

2. **Start recording server** (if not running):
   ```bash
   cd /home/nvidia/zed
   python3 optimized_recorder.py
   ```

3. **Access web interface** on your phone:
   - URL: http://100.79.213.91:5000
   - Works over Tailscale from anywhere

4. **Record your drive**:
   - Click "Start Recording" 
   - Enter session name (e.g., "highway_test")
   - Click "Stop Recording" when done

5. **Download recordings later**:
   ```bash
   ./download_recordings.sh list
   ./download_recordings.sh download highway_test
   ```

## 📊 Recording Contents

Each session creates:
- `rgb_detections.mp4` - Video with YOLO detection boxes
- `depth_map.mp4` - Colored depth visualization
- `combined_view.mp4` - Side-by-side RGB + depth
- `detections.csv` - Frame-by-frame detection data
- `session_summary.json` - Session metadata
- `README.txt` - Human-readable summary

## 🔧 System Status

- **Camera**: ZED Stereo Camera (Serial: 3585)
- **Resolution**: 1280x720 @ 30fps
- **Detection**: YOLOv4-tiny (80 object classes)
- **Focus**: Cars, trucks, vehicles, people
- **Storage**: `/home/nvidia/zed/recordings/`
- **Network**: Tailscale IP `100.79.213.91`

## 🛠️ Troubleshooting

**If recording buttons don't work**:
- The optimized_recorder.py server has working buttons
- The mobile_recorder.py had JavaScript issues
- Always use optimized_recorder.py

**If camera not found**:
```bash
# Check ZED camera
ls /dev/video*
# Restart if needed
sudo systemctl restart zed
```

**If Tailscale not working**:
```bash
# Check status
tailscale status
# Get IP
tailscale ip
```

## 📝 Quick Commands

```bash
# Start recorder
python3 optimized_recorder.py

# List recordings
./download_recordings.sh list

# Download latest
./download_recordings.sh download $(ls recordings/ | sort | tail -1)

# Clean old recordings (keep last 5)
./download_recordings.sh cleanup 5

# Check disk usage
./download_recordings.sh usage
```

---

✅ **System is ready for use!** Both local and remote access via Tailscale supported.