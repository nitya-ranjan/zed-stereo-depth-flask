#!/bin/bash

# ZED Camera Demo Startup Script
# Run this script to start the ZED camera recording demo

echo "🎥 Starting ZED Camera Recording Demo..."
echo "======================================"

# Check if we're in the right directory
if [ ! -f "optimized_recorder.py" ]; then
    echo "❌ Error: optimized_recorder.py not found!"
    echo "Please run this script from the /home/nvidia/zed directory"
    exit 1
fi

# Check if ZED camera is connected
if [ ! -e /dev/video* ]; then
    echo "⚠️  Warning: No camera devices found at /dev/video*"
    echo "Make sure ZED camera is connected"
fi

# Stop any existing server
echo "🔄 Stopping any existing servers..."
pkill -f "optimized_recorder.py" 2>/dev/null || true
sleep 2

# Start the Flask server
echo "🚀 Starting ZED Optimized Recorder Server..."
echo ""
echo "Server starting... please wait..."

python3 optimized_recorder.py &
SERVER_PID=$!

# Wait a moment for server to initialize
sleep 5

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo ""
    echo "✅ Server started successfully!"
    echo ""
    echo "📱 Access the demo at:"
    echo "   Local:     http://localhost:5000"
    echo "   Network:   http://192.168.1.196:5000"
    echo "   Tailscale: http://100.79.213.91:5000"
    echo ""
    echo "💻 For SSH port forwarding from your laptop:"
    echo "   ssh -L 5000:localhost:5000 nvidia@100.79.213.91"
    echo "   Then open: http://localhost:5000"
    echo ""
    echo "🎮 Demo Features:"
    echo "   • Live RGB + YOLO object detection"
    echo "   • Real-time depth mapping"
    echo "   • Object distance measurements"
    echo "   • Start/Stop recording with session names"
    echo ""
    echo "⏹️  To stop the demo: Press Ctrl+C or run 'pkill -f optimized_recorder.py'"
    echo ""
    
    # Keep script running so Ctrl+C stops the server
    wait $SERVER_PID
else
    echo "❌ Failed to start server!"
    echo "Check the error messages above"
    exit 1
fi