#!/bin/bash

# ZED Recording Download Script
# Downloads recordings from the ZED camera system to your local machine

TAILSCALE_IP="100.79.213.91"
JETSON_USER="nvidia"
RECORDINGS_DIR="/home/nvidia/zed/recordings"

# Auto-detect if running locally or remotely
if [ "$(hostname)" = "orin" ] || [ -d "$RECORDINGS_DIR" ]; then
    LOCAL_MODE=true
    echo "🎥 ZED Recording Manager (Local Mode)"
else
    LOCAL_MODE=false
    echo "🎥 ZED Recording Downloader (Remote Mode)"
    echo "Tailscale IP: $TAILSCALE_IP"
fi

echo "=========================="
echo "Recordings path: $RECORDINGS_DIR"
echo ""

# Function to list available recordings
list_recordings() {
    echo "📁 Available recordings:"
    echo "----------------------"
    
    if [ "$LOCAL_MODE" = true ]; then
        # Local mode
        if [ -d "$RECORDINGS_DIR" ]; then
            cd "$RECORDINGS_DIR" && ls -la | grep '^d' | awk '{print $9}' | grep -v '^\.' | sort -r | while read session; do
                if [ ! -z "$session" ]; then
                    echo "📹 $session"
                    
                    # Show file sizes
                    cd "$RECORDINGS_DIR/$session" && ls -lh *.mp4 2>/dev/null | while read line; do
                        filename=$(echo $line | awk '{print $9}')
                        size=$(echo $line | awk '{print $5}')
                        echo "   📄 $filename ($size)"
                    done
                    echo ""
                fi
            done
        else
            echo "❌ Recordings directory not found: $RECORDINGS_DIR"
        fi
    else
        # Remote mode via SSH
        ssh $JETSON_USER@$TAILSCALE_IP "cd $RECORDINGS_DIR && ls -la | grep '^d' | awk '{print \$9}' | grep -v '^\.' | sort -r" | while read session; do
            if [ ! -z "$session" ]; then
                echo "📹 $session"
                
                # Show file sizes
                ssh $JETSON_USER@$TAILSCALE_IP "cd $RECORDINGS_DIR/$session && ls -lh *.mp4 2>/dev/null" | while read line; do
                    filename=$(echo $line | awk '{print $9}')
                    size=$(echo $line | awk '{print $5}')
                    echo "   📄 $filename ($size)"
                done
                echo ""
            fi
        done
    fi
}

# Function to download/copy a specific recording
download_recording() {
    local session_name=$1
    local target_dir=${2:-"./downloads"}
    
    if [ "$LOCAL_MODE" = true ]; then
        echo "📁 Copying recording: $session_name"
        echo "Target directory: $target_dir"
        
        # Create target directory
        mkdir -p "$target_dir/$session_name"
        
        # Copy entire session directory
        cp -r "$RECORDINGS_DIR/$session_name/"* "$target_dir/$session_name/"
    else
        echo "⬇️  Downloading recording: $session_name"
        echo "Target directory: $target_dir"
        
        # Create target directory
        mkdir -p "$target_dir/$session_name"
        
        # Download entire session directory via rsync
        rsync -avz --progress "$JETSON_USER@$TAILSCALE_IP:$RECORDINGS_DIR/$session_name/" "$target_dir/$session_name/"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Download completed successfully!"
        echo "📁 Files saved to: $target_dir/$session_name/"
        
        # Show what was downloaded
        echo ""
        echo "📦 Downloaded files:"
        ls -lh "$target_dir/$session_name/"
        
        # Show summary if available
        if [ -f "$target_dir/$session_name/session_summary.json" ]; then
            echo ""
            echo "📊 Session Summary:"
            cat "$target_dir/$session_name/session_summary.json" | python3 -m json.tool 2>/dev/null || cat "$target_dir/$session_name/session_summary.json"
        fi
    else
        echo "❌ Download failed!"
    fi
}

# Function to download/copy all recordings
download_all() {
    local target_dir=${1:-"./downloads"}
    
    if [ "$LOCAL_MODE" = true ]; then
        echo "📁 Copying all recordings..."
        echo "Target directory: $target_dir"
        
        mkdir -p "$target_dir"
        
        # Copy entire recordings directory
        cp -r "$RECORDINGS_DIR/"* "$target_dir/"
    else
        echo "⬇️  Downloading all recordings..."
        echo "Target directory: $target_dir"
        
        mkdir -p "$target_dir"
        
        # Download entire recordings directory via rsync
        rsync -avz --progress "$JETSON_USER@$TAILSCALE_IP:$RECORDINGS_DIR/" "$target_dir/"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ All recordings downloaded successfully!"
        echo "📁 Files saved to: $target_dir/"
        
        # Show summary
        echo ""
        echo "📦 Downloaded sessions:"
        ls -la "$target_dir/" | grep '^d' | awk '{print $9}' | grep -v '^\.'
    else
        echo "❌ Download failed!"
    fi
}

# Function to clean old recordings (keep last N)
cleanup_remote() {
    local keep_count=${1:-5}
    
    echo "🧹 Cleaning up recordings (keeping last $keep_count)..."
    
    if [ "$LOCAL_MODE" = true ]; then
        cd "$RECORDINGS_DIR" && ls -t | tail -n +$((keep_count + 1)) | xargs -r rm -rf
    else
        ssh $JETSON_USER@$TAILSCALE_IP "cd $RECORDINGS_DIR && ls -t | tail -n +$((keep_count + 1)) | xargs -r rm -rf"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Cleanup completed!"
        echo "📁 Recordings remaining:"
        list_recordings
    else
        echo "❌ Cleanup failed!"
    fi
}

# Function to get disk usage
disk_usage() {
    echo "💾 Disk usage:"
    if [ "$LOCAL_MODE" = true ]; then
        echo "Recordings directory:"
        du -sh "$RECORDINGS_DIR"
    else
        echo "Remote recordings directory:"
        ssh $JETSON_USER@$TAILSCALE_IP "du -sh $RECORDINGS_DIR"
    fi
    
    echo "Local downloads directory:"
    if [ -d "./downloads" ]; then
        du -sh ./downloads
    else
        echo "No local downloads directory found"
    fi
}

# Main script logic
case "$1" in
    "list"|"ls")
        list_recordings
        ;;
    "download"|"get")
        if [ -z "$2" ]; then
            echo "Usage: $0 download <session_name> [target_directory]"
            echo ""
            list_recordings
        else
            download_recording "$2" "$3"
        fi
        ;;
    "all")
        download_all "$2"
        ;;
    "cleanup")
        cleanup_remote "$2"
        ;;
    "usage"|"du")
        disk_usage
        ;;
    "help"|"-h"|"--help")
        echo "ZED Recording Download Script"
        echo ""
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  list, ls              List available recordings"
        echo "  download <session>    Download specific recording session"
        echo "  all [target_dir]      Download all recordings"
        echo "  cleanup [keep_count]  Remove old recordings (keep last N, default 5)"
        echo "  usage, du             Show disk usage"
        echo "  help                  Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 list                                    # List all recordings"
        echo "  $0 download try2_20250907_235816          # Download specific session"
        echo "  $0 download try2_20250907_235816 ./videos # Download to specific directory"
        echo "  $0 all                                     # Download all recordings"
        echo "  $0 cleanup 3                              # Keep only last 3 recordings"
        echo ""
        ;;
    *)
        echo "🎥 ZED Recording Downloader"
        echo ""
        echo "Quick commands:"
        echo "  $0 list      - List available recordings"
        echo "  $0 all       - Download all recordings"
        echo "  $0 help      - Show full help"
        echo ""
        list_recordings
        ;;
esac