# ZED Stereo Depth Flask

A real-time stereo depth visualization web application built with Flask and the Stereolabs ZED SDK. View colour and depth streams, query depth at any pixel, detect obstacles, and visualise depth histograms — all from your browser.

When no ZED camera is connected, the application runs in **simulation mode** with synthetic depth data, making it easy to develop and test on any machine.

## Features

- **Live MJPEG streaming** of colour and colourised depth feeds
- **Click-to-query depth** — click anywhere on the colour feed to read the depth in metres
- **Depth statistics** — min, max, mean, median, std, valid-pixel ratio
- **Obstacle detection grid** — configurable threshold, rows, and columns
- **Depth histogram** — distribution of depth values rendered on a canvas
- **Full REST API** for programmatic access to all depth data
- **Simulated mode** — runs without ZED hardware using synthetic stereo depth

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
python run.py
```

Open <http://localhost:5000> in your browser.

## Configuration

Set environment variables or edit `config.py`:

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | Config profile: development, testing, production |
| `ZED_RESOLUTION` | `HD720` | Camera resolution (HD2K, HD1080, HD720, VGA) |
| `ZED_FPS` | `30` | Camera frame rate |
| `ZED_DEPTH_MODE` | `PERFORMANCE` | Depth quality (PERFORMANCE, QUALITY, ULTRA, NEURAL) |
| `DEPTH_MIN_DISTANCE` | `0.3` | Minimum depth in metres |
| `DEPTH_MAX_DISTANCE` | `20.0` | Maximum depth in metres |
| `STREAM_QUALITY` | `70` | JPEG quality for streams (1-100) |
| `STREAM_WIDTH` | `640` | Stream output width |
| `STREAM_HEIGHT` | `360` | Stream output height |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/video/color` | MJPEG colour stream |
| GET | `/video/depth` | MJPEG depth-map stream |
| GET | `/api/status` | Camera status |
| GET | `/api/depth/stats` | Depth statistics |
| GET | `/api/depth/at?x=&y=` | Depth at a pixel |
| GET | `/api/depth/region?x=&y=&size=` | Average depth in a region |
| GET | `/api/depth/histogram?bins=` | Depth histogram |
| GET | `/api/depth/obstacles?threshold=&rows=&cols=` | Obstacle grid |
| GET | `/api/snapshot/color` | Single JPEG colour frame |
| GET | `/api/snapshot/depth` | Single JPEG depth frame |

## Tests

```bash
pip install pytest
pytest -v
```

## Project Structure

```
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── camera.py            # ZED camera interface + simulation fallback
│   ├── depth_processing.py  # Depth analysis utilities
│   ├── routes.py            # Flask routes and API endpoints
│   ├── static/
│   │   ├── css/style.css    # Dashboard styles
│   │   └── js/app.js        # Client-side logic
│   └── templates/
│       └── index.html       # Main viewer page
├── tests/
│   ├── test_camera.py
│   ├── test_depth_processing.py
│   └── test_routes.py
├── config.py                # Configuration classes
├── run.py                   # Application entry point
└── requirements.txt
```

## License

MIT
