import os


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = False
    TESTING = False

    # ZED Camera settings
    ZED_RESOLUTION = os.environ.get("ZED_RESOLUTION", "HD720")
    ZED_FPS = int(os.environ.get("ZED_FPS", "30"))
    ZED_DEPTH_MODE = os.environ.get("ZED_DEPTH_MODE", "PERFORMANCE")

    # Depth map settings
    DEPTH_MIN_DISTANCE = float(os.environ.get("DEPTH_MIN_DISTANCE", "0.3"))
    DEPTH_MAX_DISTANCE = float(os.environ.get("DEPTH_MAX_DISTANCE", "20.0"))
    COLORMAP = os.environ.get("COLORMAP", "INFERNO")

    # Streaming settings
    STREAM_QUALITY = int(os.environ.get("STREAM_QUALITY", "70"))
    STREAM_WIDTH = int(os.environ.get("STREAM_WIDTH", "640"))
    STREAM_HEIGHT = int(os.environ.get("STREAM_HEIGHT", "360"))

    # YOLO detection settings
    YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo11n.pt")
    YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.35"))
    YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "cpu")


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    ZED_RESOLUTION = "VGA"
    ZED_FPS = 15


class ProductionConfig(Config):
    pass


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
