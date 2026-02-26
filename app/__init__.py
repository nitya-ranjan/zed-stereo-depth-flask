from flask import Flask

from config import config_by_name


def create_app(config_name="development"):
    """Application factory for the Flask app."""
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    from app.routes import main_bp, api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
