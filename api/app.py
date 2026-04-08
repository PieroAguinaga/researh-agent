"""
api/app.py

Flask application factory.
Call create_app() to get a fully configured Flask instance.
"""

import logging
from flask import Flask
from flask_cors import CORS

from config.settings import settings


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask app with all blueprints registered.
    """
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    app.secret_key = settings.flask_secret_key

    # Allow cross-origin requests to /api/* (useful during development)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )


    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/")
    def index():
        from flask import render_template
        return render_template("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=settings.flask_debug,
    )