# SIH26188 - AI-Based Fake Identity & Document Screening System
# Flask application entry point

import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template

from backend.config import Config
from backend.routes.document_routes import document_bp
from database.db import init_db


def create_app(config_override=None):
    """
    Flask application factory.

    Args:
        config_override: Optional dictionary of config values to override
                         defaults (useful for testing).

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Load default configuration
    app.config.from_object(Config)

    # Apply any overrides (e.g., test configuration)
    if config_override:
        app.config.update(config_override)

    # --- Ensure upload directory exists ---
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # --- Initialize database ---
    init_db(app.config['DATABASE_PATH'])

    # --- Register blueprints ---
    app.register_blueprint(document_bp, url_prefix='/api/documents')

    # NOTE: identity_bp and audit_routes are not registered yet.
    # They will be implemented in future phases.

    @app.route('/', methods=['GET'])
    def web_interface():
        """Serve the browser interface for the existing document workflow."""
        return render_template('index.html')

    # --- Health check endpoint ---
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Return service health status."""
        return jsonify({
            "status": "running",
            "service": "SIH26188",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200

    return app


if __name__ == '__main__':
    app = create_app()
    # Run on localhost:5000 with debug mode for development
    app.run(host='127.0.0.1', port=5055, debug=True)

