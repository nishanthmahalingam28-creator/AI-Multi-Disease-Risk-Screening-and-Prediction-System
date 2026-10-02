"""Flask Application Factory for AI Multi-Disease Risk Screening System."""

import os
from typing import Optional
from flask import Flask, jsonify
from flask_cors import CORS

from backend.api.routes.health import health_bp
from backend.api.routes.predictions import predictions_bp
from backend.api.routes.screenings import screenings_bp


def create_app(config: Optional[dict] = None) -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)

    # Default configuration
    app.config.update(
        JSON_SORT_KEYS=False,
        TESTING=False,
    )

    if config:
        app.config.update(config)

    # Configure CORS narrowly for the API routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints under the /api prefix
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(predictions_bp, url_prefix="/api")
    app.register_blueprint(screenings_bp, url_prefix="/api")

    # Global HTTP error handlers returning structured JSON
    @app.errorhandler(404)
    def handle_not_found(err):
        return jsonify({
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested endpoint or resource was not found.",
            }
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(err):
        return jsonify({
            "error": {
                "code": "METHOD_NOT_ALLOWED",
                "message": "HTTP method not allowed for this endpoint.",
            }
        }), 405

    @app.errorhandler(500)
    def handle_internal_error(err):
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server-side error occurred.",
            }
        }), 500

    return app


# Module-level application instance
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
