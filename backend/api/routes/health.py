"""API Health and Model Availability Routes."""

from flask import Blueprint, jsonify
from backend.services.prediction_service import get_prediction_service

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def api_health():
    """Liveness health check endpoint.

    Does not require database access or authentication.
    """
    return jsonify({
        "status": "ok",
        "service": "ai-multi-disease-risk-screening-api",
    }), 200


@health_bp.route("/models", methods=["GET"])
def list_available_models():
    """Model availability and readiness verification endpoint.

    Verifies artifact loadability for completed disease models.
    """
    service = get_prediction_service()
    available_models = service.get_available_models()
    return jsonify({
        "models": available_models,
    }), 200
