"""Prediction API Endpoints for Completed Disease Screening Models."""

from flask import Blueprint, jsonify, request
from backend.api.schemas.prediction import validate_payload_for_disease
from backend.services.prediction_service import get_prediction_service

predictions_bp = Blueprint("predictions", __name__)


def _process_prediction_request(disease_key: str):
    """Common handler for disease risk prediction endpoints."""
    # 1. Validate JSON presence
    if not request.is_json:
        return jsonify({
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Request content-type must be application/json.",
            }
        }), 400

    payload = request.get_json(silent=True)

    # 2. Validate payload is a non-empty object
    if payload is None:
        return jsonify({
            "error": {
                "code": "INVALID_INPUT",
                "message": "Malformed JSON payload in request body.",
            }
        }), 400

    if not isinstance(payload, dict):
        return jsonify({
            "error": {
                "code": "INVALID_INPUT",
                "message": "Request body must be a JSON object containing patient features.",
            }
        }), 400

    if not payload:
        return jsonify({
            "error": {
                "code": "INVALID_INPUT",
                "message": "Request body cannot be empty. Please provide required feature values.",
            }
        }), 400

    # 3. Validate against formal schema contract
    try:
        validate_payload_for_disease(disease_key, payload)
    except ValueError as err:
        return jsonify({
            "error": {
                "code": "INVALID_INPUT",
                "message": str(err),
            }
        }), 400

    # 4. Invoke centralized PredictionService
    try:
        service = get_prediction_service()
        response_data = service.predict(disease_key, payload)
        return jsonify(response_data), 200
    except ValueError as err:
        return jsonify({
            "error": {
                "code": "INVALID_INPUT",
                "message": str(err),
            }
        }), 400
    except FileNotFoundError:
        return jsonify({
            "error": {
                "code": "MODEL_UNAVAILABLE",
                "message": f"Model artifacts for '{disease_key}' are currently unavailable.",
            }
        }), 500
    except Exception:
        # Never expose internal tracebacks or system paths to clients
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during prediction processing.",
            }
        }), 500


@predictions_bp.route("/predict/lung-cancer", methods=["POST"])
def predict_lung_cancer():
    """Lung cancer risk screening prediction endpoint."""
    return _process_prediction_request("lung_cancer")


@predictions_bp.route("/predict/asthma", methods=["POST"])
def predict_asthma():
    """Asthma risk screening prediction endpoint."""
    return _process_prediction_request("asthma")


@predictions_bp.route("/predict/parkinsons", methods=["POST"])
def predict_parkinsons():
    """Parkinson's disease risk screening prediction endpoint."""
    return _process_prediction_request("parkinsons")
