"""Screening Persistence and Clinical History API Endpoints.

Provides safe REST endpoints for:
- POST /api/screenings (atomic persistence of screening, inputs, and prediction)
- GET  /api/history/user (chronological history for authenticated patient)
- GET  /api/history/clinic (chronological history for clinical provider)
- GET  /api/screenings/<id> (detailed report retrieval with strict actor ownership check)
"""

from datetime import datetime
from typing import Any, Dict, Optional
from flask import Blueprint, jsonify, request

from backend.db.database import get_db_session
from backend.db.models import ClinicUser, Disease, User
from backend.db.repositories.screenings import (
    can_access_screening,
    get_clinic_screening_history,
    get_disease_by_code,
    get_screening_by_id,
    get_user_screening_history,
    record_completed_screening,
)
from backend.db.repositories.users import (
    create_clinic_user,
    create_user,
    get_clinic_user_by_email,
    get_clinic_user_by_id,
    get_user_by_email,
    get_user_by_id,
)

screenings_bp = Blueprint("screenings", __name__)

SUPPORTED_DISEASES = {"lung_cancer", "asthma", "parkinsons"}

DISEASE_METADATA = {
    "lung_cancer": {
        "name": "Lung Cancer Risk Screening",
        "category": "Pulmonary & Oncology",
        "icon": "🫁",
    },
    "asthma": {
        "name": "Asthma Risk Screening",
        "category": "Respiratory Health",
        "icon": "💨",
    },
    "parkinsons": {
        "name": "Parkinson's Disease Risk Screening",
        "category": "Neurology",
        "icon": "🧠",
    },
}


def _resolve_user_id(session, user_id_param: Optional[str], patient_info: Optional[Dict[str, Any]]) -> str:
    """Resolve an existing database user or create a safe record if needed."""
    if user_id_param:
        existing = get_user_by_id(session, user_id_param)
        if existing:
            return existing.id

    email = None
    full_name = "Patient User"
    mobile = None
    dob = None

    if patient_info and isinstance(patient_info, dict):
        email = patient_info.get("email") or patient_info.get("patientEmail")
        full_name = patient_info.get("fullName") or patient_info.get("patientFullName") or full_name
        mobile = patient_info.get("mobile") or patient_info.get("patientMobile")
        dob_str = patient_info.get("dateOfBirth") or patient_info.get("patientDob")
        if dob_str:
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
            except Exception:
                dob = None

    if not email:
        email = f"patient_{user_id_param or 'anonymous'}@example.com"

    existing_by_email = get_user_by_email(session, email)
    if existing_by_email:
        return existing_by_email.id

    new_user = create_user(
        session=session,
        full_name=full_name,
        email=email,
        mobile=mobile,
        date_of_birth=dob,
    )
    return new_user.id


def _resolve_clinic_user_id(session, clinic_user_id_param: Optional[str]) -> Optional[str]:
    """Resolve clinic user id or return None."""
    if not clinic_user_id_param:
        return None

    existing = get_clinic_user_by_id(session, clinic_user_id_param)
    if existing:
        return existing.id

    # If clinic user doesn't exist by ID, check default demo email
    existing_demo = get_clinic_user_by_email(session, "clinic@generalhospital.org")
    if existing_demo:
        return existing_demo.id

    # Create demo clinic user in database
    new_clinic = create_clinic_user(
        session=session,
        full_name="Dr. Sarah Mitchell, MD",
        email="clinic@generalhospital.org",
        clinic_name="Metro Health Pulmonary & Neurology Clinic",
        mobile="+1-555-0344",
    )
    return new_clinic.id


@screenings_bp.route("/screenings", methods=["POST"])
def record_screening():
    """Atomically persist a completed screening, its inputs, and its prediction result."""
    if not request.is_json:
        return jsonify({
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Content-Type must be application/json.",
            }
        }), 400

    payload = request.get_json(silent=True) or {}
    disease_code = str(payload.get("disease_code", "")).strip().lower().replace("-", "_")

    if disease_code not in SUPPORTED_DISEASES:
        return jsonify({
            "error": {
                "code": "UNSUPPORTED_DISEASE",
                "message": f"Screening persistence for '{disease_code}' is unsupported. Available models: {sorted(SUPPORTED_DISEASES)}",
            }
        }), 400

    input_data = payload.get("input_data")
    if input_data is None:
        input_data = payload.get("inputs", {})
    if not isinstance(input_data, dict):
        return jsonify({
            "error": {
                "code": "INVALID_INPUT_DATA",
                "message": "Valid 'input_data' dictionary is required.",
            }
        }), 400

    prediction = payload.get("prediction") or payload.get("prediction_result")
    if not isinstance(prediction, dict) or not prediction:
        return jsonify({
            "error": {
                "code": "INVALID_PREDICTION",
                "message": "Valid non-empty 'prediction' dictionary is required.",
            }
        }), 400


    # Probability range validation
    prob = prediction.get("probability")
    if prob is None or not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
        return jsonify({
            "error": {
                "code": "INVALID_PROBABILITY",
                "message": f"Prediction probability must be a float between 0.0 and 1.0, got: {prob}",
            }
        }), 400

    user_id_param = payload.get("user_id")
    patient_info = payload.get("patient_info")
    clinic_user_id_param = payload.get("clinic_user_id")

    try:
        with get_db_session() as session:
            user_id = _resolve_user_id(session, user_id_param, patient_info)
            clinic_user_id = _resolve_clinic_user_id(session, clinic_user_id_param)

            screening = record_completed_screening(
                session=session,
                user_id=user_id,
                disease_code=disease_code,
                input_data=input_data,
                prediction_data=prediction,
                clinic_user_id=clinic_user_id,
                status="COMPLETED",
            )
            screening_id = str(screening.id)

        return jsonify({
            "id": screening_id,
            "screening_id": screening_id,
            "status": "COMPLETED",
            "message": "Screening successfully recorded.",
        }), 201


    except ValueError as err:
        return jsonify({
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(err),
            }
        }), 400
    except Exception:
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An error occurred while saving the screening record.",
            }
        }), 500


@screenings_bp.route("/history/user", methods=["GET"])
def get_user_history():
    """Retrieve chronological screening history for a specific patient."""
    user_id = request.headers.get("X-User-Id") or request.args.get("user_id")
    email = request.headers.get("X-User-Email") or request.args.get("email")

    if not user_id and not email:
        return jsonify({
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Patient identification (X-User-Id or email) is required to retrieve history.",
            }
        }), 401

    try:
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        limit, offset = 50, 0

    try:
        with get_db_session() as session:
            target_user_id = user_id
            if email and not target_user_id:
                user_rec = get_user_by_email(session, email.strip().lower())
                target_user_id = user_rec.id if user_rec else None

            if not target_user_id:
                return jsonify({"history": [], "total": 0}), 200

            screenings = get_user_screening_history(session, target_user_id, limit=limit, offset=offset)

            records = []
            for sc in screenings:
                pred = sc.prediction_result
                disease = sc.disease
                records.append({
                    "id": str(sc.id),
                    "disease_code": disease.code if disease else "unknown",
                    "disease_name": disease.name if disease else "Unknown Disease",
                    "status": sc.status,
                    "probability": pred.probability if pred else 0.0,
                    "risk_level": pred.risk_level if pred else "LOW",
                    "predicted_class": pred.predicted_class if pred else 0,
                    "model_version": pred.model_version if pred else "1.0.0",
                    "disclaimer": pred.disclaimer if pred else "",
                    "created_at": sc.created_at.isoformat() if sc.created_at else "",
                })

            return jsonify({
                "history": records,
                "items": records,
                "total": len(records),
            }), 200

    except Exception:
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Unable to load patient screening history.",
            }
        }), 500


@screenings_bp.route("/history/clinic", methods=["GET"])
def get_clinic_history():
    """Retrieve chronological screening history for an authorized clinic practitioner."""
    clinic_user_id = request.headers.get("X-Clinic-User-Id") or request.args.get("clinic_user_id")
    email = request.headers.get("X-Clinic-Email") or request.args.get("email")

    if not clinic_user_id and not email:
        return jsonify({
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Clinical provider identification (X-Clinic-User-Id or email) is required.",
            }
        }), 401

    try:
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        limit, offset = 50, 0

    try:
        with get_db_session() as session:
            target_clinic_id = clinic_user_id
            if email and not target_clinic_id:
                clinic_rec = get_clinic_user_by_email(session, email.strip().lower())
                target_clinic_id = clinic_rec.id if clinic_rec else None

            if not target_clinic_id:
                # If demo provider accessed before any record saved, check demo ID
                demo_clinic = get_clinic_user_by_email(session, "clinic@generalhospital.org")
                target_clinic_id = demo_clinic.id if demo_clinic else None

            if not target_clinic_id:
                return jsonify({"history": [], "total": 0}), 200

            screenings = get_clinic_screening_history(session, target_clinic_id, limit=limit, offset=offset)

            records = []
            for sc in screenings:
                pred = sc.prediction_result
                disease = sc.disease
                patient = sc.user
                records.append({
                    "id": str(sc.id),
                    "disease_code": disease.code if disease else "unknown",
                    "disease_name": disease.name if disease else "Unknown Disease",
                    "status": sc.status,
                    "probability": pred.probability if pred else 0.0,
                    "risk_level": pred.risk_level if pred else "LOW",
                    "predicted_class": pred.predicted_class if pred else 0,
                    "model_version": pred.model_version if pred else "1.0.0",
                    "disclaimer": pred.disclaimer if pred else "",
                    "created_at": sc.created_at.isoformat() if sc.created_at else "",
                    "patient": {
                        "id": str(patient.id) if patient else "",
                        "full_name": patient.full_name if patient else "Confidential Patient",
                        "email": patient.email if patient else "",
                    },
                })

            return jsonify({
                "history": records,
                "items": records,
                "total": len(records),
            }), 200

    except Exception:
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Unable to load clinic screening history.",
            }
        }), 500


@screenings_bp.route("/screenings/<screening_id>", methods=["GET"])
def get_screening_detail(screening_id: str):
    """Retrieve full screening report detail with strict authorization check."""
    caller_user_id = request.headers.get("X-User-Id") or request.args.get("user_id")
    caller_clinic_id = request.headers.get("X-Clinic-User-Id") or request.args.get("clinic_user_id")
    caller_email = request.headers.get("X-User-Email") or request.args.get("email")

    try:
        with get_db_session() as session:
            screening = get_screening_by_id(session, screening_id)
            if not screening:
                return jsonify({
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Screening record not found.",
                    }
                }), 404

            # Resolve email to user id if needed
            if caller_email and not caller_user_id:
                u = get_user_by_email(session, caller_email.strip().lower())
                if u:
                    caller_user_id = u.id

            # Enforce ownership
            if not can_access_screening(screening, user_id=caller_user_id, clinic_user_id=caller_clinic_id):
                return jsonify({
                    "error": {
                        "code": "FORBIDDEN",
                        "message": "You are not authorized to view this screening record.",
                    }
                }), 403

            disease = screening.disease
            pred = screening.prediction_result
            inputs = screening.screening_input
            patient = screening.user
            clinic = screening.clinic_user

            disease_meta = DISEASE_METADATA.get(disease.code, {
                "name": disease.name,
                "category": "General Clinical",
                "icon": "⚕️",
            })

            data = {
                "id": str(screening.id),
                "status": screening.status,
                "created_at": screening.created_at.isoformat() if screening.created_at else "",
                "disease": {
                    "code": disease.code,
                    "name": disease_meta["name"],
                    "category": disease_meta["category"],
                    "icon": disease_meta["icon"],
                },
                "prediction": {
                    "predicted_class": pred.predicted_class if pred else 0,
                    "probability": pred.probability if pred else 0.0,
                    "risk_level": pred.risk_level if pred else "LOW",
                    "model_version": pred.model_version if pred else "1.0.0",
                    "disclaimer": pred.disclaimer if pred else "",
                },
                "patient": {
                    "id": str(patient.id) if patient else "",
                    "full_name": patient.full_name if patient else "Confidential Patient",
                    "email": patient.email if patient else "",
                    "mobile": patient.mobile if patient else "",
                    "date_of_birth": patient.date_of_birth.isoformat() if (patient and patient.date_of_birth) else "",
                },
                "clinic_user": {
                    "id": str(clinic.id) if clinic else "",
                    "full_name": clinic.full_name if clinic else "",
                    "clinic_name": clinic.clinic_name if clinic else "",
                } if clinic else None,
                "input_data": inputs.input_data if inputs else {},
            }

            return jsonify({"screening": data, **data}), 200


    except Exception:
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while fetching the screening report.",
            }
        }), 500
