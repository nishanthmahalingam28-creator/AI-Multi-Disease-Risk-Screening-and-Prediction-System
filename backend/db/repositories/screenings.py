"""Disease and Screening Session Data Access Repositories."""

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from backend.db.models import Disease, PredictionResult, Screening, ScreeningInput

# Reference seed catalog for all 10 planned multi-disease categories
DISEASE_SEEDS: List[Dict[str, str]] = [
    {
        "code": "breast_cancer",
        "name": "Breast Cancer",
        "description": "Biomarker risk screening based on tissue biopsy cell nuclei features.",
    },
    {
        "code": "diabetes",
        "name": "Diabetes",
        "description": "Type 2 diabetes risk screening based on metabolic, glucose, and biometric indices.",
    },
    {
        "code": "heart_disease",
        "name": "Heart Disease",
        "description": "Cardiovascular risk screening based on clinical vitals, cholesterol, and cardiac metrics.",
    },
    {
        "code": "stroke",
        "name": "Stroke",
        "description": "Cerebrovascular risk assessment based on physiological factors, hypertension, and medical history.",
    },
    {
        "code": "kidney",
        "name": "Chronic Kidney Disease",
        "description": "Renal function screening assessing filtration, electrolytes, and urinalysis profiles.",
    },
    {
        "code": "liver",
        "name": "Liver Disease",
        "description": "Hepatic risk assessment utilizing liver enzyme ratios and metabolic indicators.",
    },
    {
        "code": "thyroid",
        "name": "Thyroid Disease",
        "description": "Endocrine screening for hypothyroidism and thyroid dysfunction risk.",
    },
    {
        "code": "lung_cancer",
        "name": "Lung Cancer",
        "description": "Pulmonary risk screening analyzing demographic risks, respiratory symptoms, and exposures.",
    },
    {
        "code": "asthma",
        "name": "Asthma",
        "description": "Airway disease screening based on inflammatory biomarkers, spirometry, and clinical triggers.",
    },
    {
        "code": "parkinsons",
        "name": "Parkinson's Disease",
        "description": "Acoustic dysphonia analysis of sustained phonation to detect neurodegenerative vocal tremors.",
    },
]


def seed_diseases(session: Session) -> List[Disease]:
    """Seed the reference disease catalog with all 10 planned diseases if not present."""
    seeded = []
    for item in DISEASE_SEEDS:
        existing = session.query(Disease).filter(Disease.code == item["code"]).first()
        if not existing:
            new_disease = Disease(
                code=item["code"],
                name=item["name"],
                description=item["description"],
                is_active=True,
            )
            session.add(new_disease)
            seeded.append(new_disease)
        else:
            seeded.append(existing)
    session.flush()
    return seeded


def get_disease_by_code(session: Session, code: str) -> Optional[Disease]:
    """Retrieve disease record by canonical code."""
    clean_code = str(code).strip().lower().replace("-", "_")
    return session.query(Disease).filter(Disease.code == clean_code).first()


def list_diseases(session: Session, active_only: bool = True) -> List[Disease]:
    """List all disease reference records."""
    query = session.query(Disease)
    if active_only:
        query = query.filter(Disease.is_active.is_(True))
    return query.order_by(Disease.id).all()


def create_screening(
    session: Session,
    user_id: str,
    disease_id: int,
    clinic_user_id: Optional[str] = None,
    status: str = "COMPLETED",
) -> Screening:
    """Create a new screening record for a patient (self-screening or clinic-assisted)."""
    screening = Screening(
        user_id=str(user_id),
        disease_id=int(disease_id),
        clinic_user_id=str(clinic_user_id) if clinic_user_id else None,
        status=status.strip().upper(),
    )
    session.add(screening)
    session.flush()
    return screening


def get_screening_by_id(session: Session, screening_id: str) -> Optional[Screening]:
    """Retrieve screening session by UUID primary key."""
    return session.query(Screening).filter(Screening.id == str(screening_id)).first()


def save_screening_input(
    session: Session,
    screening_id: str,
    input_data: Dict[str, Any],
) -> ScreeningInput:
    """Persist submitted patient feature payload associated with a screening."""
    screening_input = ScreeningInput(
        screening_id=str(screening_id),
        input_data=input_data,
    )
    session.add(screening_input)
    session.flush()
    return screening_input


def save_prediction_result(
    session: Session,
    screening_id: str,
    predicted_class: int,
    probability: float,
    risk_level: str,
    disclaimer: str,
    model_version: str = "1.0.0",
) -> PredictionResult:
    """Persist prediction output associated with a screening session."""
    prediction_result = PredictionResult(
        screening_id=str(screening_id),
        predicted_class=int(predicted_class),
        probability=float(probability),
        risk_level=risk_level.strip(),
        disclaimer=disclaimer.strip(),
        model_version=model_version.strip(),
    )
    session.add(prediction_result)
    session.flush()
    return prediction_result


def get_user_screening_history(
    session: Session,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[Screening]:
    """Retrieve chronological screening history for a specific patient."""
    query = (
        session.query(Screening)
        .filter(Screening.user_id == str(user_id))
        .order_by(Screening.created_at.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)
    return query.all()


def get_clinic_screening_history(
    session: Session,
    clinic_user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[Screening]:
    """Retrieve chronological screening history conducted by a specific clinical practitioner."""
    query = (
        session.query(Screening)
        .filter(Screening.clinic_user_id == str(clinic_user_id))
        .order_by(Screening.created_at.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)
    return query.all()


def record_completed_screening(
    session: Session,
    user_id: Optional[str] = None,
    disease_code: str = "",
    input_data: Optional[Dict[str, Any]] = None,
    prediction_data: Optional[Dict[str, Any]] = None,
    clinic_user_id: Optional[str] = None,
    status: str = "COMPLETED",
    prediction_result: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Screening:
    """Atomically record a completed screening session, its input data, and prediction result.

    Executes within the caller's session transaction. If any component fails
    (e.g., probability out of bounds, missing disease), an exception is raised
    triggering clean rollback.

    Args:
        session: Active SQLAlchemy database session.
        user_id: Identifier of the patient.
        disease_code: Canonical code of the screened disease (e.g. 'lung_cancer').
        input_data: Raw patient clinical feature inputs.
        prediction_data: Model output dictionary containing predicted_class, probability,
                         risk_level, model_version, and disclaimer.
        clinic_user_id: Optional identifier of the assisting clinical provider.
        status: Screening lifecycle status (default 'COMPLETED').

    Returns:
        The newly persisted Screening instance with relationships.
    """
    # Normalize argument aliases
    user_id = user_id or kwargs.get("user_id")
    disease_code = disease_code or kwargs.get("disease_code", "")
    input_data = input_data if input_data is not None else kwargs.get("input_data", {})
    prediction_data = prediction_data or prediction_result or kwargs.get("prediction_result") or {}
    clinic_user_id = clinic_user_id or kwargs.get("clinic_user_id")

    # 1. Resolve and validate disease
    disease = get_disease_by_code(session, disease_code)
    if not disease:
        # If disease table wasn't seeded yet, seed it
        seed_diseases(session)
        disease = get_disease_by_code(session, disease_code)
        if not disease:
            raise ValueError(f"Unknown disease code: '{disease_code}'")

    # 2. Validate prediction probability bounds
    prob = prediction_data.get("probability")
    if prob is None or not isinstance(prob, (int, float)):
        raise ValueError("Prediction data must contain numerical 'probability'")
    prob_float = float(prob)
    if prob_float < 0.0 or prob_float > 1.0:
        raise ValueError(f"Prediction probability must be between 0.0 and 1.0, got {prob_float}")

    pred_class = prediction_data.get("predicted_class")
    if pred_class is None:
        pred_class = prediction_data.get("class")
    if pred_class is None:
        pred_class = 1 if prob_float >= 0.5 else 0


    risk_level = str(prediction_data.get("risk_level", "LOW")).strip()
    disclaimer = str(prediction_data.get("disclaimer", "")).strip()
    model_version = str(prediction_data.get("model_version", "1.0.0")).strip()

    if not disclaimer:
        disclaimer = (
            "This result is an AI-based screening/risk estimate and is not a medical diagnosis. "
            "Please consult a qualified healthcare professional."
        )

    try:
        # 3. Create parent screening entity
        screening = create_screening(
            session=session,
            user_id=user_id,
            disease_id=disease.id,
            clinic_user_id=clinic_user_id,
            status=status,
        )

        # 4. Save input payload
        save_screening_input(
            session=session,
            screening_id=screening.id,
            input_data=input_data,
        )

        # 5. Save prediction result (enforces check constraint)
        save_prediction_result(
            session=session,
            screening_id=screening.id,
            predicted_class=int(pred_class),
            probability=prob_float,
            risk_level=risk_level,
            disclaimer=disclaimer,
            model_version=model_version,
        )

        session.flush()
        return screening
    except Exception:
        session.rollback()
        raise


def can_access_screening(
    *args,
    user_id: Optional[str] = None,
    clinic_user_id: Optional[str] = None,
    **kwargs,
) -> bool:
    """Verify if the requesting actor is authorized to view the screening record.

    Access is granted if:
    - Actor is the patient (user_id matches screening.user_id)
    - Actor is the assisting clinic user (clinic_user_id matches screening.clinic_user_id)
    """
    if len(args) >= 2 and isinstance(args[1], Screening):
        screening = args[1]
    elif len(args) >= 1 and isinstance(args[0], Screening):
        screening = args[0]
    elif "screening" in kwargs:
        screening = kwargs["screening"]
    else:
        return False

    uid = user_id or kwargs.get("actor_user_id")
    cid = clinic_user_id or kwargs.get("actor_clinic_user_id")
    if uid and screening.user_id and str(screening.user_id) == str(uid):
        return True
    if cid and screening.clinic_user_id and str(screening.clinic_user_id) == str(cid):
        return True
    return False


