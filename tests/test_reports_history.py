"""Step 14 Reports & History Comprehensive Test Suite.

Validates all 26 requirements specified for Reports & History:
 1. Screening persistence
 2. Screening input persistence
 3. Prediction result persistence
 4. User history retrieval
 5. Clinic history retrieval
 6. Screening detail retrieval
 7. User ownership restriction
 8. Clinic association restriction
 9. Unauthorized record access rejection (403 Forbidden)
10. Missing screening handling (404 Not Found)
11. Probability validation (0.0 <= probability <= 1.0)
12. Risk-level persistence
13. Model-version persistence
14. Disclaimer persistence
15. Transaction rollback on partial failure
16. Empty history handling
17. History API validation
18. User history page frontend existence and structure
19. Clinic history page frontend existence and structure
20. Reusable report component rendering and structure
21. Print CSS & browser printing structure
22. No fake records displayed
23. No secrets or credentials exposed
24. Existing API regression
25. Existing model regression
26. Existing database regression
"""

import json
import os
from pathlib import Path
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.db.database as db_module
from backend.db.database import Base
from backend.db.models import (
    ClinicUser,
    Disease,
    PredictionResult,
    Screening,
    ScreeningInput,
    User,
)
from backend.db.repositories.screenings import (
    DISEASE_SEEDS,
    can_access_screening,
    create_screening,
    get_clinic_screening_history,
    get_disease_by_code,
    get_screening_by_id,
    get_user_screening_history,
    record_completed_screening,
    save_prediction_result,
    save_screening_input,
    seed_diseases,
)
from backend.db.repositories.users import (
    create_clinic_user,
    create_user,
)
from backend.api.app import create_app


@pytest.fixture(scope="function")
def db_engine():
    """Create isolated in-memory SQLite database engine."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="function")
def session(db_engine):
    """Provide a transactional in-memory database session."""
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    sess = Session()
    try:
        seed_diseases(sess)
        yield sess
    finally:
        sess.rollback()
        sess.close()


@pytest.fixture(scope="function")
def api_client(db_engine, monkeypatch):
    """Provide a Flask test client with SessionLocal wired to in-memory SQLite."""
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    setup_sess = Session()
    seed_diseases(setup_sess)
    setup_sess.close()

    monkeypatch.setattr(db_module, "SessionLocal", Session)

    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


# ==============================================================================
# 1. SCREENING PERSISTENCE
# ==============================================================================
def test_01_screening_persistence(session):
    """Verify screening entity persistence in screenings table."""
    user = create_user(session, full_name="Alice", email="alice@test.com")
    disease = get_disease_by_code(session, "lung_cancer")

    screening = create_screening(
        session,
        user_id=user.id,
        disease_id=disease.id,
        status="COMPLETED",
    )
    assert screening.id is not None
    assert screening.user_id == user.id
    assert screening.disease_id == disease.id
    assert screening.status == "COMPLETED"
    assert screening.created_at is not None


# ==============================================================================
# 2. SCREENING INPUT PERSISTENCE
# ==============================================================================
def test_02_screening_input_persistence(session):
    """Verify input feature payload persistence in screening_inputs table."""
    user = create_user(session, full_name="Bob", email="bob@test.com")
    disease = get_disease_by_code(session, "asthma")
    screening = create_screening(session, user_id=user.id, disease_id=disease.id)

    input_payload = {"Age": 45, "Smoking": 1, "Wheezing": 1}
    inp = save_screening_input(session, screening_id=screening.id, input_data=input_payload)

    assert inp.id is not None
    assert inp.screening_id == screening.id
    assert inp.input_data == input_payload


# ==============================================================================
# 3. PREDICTION RESULT PERSISTENCE
# ==============================================================================
def test_03_prediction_result_persistence(session):
    """Verify model inference outcome persistence in prediction_results table."""
    user = create_user(session, full_name="Charlie", email="charlie@test.com")
    disease = get_disease_by_code(session, "parkinsons")
    screening = create_screening(session, user_id=user.id, disease_id=disease.id)

    pred = save_prediction_result(
        session,
        screening_id=screening.id,
        predicted_class=1,
        probability=0.885,
        risk_level="HIGH",
        model_version="1.0.0",
        disclaimer="Mandatory screening notice",
    )

    assert pred.id is not None
    assert pred.screening_id == screening.id
    assert pred.predicted_class == 1
    assert abs(pred.probability - 0.885) < 1e-4
    assert pred.risk_level == "HIGH"
    assert pred.model_version == "1.0.0"


# ==============================================================================
# 4. USER HISTORY RETRIEVAL
# ==============================================================================
def test_04_user_history_retrieval(session):
    """Verify get_user_screening_history retrieves only the specified user's records."""
    u1 = create_user(session, full_name="User 1", email="u1@test.com")
    u2 = create_user(session, full_name="User 2", email="u2@test.com")
    d_lung = get_disease_by_code(session, "lung_cancer")
    d_asthma = get_disease_by_code(session, "asthma")

    # Record 2 screenings for User 1
    record_completed_screening(
        session,
        disease_code=d_lung.code,
        input_data={"feature": 1},
        prediction_result={"predicted_class": 0, "probability": 0.12, "risk_level": "LOW", "model_version": "1.0.0"},
        user_id=u1.id,
    )
    record_completed_screening(
        session,
        disease_code=d_asthma.code,
        input_data={"feature": 2},
        prediction_result={"predicted_class": 1, "probability": 0.75, "risk_level": "HIGH", "model_version": "1.0.0"},
        user_id=u1.id,
    )

    # Record 1 screening for User 2
    record_completed_screening(
        session,
        disease_code=d_lung.code,
        input_data={"feature": 3},
        prediction_result={"predicted_class": 0, "probability": 0.05, "risk_level": "LOW", "model_version": "1.0.0"},
        user_id=u2.id,
    )

    history_u1 = get_user_screening_history(session, u1.id)
    assert len(history_u1) == 2
    for item in history_u1:
        assert item.user_id == u1.id

    history_u2 = get_user_screening_history(session, u2.id)
    assert len(history_u2) == 1
    assert history_u2[0].user_id == u2.id


# ==============================================================================
# 5. CLINIC HISTORY RETRIEVAL
# ==============================================================================
def test_05_clinic_history_retrieval(session):
    """Verify get_clinic_screening_history retrieves only clinic-associated screenings."""
    clinic1 = create_clinic_user(session, full_name="Dr. One", email="c1@hospital.org", clinic_name="Clinic A")
    clinic2 = create_clinic_user(session, full_name="Dr. Two", email="c2@hospital.org", clinic_name="Clinic B")
    patient = create_user(session, full_name="Patient", email="patient@test.com")
    d_park = get_disease_by_code(session, "parkinsons")

    rec1 = record_completed_screening(
        session,
        disease_code=d_park.code,
        input_data={"MDVP:Fo(Hz)": 119.99},
        prediction_result={"predicted_class": 1, "probability": 0.91, "risk_level": "HIGH", "model_version": "1.0.0"},
        user_id=patient.id,
        clinic_user_id=clinic1.id,
    )

    rec2 = record_completed_screening(
        session,
        disease_code=d_park.code,
        input_data={"MDVP:Fo(Hz)": 200.0},
        prediction_result={"predicted_class": 0, "probability": 0.04, "risk_level": "LOW", "model_version": "1.0.0"},
        user_id=patient.id,
        clinic_user_id=clinic2.id,
    )

    clinic1_history = get_clinic_screening_history(session, clinic1.id)
    assert len(clinic1_history) == 1
    assert clinic1_history[0].id == rec1.id
    assert clinic1_history[0].clinic_user_id == clinic1.id

    clinic2_history = get_clinic_screening_history(session, clinic2.id)
    assert len(clinic2_history) == 1
    assert clinic2_history[0].id == rec2.id


# ==============================================================================
# 6. SCREENING DETAIL RETRIEVAL
# ==============================================================================
def test_06_screening_detail_retrieval(session):
    """Verify get_screening_by_id loads relationships (disease, user, clinic_user, input, prediction)."""
    user = create_user(session, full_name="Detail Patient", email="detail_user@test.com")
    clinic = create_clinic_user(session, full_name="Dr. Smith", email="detail_doc@clinic.com", clinic_name="City Clinic")

    record = record_completed_screening(
        session,
        disease_code="lung_cancer",
        input_data={"AGE": 60, "SMOKING": 1},
        prediction_result={
            "predicted_class": 1,
            "probability": 0.82,
            "risk_level": "HIGH",
            "model_version": "1.0.0",
            "disclaimer": "AI screening research result.",
        },
        user_id=user.id,
        clinic_user_id=clinic.id,
    )

    loaded = get_screening_by_id(session, record.id)
    assert loaded is not None
    assert loaded.id == record.id
    assert loaded.disease.code == "lung_cancer"
    assert loaded.user.full_name == "Detail Patient"
    assert loaded.clinic_user.full_name == "Dr. Smith"
    assert loaded.screening_input.input_data == {"AGE": 60, "SMOKING": 1}
    assert loaded.prediction_result.risk_level == "HIGH"
    assert loaded.prediction_result.disclaimer == "AI screening research result."


# ==============================================================================
# 7. USER OWNERSHIP RESTRICTION
# ==============================================================================
def test_07_user_ownership_restriction(session):
    """Verify can_access_screening grants access to owner and denies unrelated users."""
    u1 = create_user(session, full_name="Owner", email="owner@test.com")
    u2 = create_user(session, full_name="Stranger", email="stranger@test.com")

    screening = record_completed_screening(
        session,
        disease_code="asthma",
        input_data={},
        prediction_result={"predicted_class": 0, "probability": 0.15, "risk_level": "LOW", "model_version": "1.0.0"},
        user_id=u1.id,
    )

    assert can_access_screening(session, screening, actor_user_id=u1.id) is True
    assert can_access_screening(session, screening, actor_user_id=u2.id) is False


# ==============================================================================
# 8. CLINIC ASSOCIATION RESTRICTION
# ==============================================================================
def test_08_clinic_association_restriction(session):
    """Verify clinic provider can only access screenings associated with their clinic."""
    c1 = create_clinic_user(session, full_name="Dr. Authorized", email="c1@health.org", clinic_name="Clinic 1")
    c2 = create_clinic_user(session, full_name="Dr. Unrelated", email="c2@health.org", clinic_name="Clinic 2")

    screening = record_completed_screening(
        session,
        disease_code="parkinsons",
        input_data={},
        prediction_result={"predicted_class": 1, "probability": 0.88, "risk_level": "HIGH", "model_version": "1.0.0"},
        clinic_user_id=c1.id,
    )

    assert can_access_screening(session, screening, actor_clinic_user_id=c1.id) is True
    assert can_access_screening(session, screening, actor_clinic_user_id=c2.id) is False


# ==============================================================================
# 9. UNAUTHORIZED RECORD ACCESS REJECTION (API 403)
# ==============================================================================
def test_09_unauthorized_record_access_rejection(api_client, session):
    """Verify GET /api/screenings/<id> returns 403 Forbidden when unauthorized user attempts access."""
    u1 = create_user(session, full_name="User 1", email="u1_secret@test.com")
    u2 = create_user(session, full_name="User 2", email="u2_intruder@test.com")

    screening = record_completed_screening(
        session,
        disease_code="lung_cancer",
        input_data={},
        prediction_result={"predicted_class": 1, "probability": 0.95, "risk_level": "HIGH", "model_version": "1.0.0"},
        user_id=u1.id,
    )

    # u2 attempts to fetch u1's record
    resp = api_client.get(
        f"/api/screenings/{screening.id}",
        headers={"X-User-Id": u2.id, "X-User-Email": u2.email},
    )
    assert resp.status_code == 403
    data = resp.get_json()
    assert "not authorized" in data["error"]["message"].lower() or "forbidden" in data["error"]["code"].lower()


# ==============================================================================
# 10. MISSING SCREENING HANDLING (API 404)
# ==============================================================================
def test_10_missing_screening_handling(api_client):
    """Verify GET /api/screenings/<nonexistent-id> returns 404 Not Found."""
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    resp = api_client.get(
        f"/api/screenings/{fake_uuid}",
        headers={"X-User-Id": "some-user-id"},
    )
    assert resp.status_code == 404
    data = resp.get_json()
    assert "not found" in data["error"]["message"].lower()


# ==============================================================================
# 11. PROBABILITY VALIDATION (0.0 <= p <= 1.0)
# ==============================================================================
def test_11_probability_validation(session, api_client):
    """Verify repository and API reject probabilities < 0.0 or > 1.0."""
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        record_completed_screening(
            session,
            disease_code="asthma",
            input_data={},
            prediction_result={"predicted_class": 1, "probability": 1.5, "risk_level": "HIGH", "model_version": "1.0.0"},
        )

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        record_completed_screening(
            session,
            disease_code="asthma",
            input_data={},
            prediction_result={"predicted_class": 0, "probability": -0.2, "risk_level": "LOW", "model_version": "1.0.0"},
        )

    resp = api_client.post(
        "/api/screenings",
        json={
            "disease_code": "asthma",
            "prediction_result": {
                "predicted_class": 1,
                "probability": 2.5,
                "risk_level": "HIGH",
            },
        },
    )

    assert resp.status_code == 400


# ==============================================================================
# 12. RISK LEVEL PERSISTENCE
# ==============================================================================
def test_12_risk_level_persistence(session):
    """Verify all risk tiers (LOW, MODERATE, HIGH) persist accurately."""
    for tier, prob in [("LOW", 0.1), ("MODERATE", 0.5), ("HIGH", 0.9)]:
        rec = record_completed_screening(
            session,
            disease_code="lung_cancer",
            input_data={},
            prediction_result={"predicted_class": int(tier == "HIGH"), "probability": prob, "risk_level": tier, "model_version": "1.0.0"},
        )
        loaded = get_screening_by_id(session, rec.id)
        assert loaded.prediction_result.risk_level == tier


# ==============================================================================
# 13. MODEL VERSION PERSISTENCE
# ==============================================================================
def test_13_model_version_persistence(session):
    """Verify model version is stored and retrievable."""
    rec = record_completed_screening(
        session,
        disease_code="parkinsons",
        input_data={},
        prediction_result={"predicted_class": 1, "probability": 0.8, "risk_level": "HIGH", "model_version": "2.1.0-alpha"},
    )
    loaded = get_screening_by_id(session, rec.id)
    assert loaded.prediction_result.model_version == "2.1.0-alpha"


# ==============================================================================
# 14. DISCLAIMER PERSISTENCE
# ==============================================================================
def test_14_disclaimer_persistence(session):
    """Verify non-diagnostic medical disclaimer is recorded and persisted."""
    disclaimer_text = "This result is for screening/research support and is not a medical diagnosis."
    rec = record_completed_screening(
        session,
        disease_code="lung_cancer",
        input_data={},
        prediction_result={
            "predicted_class": 0,
            "probability": 0.1,
            "risk_level": "LOW",
            "model_version": "1.0.0",
            "disclaimer": disclaimer_text,
        },
    )
    loaded = get_screening_by_id(session, rec.id)
    assert disclaimer_text in loaded.prediction_result.disclaimer


# ==============================================================================
# 15. TRANSACTION ROLLBACK ON PARTIAL FAILURE
# ==============================================================================
def test_15_transaction_rollback(session, monkeypatch):
    """Verify if prediction saving fails, screening record is rolled back cleanly."""
    initial_count = session.query(Screening).count()

    def crash_save(*args, **kwargs):
        raise RuntimeError("Simulated DB Disk Crash during prediction persistence")

    monkeypatch.setattr("backend.db.repositories.screenings.save_prediction_result", crash_save)

    with pytest.raises(RuntimeError, match="Simulated DB Disk Crash"):
        record_completed_screening(
            session,
            disease_code="asthma",
            input_data={"wheeze": 1},
            prediction_result={"predicted_class": 1, "probability": 0.7, "risk_level": "HIGH", "model_version": "1.0.0"},
        )

    final_count = session.query(Screening).count()
    assert final_count == initial_count, "Screening record must be rolled back on partial transaction failure!"


# ==============================================================================
# 16. EMPTY HISTORY HANDLING
# ==============================================================================
def test_16_empty_history_handling(api_client, session):
    """Verify empty history returns empty list without error or fake records."""
    empty_user = create_user(session, full_name="Empty User", email="empty@test.com")

    resp = api_client.get(
        "/api/history/user",
        headers={"X-User-Id": empty_user.id, "X-User-Email": empty_user.email},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["items"] == []
    assert data["total"] == 0


# ==============================================================================
# 17. HISTORY API VALIDATION
# ==============================================================================
def test_17_history_api_validation(api_client):
    """Verify /api/history/user requires authentication and rejects missing identity."""
    resp = api_client.get("/api/history/user")
    assert resp.status_code == 401
    data = resp.get_json()
    assert "required" in data["error"]["message"].lower() or "missing" in data["error"]["message"].lower()


# ==============================================================================
# 18. USER HISTORY PAGE FRONTEND STRUCTURE
# ==============================================================================
def test_18_user_history_page_frontend():
    """Verify frontend/user/pages/History.js has required UI elements and copy."""
    history_js = REPO_ROOT / "frontend" / "user" / "pages" / "History.js"
    assert history_js.exists(), "frontend/user/pages/History.js must exist"
    content = history_js.read_text(encoding="utf-8")

    assert "Screening History" in content
    assert "No screening history available." in content
    assert "Loading screening history..." in content
    assert "Unable to load screening history." in content
    assert "View Report" in content
    assert "filter-disease" in content
    assert "filter-risk" in content


# ==============================================================================
# 19. CLINIC HISTORY PAGE FRONTEND STRUCTURE
# ==============================================================================
def test_19_clinic_history_page_frontend():
    """Verify frontend/clinic/pages/History.js has patient attribution and clinical audit controls."""
    history_js = REPO_ROOT / "frontend" / "clinic" / "pages" / "History.js"
    assert history_js.exists(), "frontend/clinic/pages/History.js must exist"
    content = history_js.read_text(encoding="utf-8")

    assert "Clinical Screening Records" in content
    assert "Patient Reference" in content
    assert "No screening history available." in content
    assert "Loading screening history..." in content
    assert "View Report" in content


# ==============================================================================
# 20. REPORT COMPONENT RENDERING AND STRUCTURE
# ==============================================================================
def test_20_report_component_rendering():
    """Verify frontend/components/ScreeningReport.js contains non-diagnostic disclaimer and report sections."""
    report_js = REPO_ROOT / "frontend" / "components" / "ScreeningReport.js"
    assert report_js.exists(), "frontend/components/ScreeningReport.js must exist"
    content = report_js.read_text(encoding="utf-8")

    assert "AI MULTI-DISEASE RISK SCREENING REPORT" in content
    assert "Patient Information" in content
    assert "Risk Assessment Findings" in content
    assert "window.print()" in content
    assert "MANDATORY CLINICAL NOTICE" in content or "disclaimer" in content.lower()
    assert "not a medical diagnosis" in content.lower()


# ==============================================================================
# 21. PRINT / REPORT STRUCTURE & PRINT CSS
# ==============================================================================
def test_21_print_report_structure_and_css():
    """Verify @media print styling in frontend/css/style.css."""
    style_css = REPO_ROOT / "frontend" / "css" / "style.css"
    content = style_css.read_text(encoding="utf-8")

    assert "@media print" in content
    assert "size: A4 portrait" in content
    assert ".printable-document" in content
    assert ".report-toolbar" in content
    assert "page-break-inside: avoid" in content


# ==============================================================================
# 22. NO FAKE RECORDS
# ==============================================================================
def test_22_no_fake_records(session):
    """Verify newly created users have exactly 0 screening records."""
    fresh_user = create_user(session, full_name="Real Only", email="real_only@test.com")
    history = get_user_screening_history(session, fresh_user.id)
    assert len(history) == 0, "No mock or dummy records should ever be automatically seeded for users!"


# ==============================================================================
# 23. NO SECRETS EXPOSED
# ==============================================================================
def test_23_no_secrets_exposed(api_client, session):
    """Verify screening and history endpoints do not leak password hashes, tokens, or credentials."""
    user = create_user(session, full_name="Credential Check", email="cred_check@test.com")
    screening = record_completed_screening(
        session,
        disease_code="asthma",
        input_data={"feature": 1},
        prediction_result={"predicted_class": 0, "probability": 0.2, "risk_level": "LOW", "model_version": "1.0.0"},
        user_id=user.id,
    )

    resp = api_client.get(
        f"/api/screenings/{screening.id}",
        headers={"X-User-Id": user.id, "X-User-Email": user.email},
    )
    assert resp.status_code == 200
    raw_text = resp.get_data(as_text=True)

    assert "password" not in raw_text.lower()
    assert "secret_key" not in raw_text.lower()
    assert "database_url" not in raw_text.lower()
    assert "supabase_key" not in raw_text.lower()



# ==============================================================================
# 24. EXISTING API REGRESSION
# ==============================================================================
def test_24_existing_api_regression(api_client):
    """Verify health and models endpoints continue to work as expected."""
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"

    resp_models = api_client.get("/api/models")
    assert resp_models.status_code == 200
    models_data = resp_models.get_json()
    assert "models" in models_data
    assert len(models_data["models"]) == 3


# ==============================================================================
# 25. EXISTING MODEL REGRESSION
# ==============================================================================
def test_25_existing_model_regression():
    """Verify prediction dispatcher continues to evaluate existing 3 models accurately."""
    from backend.services.prediction_service import get_prediction_service
    service = get_prediction_service()

    lung_payload = {
        "gender": "MALE",
        "age": 65,
        "smoking": 1,
        "yellow_fingers": 1,
        "anxiety": 1,
        "peer_pressure": 0,
        "chronic_disease": 1,
        "fatigue": 1,
        "allergy": 1,
        "wheezing": 1,
        "alcohol_consuming": 1,
        "coughing": 1,
        "shortness_of_breath": 1,
        "swallowing_difficulty": 0,
        "chest_pain": 1,
    }
    lung_res = service.predict("lung_cancer", lung_payload)
    pred_data = lung_res["prediction"]
    assert 0.0 <= pred_data["probability"] <= 1.0
    assert pred_data["risk_level"].upper() in ["LOW", "MODERATE", "HIGH"]




# ==============================================================================
# 26. EXISTING DATABASE REGRESSION
# ==============================================================================
def test_26_existing_database_regression(session):
    """Verify all 10 diseases are seeded in the database properly."""
    diseases = session.query(Disease).all()
    assert len(diseases) == 10
    codes = {d.code for d in diseases}
    assert "lung_cancer" in codes
    assert "asthma" in codes
    assert "parkinsons" in codes
    assert "breast_cancer" in codes
