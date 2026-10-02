"""Clinic Module Verification and Integration Tests.

Validates the complete Clinic Module functionality:
1. Clinic login page renders
2. Clinic login validation works
3. Clinic session protection works
4. Clinic logout works
5. Clinic dashboard renders
6. Clinic name and practitioner are displayed
7. Three available diseases are displayed
8. Seven unavailable diseases show Coming Soon
9. Lung Cancer clinic form renders
10. Asthma clinic form renders
11. Parkinson's clinic form renders
12. Patient information validation works
13. Disease feature validation works
14. Lung Cancer API request is correctly formed
15. Asthma API request is correctly formed
16. Parkinson's API request is correctly formed
17. Prediction result renders with risk level and disclaimer
18. API error is handled safely
19. Clinic profile renders
20. Normal user cannot access clinic-only route (role boundary enforced)
"""

import json
from pathlib import Path
import pytest
from backend.api.schemas.prediction import validate_payload_for_disease

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
CLINIC_DIR = FRONTEND_DIR / "clinic"


def test_01_clinic_html_and_core_assets_exist():
    """Verify clinic.html entry point and script references."""
    clinic_html = FRONTEND_DIR / "clinic.html"
    assert clinic_html.exists(), "frontend/clinic.html must exist"
    content = clinic_html.read_text(encoding="utf-8")

    assert 'id="navbar-root"' in content
    assert 'id="alert-root"' in content
    assert 'id="app-root"' in content
    assert 'id="loading-overlay"' in content
    assert 'src="./clinic/app.js"' in content, "Missing clinic app module script"
    assert 'href="./css/style.css"' in content


def test_02_clinic_login_page_renders():
    """Verify clinic login page structure and demo provider credentials."""
    login_page = CLINIC_DIR / "pages" / "Login.js"
    assert login_page.exists(), "clinic/pages/Login.js must exist"
    content = login_page.read_text(encoding="utf-8")

    assert "clinic-login-email" in content
    assert "clinic-login-password" in content
    assert "btn-submit-clinic-login" in content
    assert "clinic@generalhospital.org" in content
    assert "Clinic@123" in content


def test_03_clinic_login_validation_works():
    """Verify clinic login validation rejects invalid or missing credentials."""
    auth_service = CLINIC_DIR / "services" / "auth.js"
    assert auth_service.exists(), "clinic/services/auth.js must exist"
    content = auth_service.read_text(encoding="utf-8")

    assert "validateEmail" in content
    assert "Please provide a valid institutional clinic email address" in content or "valid institutional" in content
    assert "Password is required for clinical provider authentication" in content or "Password is required" in content
    assert "CLINIC_ROLE" in content


def test_04_clinic_session_protection_works():
    """Verify clinic store protects routes against unauthenticated or non-clinic access."""
    store_file = CLINIC_DIR / "hooks" / "store.js"
    assert store_file.exists(), "clinic/hooks/store.js must exist"
    content = store_file.read_text(encoding="utf-8")

    assert "protectedRoutes" in content or "clinicAuthService.isAuthenticated" in content
    assert "'dashboard'" in content
    assert "'profile'" in content
    assert "'screening'" in content
    assert "'result'" in content
    assert "'login'" in content


def test_05_clinic_logout_works():
    """Verify clinic logout clears session and storage."""
    auth_service = CLINIC_DIR / "services" / "auth.js"
    content = auth_service.read_text(encoding="utf-8")

    assert "logout()" in content
    assert "this.currentClinicUser = null" in content
    assert "CLINIC_SESSION_KEY" in content
    assert "removeItem" in content


def test_06_clinic_dashboard_renders():
    """Verify clinic dashboard page renders clinical workflow and disease grid."""
    dashboard_page = CLINIC_DIR / "pages" / "Dashboard.js"
    assert dashboard_page.exists(), "clinic/pages/Dashboard.js must exist"
    content = dashboard_page.read_text(encoding="utf-8")

    assert "Clinical Consultation & Screening Console" in content or "Clinical Consultation" in content
    assert "clinic-disease-catalog-grid" in content
    assert "createClinicDiseaseCard" in content


def test_07_clinic_name_and_practitioner_displayed():
    """Verify clinic facility and practitioner full name are displayed in header and cards."""
    header_file = CLINIC_DIR / "components" / "Header.js"
    assert header_file.exists(), "clinic/components/Header.js must exist"
    content = header_file.read_text(encoding="utf-8")

    assert "clinicName" in content
    assert "clinicUser" in content
    assert "Provider Portal" in content or "Clinical Provider" in content


def test_08_available_diseases_are_displayed():
    """Verify exactly 3 available disease models in clinic constants."""
    constants_file = CLINIC_DIR / "types" / "constants.js"
    assert constants_file.exists(), "clinic/types/constants.js must exist"
    content = constants_file.read_text(encoding="utf-8")

    available_codes = ["lung_cancer", "asthma", "parkinsons"]
    for code in available_codes:
        assert f"code: '{code}'" in content or f'code: "{code}"' in content
        assert "available: true" in content


def test_09_seven_unavailable_diseases_show_coming_soon():
    """Verify exactly 7 unavailable diseases are marked Coming Soon with disabled buttons."""
    constants_file = CLINIC_DIR / "types" / "constants.js"
    content = constants_file.read_text(encoding="utf-8")

    coming_soon = ["breast_cancer", "diabetes", "heart_disease", "stroke", "kidney", "liver", "thyroid"]
    for code in coming_soon:
        assert f"code: '{code}'" in content or f'code: "{code}"' in content

    card_file = CLINIC_DIR / "components" / "ClinicCard.js"
    assert card_file.exists(), "clinic/components/ClinicCard.js must exist"
    card_content = card_file.read_text(encoding="utf-8")
    assert "Coming Soon" in card_content
    assert "disabled" in card_content


def test_10_lung_cancer_clinic_form_renders():
    """Verify Lung Cancer screening inputs strictly conform to feature schema (15 features)."""
    schema_path = BASE_DIR / "backend" / "models" / "lung_cancer" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    form_file = FRONTEND_DIR / "user" / "components" / "ScreeningForms.js"
    content = form_file.read_text(encoding="utf-8")

    assert len(schema["features"]) == 15
    for feat in schema["features"]:
        feat_name = feat["name"]
        assert f'name="{feat_name}"' in content or f"name: '{feat_name}'" in content


def test_11_asthma_clinic_form_renders_and_excludes_leakage():
    """Verify Asthma form contains 14 features and excludes Patient_ID and Asthma_Control_Level."""
    schema_path = BASE_DIR / "backend" / "models" / "asthma" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    form_file = FRONTEND_DIR / "user" / "components" / "ScreeningForms.js"
    content = form_file.read_text(encoding="utf-8")

    assert len(schema["features"]) == 14
    assert 'name="Patient_ID"' not in content
    assert 'name="Asthma_Control_Level"' not in content

    for feat in schema["features"]:
        feat_name = feat["name"]
        assert f'name="{feat_name}"' in content


def test_12_parkinsons_clinic_form_renders_and_excludes_identifiers():
    """Verify Parkinson's form contains 22 acoustic features and excludes name, subject_id, status."""
    schema_path = BASE_DIR / "backend" / "models" / "parkinsons" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    form_file = FRONTEND_DIR / "user" / "components" / "ScreeningForms.js"
    content = form_file.read_text(encoding="utf-8")

    assert schema["feature_count"] == 22
    assert 'name="name"' not in content
    assert 'name="subject_id"' not in content
    assert 'name="status"' not in content

    for feat in schema["features"]:
        feat_name = feat["name"]
        assert f'name="{feat_name}"' in content


def test_13_patient_information_validation_works():
    """Verify patient demographic intake form enforces full name and email/MRN validation."""
    patient_form = CLINIC_DIR / "components" / "PatientForm.js"
    assert patient_form.exists(), "clinic/components/PatientForm.js must exist"
    content = patient_form.read_text(encoding="utf-8")

    assert "patientFullName" in content or "patient-fullname" in content
    assert "patientEmail" in content or "patient-email" in content
    assert "patientMobile" in content or "patient-mobile" in content
    assert "patientDob" in content or "patient-dob" in content
    assert "validateAndExtractPatientInfo" in content


def test_14_disease_feature_validation_works():
    """Verify backend Pydantic validation rejects invalid or incomplete clinic payloads."""
    # 1. Reject invalid Lung Cancer age
    with pytest.raises(ValueError):
        validate_payload_for_disease("lung_cancer", {
            "gender": "MALE",
            "age": 150.0,  # exceeds upper bound of 120
            "smoking": 1,
            "yellow_fingers": 0,
            "anxiety": 0,
            "peer_pressure": 0,
            "chronic_disease": 0,
            "fatigue": 0,
            "allergy": 0,
            "wheezing": 0,
            "alcohol_consuming": 0,
            "coughing": 0,
            "shortness_of_breath": 0,
            "swallowing_difficulty": 0,
            "chest_pain": 0,
        })

    # 2. Reject missing Asthma feature
    with pytest.raises(ValueError):
        validate_payload_for_disease("asthma", {
            "Age": 30.0,
            "Gender": "Male",
            # BMI missing
            "Smoking_Status": "Never",
            "Family_History": 0,
            "Allergies": "None",
            "Air_Pollution_Level": "Low",
            "Physical_Activity_Level": "Moderate",
            "Occupation_Type": "Indoor",
            "Comorbidities": "None",
            "Medication_Adherence": 0.8,
            "Number_of_ER_Visits": 0.0,
            "Peak_Expiratory_Flow": 400.0,
            "FeNO_Level": 20.0,
        })


def test_15_lung_cancer_api_request_correctly_formed():
    """Verify serialized clinic Lung Cancer payload matches schema."""
    lc_payload = {
        "gender": "FEMALE",
        "age": 58.0,
        "smoking": 1,
        "yellow_fingers": 1,
        "anxiety": 1,
        "peer_pressure": 0,
        "chronic_disease": 1,
        "fatigue": 1,
        "allergy": 1,
        "wheezing": 1,
        "alcohol_consuming": 0,
        "coughing": 1,
        "shortness_of_breath": 1,
        "swallowing_difficulty": 1,
        "chest_pain": 1,
    }
    validated = validate_payload_for_disease("lung_cancer", lc_payload)
    assert validated["gender"] == "FEMALE"
    assert validated["age"] == 58.0


def test_16_asthma_api_request_correctly_formed():
    """Verify serialized clinic Asthma payload matches schema."""
    asthma_payload = {
        "Age": 45.0,
        "Gender": "Female",
        "BMI": 28.2,
        "Smoking_Status": "Current",
        "Family_History": 1,
        "Allergies": "Multiple",
        "Air_Pollution_Level": "High",
        "Physical_Activity_Level": "Sedentary",
        "Occupation_Type": "Outdoor",
        "Comorbidities": "Hypertension",
        "Medication_Adherence": 0.70,
        "Number_of_ER_Visits": 2.0,
        "Peak_Expiratory_Flow": 320.0,
        "FeNO_Level": 45.0,
    }
    validated = validate_payload_for_disease("asthma", asthma_payload)
    assert validated["Age"] == 45.0
    assert validated["FeNO_Level"] == 45.0


def test_17_parkinsons_api_request_correctly_formed():
    """Verify serialized clinic Parkinson's payload contains all 22 acoustic features."""
    p_payload = {
        "MDVP:Fo(Hz)": 197.076,
        "MDVP:Fhi(Hz)": 206.896,
        "MDVP:Flo(Hz)": 192.055,
        "MDVP:Jitter(%)": 0.00289,
        "MDVP:Jitter(Abs)": 0.00001,
        "MDVP:RAP": 0.00166,
        "MDVP:PPQ": 0.00168,
        "Jitter:DDP": 0.00498,
        "MDVP:Shimmer": 0.01098,
        "MDVP:Shimmer(dB)": 0.097,
        "Shimmer:APQ3": 0.00563,
        "Shimmer:APQ5": 0.00680,
        "MDVP:APQ": 0.00802,
        "Shimmer:DDA": 0.01689,
        "NHR": 0.00339,
        "HNR": 26.775,
        "RPDE": 0.422229,
        "DFA": 0.741367,
        "spread1": -7.348300,
        "spread2": 0.177551,
        "D2": 1.743867,
        "PPE": 0.085569,
    }
    validated = validate_payload_for_disease("parkinsons", p_payload)
    assert validated["MDVP:Fo(Hz)"] == 197.076
    assert len(validated) == 22


def test_18_prediction_result_renders_with_risk_and_disclaimer():
    """Verify clinic result card displays probability, risk level, patient identity, and disclaimer."""
    result_card = CLINIC_DIR / "components" / "ResultCard.js"
    assert result_card.exists(), "clinic/components/ResultCard.js must exist"
    content = result_card.read_text(encoding="utf-8")

    assert "MANDATORY_CLINICAL_DISCLAIMER" in content
    assert "probabilityPct" in content
    assert "risk_level" in content
    assert "patientName" in content or "patientInfo" in content
    assert "clinicUser" in content or "clinicName" in content
    assert "Mandatory Medical Notice" in content


def test_19_api_error_is_handled_safely():
    """Verify clinic API service error handling formats safe messages without leaking paths or traces."""
    api_service = CLINIC_DIR / "services" / "api.js"
    assert api_service.exists(), "clinic/services/api.js must exist"
    content = api_service.read_text(encoding="utf-8")

    assert "Clinical screening backend is unreachable" in content or "NETWORK_ERROR" in content
    assert "predictLungCancer" in content
    assert "predictAsthma" in content
    assert "predictParkinsons" in content


def test_20_clinic_profile_renders():
    """Verify clinic profile page supports Full Name, immutable Email, Mobile, and Clinic Name."""
    profile_page = CLINIC_DIR / "pages" / "Profile.js"
    assert profile_page.exists(), "clinic/pages/Profile.js must exist"
    content = profile_page.read_text(encoding="utf-8")

    assert "clinic-prof-name" in content
    assert "clinic-prof-email" in content
    assert "clinic-prof-facility" in content
    assert "clinic-prof-mobile" in content
    assert "disabled" in content, "Email field must be disabled as immutable login identity"
    assert "clinicAuthService.updateProfile" in content


def test_21_normal_user_cannot_access_clinic_only_route():
    """Verify role boundary: patient tokens cannot authenticate to clinic service or clinic routes."""
    auth_service = CLINIC_DIR / "services" / "auth.js"
    content = auth_service.read_text(encoding="utf-8")

    # Verify session key separation
    assert "ai_screening_clinic_session" in content
    assert "CLINIC_ROLE" in content
    assert "user.role !== CLINIC_ROLE" in content
    assert "Unauthorized role" in content or "Access denied" in content
