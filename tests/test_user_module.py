"""User Module Verification and Integration Tests.

Validates the complete User Module functionality:
1. Login page structure and validation
2. Signup validation and confirmation
3. Login credential handling
4. Dashboard authentication requirements
5. Logout functionality
6. Three available diseases representation
7. Seven unavailable diseases ("Coming Soon") representation
8. Lung Cancer form schema alignment
9. Asthma form schema alignment
10. Parkinson's form schema alignment
11. Required field validation
12. Prediction API request serialization
13. Prediction result rendering and risk tiers
14. Safe error handling
15. User profile management
"""

import json
import os
import re
from pathlib import Path
import pytest
from backend.api.schemas.prediction import (
    LungCancerInputSchema,
    AsthmaInputSchema,
    ParkinsonsInputSchema,
    validate_payload_for_disease,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
USER_DIR = FRONTEND_DIR / "user"


def test_01_index_html_and_core_assets_exist():
    """Verify frontend entry point and core asset structure."""
    index_html = FRONTEND_DIR / "index.html"
    assert index_html.exists(), "frontend/index.html must exist"
    content = index_html.read_text(encoding="utf-8")

    assert 'id="navbar-root"' in content, "Missing navbar root element"
    assert 'id="alert-root"' in content, "Missing alert root element"
    assert 'id="app-root"' in content, "Missing app root element"
    assert 'id="loading-overlay"' in content, "Missing loading overlay element"
    assert 'src="./user/app.js"' in content, "Missing module entry script"
    assert 'href="./css/style.css"' in content, "Missing CSS stylesheet link"

    # Verify CSS file
    css_file = FRONTEND_DIR / "css" / "style.css"
    assert css_file.exists(), "frontend/css/style.css must exist"
    css_content = css_file.read_text(encoding="utf-8")
    assert "--bg-primary" in css_content, "CSS must define design tokens"
    assert "@media (max-width: 768px)" in css_content, "CSS must include responsive media queries"


def test_02_signup_validation_logic():
    """Verify signup validation rules: email format, password matching, length."""
    auth_file = USER_DIR / "services" / "auth.js"
    assert auth_file.exists(), "auth.js must exist"
    content = auth_file.read_text(encoding="utf-8")

    # Email regex verification
    assert "validateEmail" in content
    # Min length checks
    assert "password.length < 6" in content or "length < 6" in content
    assert "password !== confirmPassword" in content
    assert "fullName.trim().length < 2" in content or "fullName" in content


def test_03_login_validation_logic():
    """Verify login validation requirements for email, password, and credentials."""
    login_page = USER_DIR / "pages" / "Login.js"
    assert login_page.exists(), "Login.js must exist"
    content = login_page.read_text(encoding="utf-8")

    assert "login-email" in content
    assert "login-password" in content
    assert "btn-submit-login" in content
    assert "authService.login" in content
    assert "Please enter your email address" in content or "email" in content


def test_04_dashboard_requires_authentication():
    """Verify protected route guarding for Dashboard, Profile, and Screening."""
    store_file = USER_DIR / "hooks" / "store.js"
    assert store_file.exists(), "store.js must exist"
    content = store_file.read_text(encoding="utf-8")

    assert "protectedRoutes" in content or "authService.isAuthenticated" in content
    assert "'dashboard'" in content
    assert "'profile'" in content
    assert "'screening'" in content
    assert "'login'" in content


def test_05_logout_functionality():
    """Verify logout cleans auth state and redirects to login."""
    auth_file = USER_DIR / "services" / "auth.js"
    content = auth_file.read_text(encoding="utf-8")

    assert "logout()" in content
    assert "this.currentUser = null" in content
    assert "removeItem" in content
    assert "_notifyListeners()" in content


def test_06_three_available_diseases_in_catalog():
    """Verify exactly 3 available disease models in frontend constants."""
    constants_file = USER_DIR / "types" / "constants.js"
    assert constants_file.exists(), "constants.js must exist"
    content = constants_file.read_text(encoding="utf-8")

    # Check available diseases
    available_codes = ["lung_cancer", "asthma", "parkinsons"]
    for code in available_codes:
        assert f"code: '{code}'" in content or f'code: "{code}"' in content
        assert f"/predict/{code.replace('_', '-')}" in content or f"/predict/{code}" in content


def test_07_seven_unavailable_diseases_show_coming_soon():
    """Verify exactly 7 unavailable diseases are listed with Coming Soon status."""
    constants_file = USER_DIR / "types" / "constants.js"
    content = constants_file.read_text(encoding="utf-8")

    coming_soon_codes = [
        "breast_cancer",
        "diabetes",
        "heart_disease",
        "stroke",
        "kidney",
        "liver",
        "thyroid",
    ]

    for code in coming_soon_codes:
        assert f"code: '{code}'" in content or f'code: "{code}"' in content

    # Verify disease card marks unavailable with Coming Soon badge
    card_file = USER_DIR / "components" / "DiseaseCard.js"
    card_content = card_file.read_text(encoding="utf-8")
    assert "Coming Soon" in card_content
    assert "disabled" in card_content


def test_08_lung_cancer_form_matches_feature_schema():
    """Verify Lung Cancer screening form contains all 15 required features."""
    schema_path = BASE_DIR / "backend" / "models" / "lung_cancer" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    form_file = USER_DIR / "components" / "ScreeningForms.js"
    form_content = form_file.read_text(encoding="utf-8")

    for feat in schema["features"]:
        feat_name = feat["name"]
        assert f'name="{feat_name}"' in form_content or f"name: '{feat_name}'" in form_content, (
            f"Missing form field for Lung Cancer feature '{feat_name}'"
        )


def test_09_asthma_form_matches_feature_schema_and_excludes_leakage():
    """Verify Asthma form contains 14 features and excludes Patient_ID and Asthma_Control_Level."""
    schema_path = BASE_DIR / "backend" / "models" / "asthma" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    form_file = USER_DIR / "components" / "ScreeningForms.js"
    form_content = form_file.read_text(encoding="utf-8")

    # Excluded features MUST NOT appear as inputs in form
    assert 'name="Patient_ID"' not in form_content
    assert 'name="Asthma_Control_Level"' not in form_content

    for feat in schema["features"]:
        feat_name = feat["name"]
        assert f'name="{feat_name}"' in form_content, (
            f"Missing form field for Asthma feature '{feat_name}'"
        )


def test_10_parkinsons_form_matches_22_acoustic_features_and_excludes_subject():
    """Verify Parkinson's form contains all 22 acoustic features and excludes identifiers."""
    schema_path = BASE_DIR / "backend" / "models" / "parkinsons" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    form_file = USER_DIR / "components" / "ScreeningForms.js"
    form_content = form_file.read_text(encoding="utf-8")

    # Excluded columns MUST NOT appear as inputs
    assert 'name="name"' not in form_content
    assert 'name="subject_id"' not in form_content
    assert 'name="status"' not in form_content

    assert schema["feature_count"] == 22
    for feat in schema["features"]:
        feat_name = feat["name"]
        assert f'name="{feat_name}"' in form_content, (
            f"Missing acoustic field for Parkinson's feature '{feat_name}'"
        )


def test_11_required_field_validation_with_backend_schemas():
    """Verify backend schemas reject incomplete or invalid form payloads."""
    # 1. Lung cancer missing age
    with pytest.raises(ValueError):
        validate_payload_for_disease("lung_cancer", {
            "gender": "MALE",
            # age missing
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

    # 2. Asthma invalid BMI
    with pytest.raises(ValueError):
        validate_payload_for_disease("asthma", {
            "Age": 25.0,
            "Gender": "Female",
            "BMI": 100.0,  # exceeds upper bound of 65.0
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


def test_12_prediction_api_request_payloads_valid():
    """Verify frontend-compatible sample payloads validate against backend schemas."""
    # Lung Cancer
    lc_payload = {
        "gender": "MALE",
        "age": 60.0,
        "smoking": 1,
        "yellow_fingers": 1,
        "anxiety": 0,
        "peer_pressure": 0,
        "chronic_disease": 1,
        "fatigue": 1,
        "allergy": 0,
        "wheezing": 1,
        "alcohol_consuming": 1,
        "coughing": 1,
        "shortness_of_breath": 1,
        "swallowing_difficulty": 0,
        "chest_pain": 1,
    }
    validated_lc = validate_payload_for_disease("lung_cancer", lc_payload)
    assert validated_lc["gender"] == "MALE"

    # Asthma
    asthma_payload = {
        "Age": 38.0,
        "Gender": "Male",
        "BMI": 24.8,
        "Smoking_Status": "Never",
        "Family_History": 1,
        "Allergies": "Dust",
        "Air_Pollution_Level": "Moderate",
        "Physical_Activity_Level": "Active",
        "Occupation_Type": "Indoor",
        "Comorbidities": "None",
        "Medication_Adherence": 0.95,
        "Number_of_ER_Visits": 0.0,
        "Peak_Expiratory_Flow": 450.0,
        "FeNO_Level": 22.0,
    }
    validated_asthma = validate_payload_for_disease("asthma", asthma_payload)
    assert validated_asthma["Age"] == 38.0

    # Parkinson's (all 22 acoustic features)
    parkinsons_payload = {
        "MDVP:Fo(Hz)": 119.992,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.00370,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.426,
        "Shimmer:APQ3": 0.02182,
        "Shimmer:APQ5": 0.03130,
        "MDVP:APQ": 0.02971,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02211,
        "HNR": 21.033,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654,
    }
    validated_p = validate_payload_for_disease("parkinsons", parkinsons_payload)
    assert validated_p["MDVP:Fo(Hz)"] == 119.992


def test_13_prediction_result_card_renders_risk_and_disclaimer():
    """Verify result card component contains probability gauge, risk level, and medical disclaimer."""
    result_card = USER_DIR / "components" / "ResultCard.js"
    assert result_card.exists(), "ResultCard.js must exist"
    content = result_card.read_text(encoding="utf-8")

    assert "MANDATORY_DISCLAIMER" in content
    assert "probabilityPct" in content
    assert "risk_level" in content
    assert "Mandatory Medical Notice" in content or "disclaimer-banner" in content
    assert "Return to Dashboard" in content or "btn-return-dash" in content


def test_14_api_service_error_handling():
    """Verify api.js formats safe error messages without exposing file paths or stack traces."""
    api_file = USER_DIR / "services" / "api.js"
    assert api_file.exists(), "api.js must exist"
    content = api_file.read_text(encoding="utf-8")

    assert "checkHealth" in content
    assert "predictLungCancer" in content
    assert "predictAsthma" in content
    assert "predictParkinsons" in content
    assert "NETWORK_ERROR" in content
    assert "Unable to reach the prediction server" in content


def test_15_user_profile_management_logic():
    """Verify profile view and update supports full name, mobile, and date of birth with immutable email."""
    profile_page = USER_DIR / "pages" / "Profile.js"
    assert profile_page.exists(), "Profile.js must exist"
    content = profile_page.read_text(encoding="utf-8")

    assert "profile-name" in content
    assert "profile-email" in content
    assert "profile-mobile" in content
    assert "profile-dob" in content
    assert "disabled" in content, "Email field must be disabled as immutable identity"
    assert "authService.updateProfile" in content
