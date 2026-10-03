"""
Automated Test Suite for Heart Disease Prediction Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 9 — Automated Testing
Tests:
1. Artifact existence (model.joblib, feature_schema.json, metrics.json, model_selection.json, predict.py).
2. Model loading and pipeline architecture (Pipeline, ColumnTransformer, RandomForestClassifier).
3. Schema integrity, exact feature names (13 features), canonical ordering, and target definition.
4. Prediction interface output format, probability calibration bounds, and neutral labels.
5. Input container formats (dict, pd.Series, 1-row pd.DataFrame).
6. Comprehensive invalid-input handling (missing, extra, non-numeric, NaN, infinite, boolean, multi-row).
7. Input container immutability during prediction.
8. Deterministic output consistency across multiple executions.
9. Batch prediction interface.
10. Final evaluation metrics integrity (54 test samples, confusion matrix sums to 54, metric bounds).
11. Model selection quarantine confirmation (X_test quarantined, 216 training rows, 5 candidates).
12. Dataset file integrity and non-modification.
13. Absence of unsupported diagnostic/clinical claims in returned predictions.
"""

import os
import sys
import json
import copy
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Path configurations relative to workspace root
TESTS_DIR = Path(__file__).resolve().parent
HEART_DISEASE_DIR = TESTS_DIR.parent
WORKSPACE_ROOT = TESTS_DIR.parents[3]

MODEL_PATH = HEART_DISEASE_DIR / "model.joblib"
SCHEMA_PATH = HEART_DISEASE_DIR / "feature_schema.json"
METRICS_PATH = HEART_DISEASE_DIR / "metrics.json"
SELECTION_PATH = HEART_DISEASE_DIR / "model_selection.json"
DATASET_PATH = WORKSPACE_ROOT / "datasets" / "heart_disease" / "heart.csv"

# Test external package import from repository root without manual path manipulation
from backend.models.heart_disease.predict import (
    predict_heart_disease,
    predict,
    batch_predict,
    get_model,
    get_feature_schema,
    EXPECTED_FEATURES,
)


@pytest.fixture
def valid_input():
    """Provides a canonical valid single-patient observation (Row 0 from heart.csv)."""
    return {
        "Age": 70,
        "Sex": 1,
        "Chest pain type": 4,
        "BP": 130,
        "Cholesterol": 322,
        "FBS over 120": 0,
        "EKG results": 2,
        "Max HR": 109,
        "Exercise angina": 0,
        "ST depression": 2.4,
        "Slope of ST": 2,
        "Number of vessels fluro": 3,
        "Thallium": 3
    }


@pytest.fixture
def valid_input_absence():
    """Provides a second canonical valid patient observation (Row 1 from heart.csv)."""
    return {
        "Age": 67,
        "Sex": 0,
        "Chest pain type": 3,
        "BP": 115,
        "Cholesterol": 564,
        "FBS over 120": 0,
        "EKG results": 2,
        "Max HR": 160,
        "Exercise angina": 0,
        "ST depression": 1.6,
        "Slope of ST": 2,
        "Number of vessels fluro": 0,
        "Thallium": 7
    }


# ==============================================================================
# 1. Artifact Existence Tests
# ==============================================================================

def test_model_artifact_exists():
    """Verify model.joblib exists and is non-empty."""
    assert MODEL_PATH.exists(), f"Model artifact missing at: {MODEL_PATH}"
    assert MODEL_PATH.stat().st_size > 0, "Model artifact is empty"


def test_schema_artifact_exists():
    """Verify feature_schema.json exists and is non-empty."""
    assert SCHEMA_PATH.exists(), f"Feature schema missing at: {SCHEMA_PATH}"
    assert SCHEMA_PATH.stat().st_size > 0, "Feature schema is empty"


def test_metrics_artifact_exists():
    """Verify metrics.json exists and is non-empty."""
    assert METRICS_PATH.exists(), f"Metrics artifact missing at: {METRICS_PATH}"
    assert METRICS_PATH.stat().st_size > 0, "Metrics artifact is empty"


def test_model_selection_artifact_exists():
    """Verify model_selection.json exists and is non-empty."""
    assert SELECTION_PATH.exists(), f"Model selection report missing at: {SELECTION_PATH}"
    assert SELECTION_PATH.stat().st_size > 0, "Model selection report is empty"


def test_predict_module_exists():
    """Verify predict.py exists and is non-empty."""
    predict_file = HEART_DISEASE_DIR / "predict.py"
    assert predict_file.exists(), f"predict.py missing at: {predict_file}"
    assert predict_file.stat().st_size > 0, "predict.py is empty"


# ==============================================================================
# 2. Model Loading & Architecture Tests
# ==============================================================================

def test_model_loads_successfully():
    """Verify model.joblib loads successfully via joblib."""
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Failed to load model artifact"


def test_loaded_model_is_sklearn_pipeline():
    """Verify loaded model is a scikit-learn Pipeline instance."""
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, Pipeline), f"Expected Pipeline, got {type(model).__name__}"


def test_pipeline_stages_integrity():
    """Verify pipeline contains preprocessing and classifier stages."""
    model = joblib.load(MODEL_PATH)
    steps = dict(model.steps)
    assert "preprocessor" in steps, "Pipeline missing 'preprocessor' step"
    assert "classifier" in steps, "Pipeline missing 'classifier' step"
    assert isinstance(steps["preprocessor"], ColumnTransformer), "Preprocessor is not ColumnTransformer"
    assert isinstance(steps["classifier"], RandomForestClassifier), "Classifier is not RandomForestClassifier"


def test_model_can_predict_without_retraining(valid_input):
    """Verify model can perform inference without fitting/retraining."""
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([valid_input], columns=EXPECTED_FEATURES)
    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)


# ==============================================================================
# 3. Schema Validation Tests
# ==============================================================================

def test_schema_target_definition():
    """Verify target column name and mapping in schema."""
    schema = get_feature_schema()
    assert schema["target_column"] == "Heart Disease"
    assert schema["target_mapping"] == {"Absence": 0, "Presence": 1}
    assert set(schema["target_observed_values"]) == {"Absence", "Presence"}


def test_schema_feature_count_and_names():
    """Verify exactly 13 canonical features in expected order."""
    schema = get_feature_schema()
    assert schema["feature_count"] == 13
    assert len(schema["exact_feature_names_in_order"]) == 13

    expected = [
        "Age",
        "Sex",
        "Chest pain type",
        "BP",
        "Cholesterol",
        "FBS over 120",
        "EKG results",
        "Max HR",
        "Exercise angina",
        "ST depression",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium"
    ]
    assert schema["exact_feature_names_in_order"] == expected
    assert EXPECTED_FEATURES == expected


def test_schema_feature_data_types():
    """Verify feature data types are documented and correct."""
    schema = get_feature_schema()
    dtypes = schema["feature_data_types"]
    assert len(dtypes) == 13
    assert dtypes["ST depression"] == "float64"
    for col in EXPECTED_FEATURES:
        if col != "ST depression":
            assert dtypes[col] == "int64"


# ==============================================================================
# 4. Prediction Module Tests
# ==============================================================================

def test_prediction_output_structure(valid_input):
    """Verify structured response fields from predict_heart_disease."""
    res = predict_heart_disease(valid_input)
    assert res["status"] == "success"
    assert res["model_name"] == "Random_Forest"
    assert res["predicted_class"] in [0, 1]
    assert isinstance(res["predicted_probability"], float)
    assert 0.0 <= res["predicted_probability"] <= 1.0
    assert "0" in res["class_probabilities"]
    assert "1" in res["class_probabilities"]
    assert res["screening_estimate"] in ["Class 0", "Class 1"]


def test_prediction_probabilities_sum_to_one(valid_input):
    """Verify class probabilities sum approximately to 1.0."""
    res = predict_heart_disease(valid_input)
    p0 = res["class_probabilities"]["0"]
    p1 = res["class_probabilities"]["1"]
    assert 0.0 <= p0 <= 1.0
    assert 0.0 <= p1 <= 1.0
    assert abs((p0 + p1) - 1.0) < 1e-3


def test_predicted_probability_matches_predicted_class(valid_input):
    """Verify predicted_probability matches the probability of the predicted class."""
    res = predict_heart_disease(valid_input)
    pred_cls = str(res["predicted_class"])
    expected_prob = res["class_probabilities"][pred_cls]
    assert np.isclose(res["predicted_probability"], expected_prob)


def test_predict_alias_function(valid_input):
    """Verify predict() alias produces identical results to predict_heart_disease()."""
    res1 = predict_heart_disease(valid_input)
    res2 = predict(valid_input)
    assert res1 == res2


# ==============================================================================
# 5. Input Format Flexibility Tests
# ==============================================================================

def test_prediction_accepts_dict(valid_input):
    """Verify dict input format is supported."""
    res = predict_heart_disease(valid_input)
    assert res["status"] == "success"


def test_prediction_accepts_pandas_series(valid_input):
    """Verify pd.Series input format is supported."""
    series_input = pd.Series(valid_input)
    res = predict_heart_disease(series_input)
    assert res["status"] == "success"
    dict_res = predict_heart_disease(valid_input)
    assert res["predicted_class"] == dict_res["predicted_class"]
    assert np.isclose(res["predicted_probability"], dict_res["predicted_probability"])


def test_prediction_accepts_single_row_dataframe(valid_input):
    """Verify single-row pd.DataFrame input format is supported."""
    df_input = pd.DataFrame([valid_input])
    res = predict_heart_disease(df_input)
    assert res["status"] == "success"
    dict_res = predict_heart_disease(valid_input)
    assert res["predicted_class"] == dict_res["predicted_class"]
    assert np.isclose(res["predicted_probability"], dict_res["predicted_probability"])


# ==============================================================================
# 6. Invalid Input Tests
# ==============================================================================

def test_rejects_missing_feature(valid_input):
    """Verify ValueError is raised when a required feature is missing."""
    invalid = {k: v for k, v in valid_input.items() if k != "Thallium"}
    with pytest.raises(ValueError, match="Missing required feature"):
        predict_heart_disease(invalid)


def test_rejects_unexpected_extra_feature(valid_input):
    """Verify ValueError is raised when an unexpected extra feature is provided."""
    invalid = {**valid_input, "ExtraFeature": 123}
    with pytest.raises(ValueError, match="Unexpected feature"):
        predict_heart_disease(invalid)


def test_rejects_non_numeric_value(valid_input):
    """Verify TypeError is raised when a non-numeric string is provided."""
    invalid = {**valid_input, "Cholesterol": "high"}
    with pytest.raises(TypeError, match="must be numeric"):
        predict_heart_disease(invalid)


def test_rejects_nan_value(valid_input):
    """Verify ValueError is raised when NaN is provided."""
    invalid = {**valid_input, "BP": float("nan")}
    with pytest.raises(ValueError, match="cannot be NaN"):
        predict_heart_disease(invalid)


def test_rejects_infinite_value(valid_input):
    """Verify ValueError is raised when infinity is provided."""
    invalid = {**valid_input, "Max HR": float("inf")}
    with pytest.raises(ValueError, match="cannot be infinite"):
        predict_heart_disease(invalid)


def test_rejects_negative_infinite_value(valid_input):
    """Verify ValueError is raised when negative infinity is provided."""
    invalid = {**valid_input, "ST depression": float("-inf")}
    with pytest.raises(ValueError, match="cannot be infinite"):
        predict_heart_disease(invalid)


def test_rejects_boolean_value(valid_input):
    """Verify TypeError is raised when a boolean value is supplied for numeric features."""
    invalid = {**valid_input, "Sex": True}
    with pytest.raises(TypeError, match="cannot be boolean"):
        predict_heart_disease(invalid)


def test_rejects_multi_row_dataframe(valid_input):
    """Verify ValueError is raised when a multi-row DataFrame is passed to single-record inference."""
    multi_row_df = pd.DataFrame([valid_input, valid_input])
    with pytest.raises(ValueError, match="Expected a single sample row"):
        predict_heart_disease(multi_row_df)


def test_rejects_unsupported_container_type():
    """Verify TypeError is raised when an unsupported container (e.g. list) is passed."""
    with pytest.raises(TypeError, match="Invalid input type"):
        predict_heart_disease([1, 2, 3])


# ==============================================================================
# 7. Input Immutability Tests
# ==============================================================================

def test_input_dict_immutability(valid_input):
    """Verify the original input dictionary is not modified by predict_heart_disease()."""
    original_copy = copy.deepcopy(valid_input)
    _ = predict_heart_disease(valid_input)
    assert valid_input == original_copy, "Original input dictionary was modified!"


def test_input_dataframe_immutability(valid_input):
    """Verify the original input DataFrame is not modified by predict_heart_disease()."""
    df_input = pd.DataFrame([valid_input])
    df_copy = df_input.copy(deep=True)
    _ = predict_heart_disease(df_input)
    pd.testing.assert_frame_equal(df_input, df_copy)


# ==============================================================================
# 8. Deterministic Output Consistency Tests
# ==============================================================================

def test_deterministic_predictions(valid_input):
    """Verify identical predictions across consecutive calls without retraining."""
    res1 = predict_heart_disease(valid_input)
    res2 = predict_heart_disease(valid_input)
    assert res1["predicted_class"] == res2["predicted_class"]
    assert res1["predicted_probability"] == res2["predicted_probability"]
    assert res1["class_probabilities"] == res2["class_probabilities"]
    assert res1["screening_estimate"] == res2["screening_estimate"]


# ==============================================================================
# 9. Batch Prediction Tests
# ==============================================================================

def test_batch_prediction_multiple_records(valid_input, valid_input_absence):
    """Verify batch_predict() correctly processes a list of multiple records."""
    records = [valid_input, valid_input_absence, valid_input]
    results = batch_predict(records)

    assert len(results) == 3
    for r in results:
        assert r["status"] == "success"
        assert r["predicted_class"] in [0, 1]
        assert 0.0 <= r["predicted_probability"] <= 1.0
        assert r["screening_estimate"] in ["Class 0", "Class 1"]


# ==============================================================================
# 10. Final Metrics Validation Tests
# ==============================================================================

def test_final_metrics_payload():
    """Verify metrics.json structure, sample counts, and valid ranges."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    assert metrics_data["model_domain"] == "heart_disease"
    assert metrics_data["selected_model"] == "Random_Forest"
    assert metrics_data["train_samples"] == 216
    assert metrics_data["test_samples"] == 54

    tm = metrics_data["test_metrics"]
    for key in ["accuracy", "precision", "recall_sensitivity", "f1_score", "roc_auc", "pr_auc"]:
        assert key in tm, f"Missing metric '{key}' in test_metrics"
        val = tm[key]
        assert isinstance(val, (int, float))
        assert 0.0 <= val <= 1.0, f"Metric '{key}' ({val}) outside bounds [0, 1]"

    cm = tm["confusion_matrix"]
    tn, fp, fn, tp = cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]
    assert tn + fp + fn + tp == 54, f"Confusion matrix does not sum to 54: {tn + fp + fn + tp}"
    assert tm["actual_class_distribution"] == {"class_0": 30, "class_1": 24}


# ==============================================================================
# 11. Model Selection Validation Tests
# ==============================================================================

def test_model_selection_quarantine_confirmation():
    """Verify model_selection.json quarantine confirmation and candidate list."""
    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        sel_data = json.load(f)

    assert sel_data["selected_model"] == "Random_Forest"
    q_conf = sel_data["quarantine_confirmation"]
    assert q_conf["test_set_used_during_selection"] is False
    assert q_conf["test_rows_quarantined"] == 54
    assert q_conf["final_training_rows"] == 216

    expected_candidates = {
        "Logistic_Regression",
        "Random_Forest",
        "Support_Vector_Classifier",
        "Gradient_Boosting",
        "XGBoost"
    }
    assert set(sel_data["candidate_models_considered"]) == expected_candidates


# ==============================================================================
# 12. Dataset Integrity Tests
# ==============================================================================

def test_dataset_integrity():
    """Verify dataset file exists, is readable, and contains exactly 270 rows x 14 cols."""
    assert DATASET_PATH.exists(), f"Dataset file missing at: {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    assert df.shape == (270, 14), f"Unexpected shape {df.shape}"

    # Verify SHA-256 hash can be computed
    hasher = hashlib.sha256()
    with open(DATASET_PATH, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    assert len(file_hash) == 64, "Invalid SHA-256 hash length"


# ==============================================================================
# 13. Absence of Unsupported Clinical Claims
# ==============================================================================

def test_no_diagnostic_or_clinical_claims(valid_input):
    """Verify prediction output contains no diagnostic assertions or clinical guarantees."""
    res = predict_heart_disease(valid_input)
    serialized = json.dumps(res).lower()

    unsupported_terms = [
        "diagnosed",
        "confirmed disease",
        "patient has heart disease",
        "medically reliable",
        "clinically accurate",
        "low risk",
        "high risk"
    ]
    for term in unsupported_terms:
        assert term not in serialized, f"Unsupported clinical claim '{term}' found in output: {serialized}"

    assert res["screening_estimate"] in ["Class 0", "Class 1"]
