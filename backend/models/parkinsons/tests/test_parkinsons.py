"""Unit and Integration Tests for Parkinson's Disease Prediction Pipeline.

Validates artifact integrity, schema alignment, input sanitization,
absence of target/subject leakage, and inference reproducibility.
"""

import json
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure parent directory is on sys.path for local module resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARKINSONS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARKINSONS_DIR not in sys.path:
    sys.path.insert(0, PARKINSONS_DIR)

import importlib.util

_prep_path = os.path.join(PARKINSONS_DIR, "preprocessing.py")
_spec_prep = importlib.util.spec_from_file_location("parkinsons_preprocessing_test", _prep_path)
_prep_mod = importlib.util.module_from_spec(_spec_prep)
_spec_prep.loader.exec_module(_prep_mod)

EXCLUDED_COLUMNS = _prep_mod.EXCLUDED_COLUMNS
FEATURE_BOUNDS = _prep_mod.FEATURE_BOUNDS
FEATURE_COLUMNS = _prep_mod.FEATURE_COLUMNS
TARGET_COLUMN = _prep_mod.TARGET_COLUMN
extract_subject_id = _prep_mod.extract_subject_id
format_single_input = _prep_mod.format_single_input
load_dataset = _prep_mod.load_dataset

_pred_path = os.path.join(PARKINSONS_DIR, "predict.py")
_spec_pred = importlib.util.spec_from_file_location("parkinsons_predict_test", _pred_path)
_pred_mod = importlib.util.module_from_spec(_spec_pred)
_spec_pred.loader.exec_module(_pred_mod)

MODEL_PATH = _pred_mod.MODEL_PATH
PREPROCESSOR_PATH = _pred_mod.PREPROCESSOR_PATH
SCHEMA_PATH = _pred_mod.SCHEMA_PATH
compute_heuristic_risk_tier = _pred_mod.compute_heuristic_risk_tier
get_artifacts = _pred_mod.get_artifacts
predict = _pred_mod.predict

SAMPLE_VALID_RECORD = {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.0313,
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


def test_1_artifacts_exist():
    """Verify that all four required static artifacts exist on disk."""
    assert os.path.exists(MODEL_PATH), f"Missing model artifact at: {MODEL_PATH}"
    assert os.path.exists(PREPROCESSOR_PATH), f"Missing preprocessor artifact at: {PREPROCESSOR_PATH}"
    assert os.path.exists(SCHEMA_PATH), f"Missing schema artifact at: {SCHEMA_PATH}"
    metrics_path = os.path.join(PARKINSONS_DIR, "metrics.json")
    assert os.path.exists(metrics_path), f"Missing metrics artifact at: {metrics_path}"


def test_2_artifacts_load():
    """Verify that model, preprocessor, and schema load successfully and are valid."""
    model, preprocessor, schema = get_artifacts()
    assert model is not None, "Loaded model artifact is None."
    assert preprocessor is not None, "Loaded preprocessor artifact is None."
    assert schema is not None, "Loaded schema artifact is None."
    assert hasattr(model, "predict"), "Model object lacks predict method."
    assert hasattr(model, "predict_proba"), "Model object lacks predict_proba method."
    assert hasattr(preprocessor, "transform"), "Preprocessor object lacks transform method."


def test_3_expected_feature_validation():
    """Verify that expected feature list has exactly 22 features with defined bounds."""
    assert len(FEATURE_COLUMNS) == 22, f"Expected 22 features, got {len(FEATURE_COLUMNS)}"
    for col in FEATURE_COLUMNS:
        assert col in FEATURE_BOUNDS, f"Feature '{col}' missing from FEATURE_BOUNDS."
        low, high = FEATURE_BOUNDS[col]
        assert low < high, f"Invalid bounds for feature '{col}': ({low}, {high})"


def test_4_valid_sample_prediction():
    """Verify inference on a valid sample patient recording."""
    result = predict(SAMPLE_VALID_RECORD)
    assert isinstance(result, dict)
    assert result["class"] in (0, 1)
    assert result["prediction"] in ("Parkinson's Disease", "Healthy / No Parkinson's Detected")


def test_5_output_format():
    """Verify the presence and types of all required output response fields."""
    result = predict(SAMPLE_VALID_RECORD)
    required_keys = ["prediction", "class", "probability", "risk_level", "risk_tier_basis", "disclaimer"]
    for key in required_keys:
        assert key in result, f"Missing output key: '{key}'"
    assert isinstance(result["prediction"], str)
    assert isinstance(result["class"], int)
    assert isinstance(result["probability"], float)
    assert result["risk_level"] in ("Low", "Moderate", "High")
    assert "software-defined" in result["risk_tier_basis"].lower()
    assert "not a clinical" in result["disclaimer"].lower()


def test_6_probability_range():
    """Verify that predicted probabilities are bounded in [0.0, 1.0]."""
    result = predict(SAMPLE_VALID_RECORD)
    prob = result["probability"]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds [0.0, 1.0]"
    # Verify risk tier corresponds to the defined heuristic intervals
    if prob < 0.30:
        assert result["risk_level"] == "Low"
    elif prob < 0.70:
        assert result["risk_level"] == "Moderate"
    else:
        assert result["risk_level"] == "High"


def test_7_missing_feature_rejection():
    """Verify that omitting any required feature raises ValueError."""
    for col in FEATURE_COLUMNS[:5]:  # Test first 5 features
        incomplete = {k: v for k, v in SAMPLE_VALID_RECORD.items() if k != col}
        with pytest.raises(ValueError, match="Missing required feature"):
            predict(incomplete)


def test_8_unexpected_excluded_feature_handling():
    """Verify that passing excluded fields (name, subject_id, status) does not leak or crash inference."""
    with_leakage_keys = dict(SAMPLE_VALID_RECORD)
    with_leakage_keys["name"] = "phon_R01_S99_1"
    with_leakage_keys["subject_id"] = "S99"
    with_leakage_keys["status"] = 0
    with_leakage_keys["unrelated_metadata"] = "extra_field"

    # Must predict successfully and safely ignore excluded/administrative fields
    res = predict(with_leakage_keys)
    assert res["class"] in (0, 1)


def test_9_deterministic_repeated_prediction():
    """Verify that identical inputs produce identical predictions and probabilities."""
    res1 = predict(SAMPLE_VALID_RECORD)
    res2 = predict(SAMPLE_VALID_RECORD)
    assert res1["class"] == res2["class"]
    assert res1["probability"] == res2["probability"]
    assert res1["risk_level"] == res2["risk_level"]


def test_10_batch_prediction():
    """Verify that batch predictions on lists of dicts and DataFrames work seamlessly."""
    batch_list = [SAMPLE_VALID_RECORD, SAMPLE_VALID_RECORD]
    res_list = predict(batch_list)
    assert isinstance(res_list, list)
    assert len(res_list) == 2
    assert res_list[0]["probability"] == res_list[1]["probability"]

    df_batch = pd.DataFrame([SAMPLE_VALID_RECORD, SAMPLE_VALID_RECORD])
    res_df = predict(df_batch)
    assert isinstance(res_df, list)
    assert len(res_df) == 2


def test_11_subject_grouping_logic():
    """Verify accurate extraction of subject IDs from recording name identifiers."""
    assert extract_subject_id("phon_R01_S01_1") == "S01"
    assert extract_subject_id("phon_R01_S27_7") == "S27"
    assert extract_subject_id("phon_R01_S50_6") == "S50"
    assert extract_subject_id("custom_id") == "custom_id"


def test_12_no_subject_overlap_between_training_and_test():
    """Verify from metrics.json that train and test subject sets are strictly disjoint."""
    metrics_path = os.path.join(PARKINSONS_DIR, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["partitioning"]["subject_overlap"] == 0, "Subject overlap detected in metrics.json!"
    test_subjs = set(metrics["partitioning"]["test_subject_ids"])
    assert len(test_subjs) == 6, f"Expected 6 test subjects, got {len(test_subjs)}"


def test_13_feature_schema_matches_predictor():
    """Verify that feature_schema.json matches FEATURE_COLUMNS exactly in order and names."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    schema_feature_names = [f["name"] for f in schema["features"]]
    assert schema_feature_names == FEATURE_COLUMNS, "Schema features do not match preprocessing FEATURE_COLUMNS."
    assert schema["feature_count"] == 22


def test_14_excluded_columns_are_not_model_predictors():
    """Verify that name, subject_id, and status are completely excluded from predictor list and preprocessor."""
    for excl in EXCLUDED_COLUMNS:
        assert excl not in FEATURE_COLUMNS, f"Excluded column '{excl}' found in FEATURE_COLUMNS!"
    assert TARGET_COLUMN not in FEATURE_COLUMNS, "Target column 'status' found in FEATURE_COLUMNS!"

    _, preprocessor, _ = get_artifacts()
    # Check that preprocessor's transformers only reference FEATURE_COLUMNS
    for name, trans, cols in preprocessor.transformers:
        if name != "remainder":
            for c in cols:
                assert c in FEATURE_COLUMNS
                assert c not in EXCLUDED_COLUMNS
                assert c != TARGET_COLUMN
