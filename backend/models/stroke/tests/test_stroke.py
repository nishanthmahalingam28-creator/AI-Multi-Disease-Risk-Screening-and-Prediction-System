"""
Automated Test Suite for Stroke Prediction Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 9 — Automated Testing
Comprehensive test suite verifying:
1. Artifact existence and loading (model.joblib, preprocessor.joblib, feature_schema.json, metrics.json, model_selection.json)
2. Schema integrity, 10 canonical features, target exclusion
3. Prediction module API, input validation, output contract, error handling
4. Determinism, input immutability, absence of retraining
5. Dataset SHA-256 hash preservation
6. Independent evaluation metrics verification on test split
7. Confusion matrix internal consistency
8. Model selection report consistency and test quarantine confirmation
"""

import os
import sys
import json
import math
import copy
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Robust path handling relative to workspace root
TESTS_DIR = Path(__file__).resolve().parent
STROKE_DIR = TESTS_DIR.parent
WORKSPACE_ROOT = TESTS_DIR.parents[3]

MODEL_PATH = STROKE_DIR / "model.joblib"
PREPROCESSOR_PATH = STROKE_DIR / "preprocessor.joblib"
SCHEMA_PATH = STROKE_DIR / "feature_schema.json"
METRICS_PATH = STROKE_DIR / "metrics.json"
SELECTION_PATH = STROKE_DIR / "model_selection.json"
PREDICT_PATH = STROKE_DIR / "predict.py"
DATASET_PATH = WORKSPACE_ROOT / "datasets" / "stroke" / "stroke.csv"

EXPECTED_DATASET_HASH = "aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815"

# Ensure repository root is in sys.path
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Direct import from repository root
from backend.models.stroke.predict import (
    predict_one,
    predict,
    predict_stroke,
    batch_predict,
    get_model,
    get_feature_schema,
    EXPECTED_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)
from backend.models.stroke.split_data import create_train_test_split


@pytest.fixture
def valid_sample():
    """Provides a valid single-patient record (from Row 0 of stroke.csv)."""
    return {
        "gender": "Male",
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }


@pytest.fixture
def valid_sample_negative():
    """Provides a second valid patient record representing a typical negative screening observation."""
    return {
        "gender": "Female",
        "age": 35.0,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Rural",
        "avg_glucose_level": 82.5,
        "bmi": 22.4,
        "smoking_status": "never smoked"
    }


# ==============================================================================
# 1. Artifact Existence & Loading Tests (Items 1 - 5)
# ==============================================================================

def test_01_model_artifact_exists_and_loads():
    """1. Verify model.joblib exists, is non-empty, and loads successfully."""
    assert MODEL_PATH.exists(), f"Model artifact missing at: {MODEL_PATH}"
    assert MODEL_PATH.stat().st_size > 0, "Model artifact file is empty"
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Loaded model is None"


def test_02_preprocessor_artifact_exists_and_loads():
    """2. Verify preprocessor.joblib exists, is non-empty, and loads successfully."""
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor artifact missing at: {PREPROCESSOR_PATH}"
    assert PREPROCESSOR_PATH.stat().st_size > 0, "Preprocessor artifact file is empty"
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    assert preprocessor is not None, "Loaded preprocessor is None"
    assert isinstance(preprocessor, ColumnTransformer), f"Expected ColumnTransformer, got {type(preprocessor)}"


def test_03_feature_schema_json_valid():
    """3. Verify feature_schema.json exists and is valid JSON."""
    assert SCHEMA_PATH.exists(), f"feature_schema.json missing at: {SCHEMA_PATH}"
    assert SCHEMA_PATH.stat().st_size > 0, "feature_schema.json is empty"
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert isinstance(schema, dict), "feature_schema.json is not a valid JSON dictionary"
    assert "exact_feature_names_in_order" in schema


def test_04_metrics_json_valid():
    """4. Verify metrics.json exists and is valid JSON."""
    assert METRICS_PATH.exists(), f"metrics.json missing at: {METRICS_PATH}"
    assert METRICS_PATH.stat().st_size > 0, "metrics.json is empty"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert isinstance(metrics, dict), "metrics.json is not a valid JSON dictionary"
    assert "accuracy" in metrics
    assert "recall" in metrics


def test_05_model_selection_json_valid():
    """5. Verify model_selection.json exists and is valid JSON."""
    assert SELECTION_PATH.exists(), f"model_selection.json missing at: {SELECTION_PATH}"
    assert SELECTION_PATH.stat().st_size > 0, "model_selection.json is empty"
    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        selection = json.load(f)
    assert isinstance(selection, dict), "model_selection.json is not a valid JSON dictionary"
    assert "selected_model" in selection


# ==============================================================================
# 2. Package Import & Pipeline Architecture Tests (Items 6 - 9)
# ==============================================================================

def test_06_predict_module_imports_from_root():
    """6. Verify predict.py imports successfully from the repository root."""
    import backend.models.stroke.predict as stroke_predict
    assert hasattr(stroke_predict, "predict_one")
    assert callable(stroke_predict.predict_one)


def test_07_saved_model_is_sklearn_pipeline():
    """7. Verify saved model is a usable scikit-learn Pipeline instance."""
    model = get_model()
    assert isinstance(model, Pipeline), f"Expected sklearn Pipeline, got {type(model).__name__}"
    step_names = [name for name, _ in model.steps]
    assert "preprocessor" in step_names, "Pipeline missing 'preprocessor' step"
    assert "classifier" in step_names, "Pipeline missing 'classifier' step"
    assert isinstance(model.named_steps["preprocessor"], ColumnTransformer)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)


def test_08_schema_features_and_canonical_order():
    """8. Verify the expected 10 input features are present in the schema in the correct order."""
    schema = get_feature_schema()
    expected_10 = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status"
    ]
    assert schema["exact_feature_names_in_order"] == expected_10
    assert EXPECTED_FEATURES == expected_10


def test_09_target_stroke_not_in_prediction_features():
    """9. Verify target 'stroke' and identifier 'id' are excluded from prediction features."""
    schema = get_feature_schema()
    assert "stroke" not in schema["exact_feature_names_in_order"]
    assert "id" not in schema["exact_feature_names_in_order"]
    assert "stroke" not in EXPECTED_FEATURES
    assert "id" not in EXPECTED_FEATURES


# ==============================================================================
# 3. Prediction Output Specification Tests (Items 10 - 15)
# ==============================================================================

def test_10_predict_one_accepts_valid_input(valid_sample):
    """10. Verify predict_one() accepts a valid patient input dictionary."""
    result = predict_one(valid_sample)
    assert isinstance(result, dict)
    assert result["status"] == "success"


def test_11_predicted_class_is_zero_or_one(valid_sample, valid_sample_negative):
    """11. Verify predicted_class is exactly 0 or 1 integer."""
    res1 = predict_one(valid_sample)
    res2 = predict_one(valid_sample_negative)
    assert res1["predicted_class"] in [0, 1]
    assert res2["predicted_class"] in [0, 1]
    assert isinstance(res1["predicted_class"], int)
    assert isinstance(res2["predicted_class"], int)


def test_12_predicted_probability_finite_and_in_range(valid_sample):
    """12. Verify predicted_probability is finite and between 0 and 1."""
    res = predict_one(valid_sample)
    prob = res["predicted_probability"]
    assert isinstance(prob, float)
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0


def test_13_class_probabilities_contains_both_classes(valid_sample):
    """13. Verify class_probabilities contains '0' and '1' keys."""
    res = predict_one(valid_sample)
    assert "class_probabilities" in res
    assert "0" in res["class_probabilities"]
    assert "1" in res["class_probabilities"]


def test_14_class_probabilities_finite_and_in_range(valid_sample):
    """14. Verify each class probability is finite and between 0 and 1."""
    res = predict_one(valid_sample)
    p0 = res["class_probabilities"]["0"]
    p1 = res["class_probabilities"]["1"]
    assert math.isfinite(p0) and 0.0 <= p0 <= 1.0
    assert math.isfinite(p1) and 0.0 <= p1 <= 1.0


def test_15_class_probabilities_sum_to_one(valid_sample):
    """15. Verify class probabilities sum approximately to 1.0."""
    res = predict_one(valid_sample)
    p0 = res["class_probabilities"]["0"]
    p1 = res["class_probabilities"]["1"]
    assert abs((p0 + p1) - 1.0) < 1e-3


# ==============================================================================
# 4. Error Handling & Input Validation Tests (Items 16 - 22)
# ==============================================================================

def test_16_rejects_missing_required_feature(valid_sample):
    """16. Verify missing required feature is rejected with ValueError."""
    for feature in EXPECTED_FEATURES:
        incomplete = {k: v for k, v in valid_sample.items() if k != feature}
        with pytest.raises(ValueError, match="Missing required feature"):
            predict_one(incomplete)


def test_17_rejects_extra_unexpected_feature(valid_sample):
    """17. Verify extra/unexpected feature (e.g. 'stroke' or 'id') is rejected."""
    with pytest.raises(ValueError, match="Unexpected feature"):
        predict_one({**valid_sample, "stroke": 1})
    with pytest.raises(ValueError, match="Unexpected feature"):
        predict_one({**valid_sample, "id": 9999})
    with pytest.raises(ValueError, match="Unexpected feature"):
        predict_one({**valid_sample, "extra_vital": 100})


def test_18_rejects_nan_input(valid_sample):
    """18. Verify NaN input is rejected with ValueError."""
    with pytest.raises(ValueError, match="cannot be null or NaN"):
        predict_one({**valid_sample, "bmi": float("nan")})
    with pytest.raises(ValueError, match="cannot be null or NaN"):
        predict_one({**valid_sample, "avg_glucose_level": np.nan})


def test_19_rejects_infinite_numeric_input(valid_sample):
    """19. Verify infinite numeric input is rejected with ValueError."""
    with pytest.raises(ValueError, match="cannot be infinite"):
        predict_one({**valid_sample, "avg_glucose_level": float("inf")})
    with pytest.raises(ValueError, match="cannot be infinite"):
        predict_one({**valid_sample, "bmi": float("-inf")})


def test_20_rejects_invalid_numeric_input(valid_sample):
    """20. Verify non-numeric or boolean input for numeric features is rejected."""
    # String for numeric
    with pytest.raises(TypeError, match="must be numeric"):
        predict_one({**valid_sample, "age": "sixty-seven"})
    # Boolean for numeric
    with pytest.raises(TypeError, match="cannot be boolean"):
        predict_one({**valid_sample, "hypertension": True})
    with pytest.raises(TypeError, match="cannot be boolean"):
        predict_one({**valid_sample, "heart_disease": False})


def test_21_rejects_empty_input():
    """21. Verify empty dictionary, Series, or DataFrame is rejected with ValueError."""
    with pytest.raises(ValueError, match="Empty dictionary"):
        predict_one({})
    with pytest.raises(ValueError, match="Empty Series"):
        predict_one(pd.Series(dtype=object))
    with pytest.raises(ValueError, match="Empty DataFrame"):
        predict_one(pd.DataFrame())


def test_22_rejects_multiple_rows_in_dataframe(valid_sample):
    """22. Verify multi-row DataFrame is rejected by predict_one()."""
    multi_df = pd.DataFrame([valid_sample, valid_sample])
    with pytest.raises(ValueError, match="Expected a single sample row"):
        predict_one(multi_df)


# ==============================================================================
# 5. Format Flexibility, Invariance & Immutability Tests (Items 23 - 29)
# ==============================================================================

def test_23_supports_pandas_series(valid_sample):
    """23. Verify pandas Series input works and matches dictionary result."""
    dict_res = predict_one(valid_sample)
    series_res = predict_one(pd.Series(valid_sample))
    assert series_res["predicted_class"] == dict_res["predicted_class"]
    assert np.isclose(series_res["predicted_probability"], dict_res["predicted_probability"])


def test_24_supports_single_row_dataframe(valid_sample):
    """24. Verify one-row pandas DataFrame input works and matches dictionary result."""
    dict_res = predict_one(valid_sample)
    df_res = predict_one(pd.DataFrame([valid_sample]))
    assert df_res["predicted_class"] == dict_res["predicted_class"]
    assert np.isclose(df_res["predicted_probability"], dict_res["predicted_probability"])


def test_25_feature_ordering_invariance(valid_sample):
    """25. Verify input feature dictionary key ordering does not alter prediction."""
    dict_res = predict_one(valid_sample)
    # Reverse key ordering
    reversed_keys = list(valid_sample.keys())[::-1]
    reversed_sample = {k: valid_sample[k] for k in reversed_keys}
    rev_res = predict_one(reversed_sample)
    assert rev_res == dict_res


def test_26_repeated_prediction_is_deterministic(valid_sample):
    """26. Verify repeated prediction with same input produces identical output."""
    res1 = predict_one(valid_sample)
    res2 = predict_one(valid_sample)
    res3 = predict_one(valid_sample)
    assert res1 == res2 == res3


def test_27_prediction_does_not_mutate_original_input(valid_sample):
    """27. Verify prediction does not mutate caller's original input dictionary or DataFrame."""
    input_copy = copy.deepcopy(valid_sample)
    _ = predict_one(valid_sample)
    assert valid_sample == input_copy, "Original input dictionary was modified!"

    df_input = pd.DataFrame([valid_sample])
    df_copy = df_input.copy(deep=True)
    _ = predict_one(df_input)
    pd.testing.assert_frame_equal(df_input, df_copy)


def test_28_model_does_not_retrain_during_prediction(valid_sample):
    """28. Verify model weights and parameters remain strictly unmodified during prediction."""
    model = get_model()
    clf = model.named_steps["classifier"]
    coef_before = clf.coef_.copy()
    intercept_before = clf.intercept_.copy()

    _ = predict_one(valid_sample)
    _ = predict_one(valid_sample)

    assert np.array_equal(clf.coef_, coef_before), "Classifier coef_ modified during inference!"
    assert np.array_equal(clf.intercept_, intercept_before), "Classifier intercept_ modified during inference!"


def test_29_saved_model_reproducibility_after_reload(valid_sample):
    """29. Verify saved model predictions can be reproduced after reloading artifact from disk."""
    model1 = get_model()
    model2 = joblib.load(MODEL_PATH)
    df = pd.DataFrame([valid_sample], columns=EXPECTED_FEATURES)
    p1 = model1.predict(df)
    p2 = model2.predict(df)
    prob1 = model1.predict_proba(df)
    prob2 = model2.predict_proba(df)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_allclose(prob1, prob2, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 6. Dataset Hash, Evaluation Consistency & Model Selection Tests (Items 30 - 34)
# ==============================================================================

def test_30_dataset_sha256_hash_unchanged():
    """30. Verify dataset SHA-256 remains exactly identical to expected hash."""
    assert DATASET_PATH.exists(), f"Dataset file missing at: {DATASET_PATH}"
    hasher = hashlib.sha256()
    with open(DATASET_PATH, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    assert file_hash == EXPECTED_DATASET_HASH, (
        f"Dataset hash mismatch!\nExpected: {EXPECTED_DATASET_HASH}\nGot: {file_hash}"
    )


def test_31_test_metrics_consistent_with_saved_model_evaluation():
    """31. Verify stored test metrics in metrics.json match independent evaluation on test split."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Recreate identical train/test split using Step 4 protocol
    X_train, X_test, y_train, y_test, _ = create_train_test_split(
        dataset_path=str(DATASET_PATH),
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    assert len(X_test) == 1022

    # Run inference with saved pipeline
    model = get_model()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    calc_acc = round(float(accuracy_score(y_test, y_pred)), 4)
    calc_prec = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    calc_rec = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
    calc_f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    calc_roc = round(float(roc_auc_score(y_test, y_prob)), 4)
    calc_pr = round(float(average_precision_score(y_test, y_prob)), 4)

    assert metrics["accuracy"] == calc_acc
    assert metrics["precision"] == calc_prec
    assert metrics["recall"] == calc_rec
    assert metrics["f1"] == calc_f1
    assert metrics["roc_auc"] == calc_roc
    assert metrics["pr_auc"] == calc_pr


def test_32_confusion_matrix_internal_consistency():
    """32. Verify confusion matrix values are internally consistent with stored test metrics."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    # Sum of matrix elements equals total test size (1,022)
    assert tn + fp + fn + tp == 1022
    assert metrics["tn"] == tn
    assert metrics["fp"] == fp
    assert metrics["fn"] == fn
    assert metrics["tp"] == tp

    # Check rates consistency
    calculated_fpr = round(float(fp / (fp + tn)), 4)
    calculated_fnr = round(float(fn / (fn + tp)), 4)
    assert np.isclose(metrics["fpr"], calculated_fpr, atol=1e-4)
    assert np.isclose(metrics["fnr"], calculated_fnr, atol=1e-4)

    # Check accuracy, recall, precision consistency with confusion matrix counts
    matrix_accuracy = round(float((tp + tn) / (tp + tn + fp + fn)), 4)
    matrix_recall = round(float(tp / (tp + fn)), 4)
    matrix_precision = round(float(tp / (tp + fp)), 4)
    assert np.isclose(metrics["accuracy"], matrix_accuracy, atol=1e-4)
    assert np.isclose(metrics["recall"], matrix_recall, atol=1e-4)
    assert np.isclose(metrics["precision"], matrix_precision, atol=1e-4)


def test_33_model_selection_records_selected_variant():
    """33. Verify model_selection.json records Logistic Regression balanced and quarantine confirmation."""
    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        selection = json.load(f)

    assert selection["selected_model"] == "Logistic_Regression"
    assert selection["selected_model_variant"] == "Logistic_Regression_balanced"
    assert selection["selected_model_parameters"]["class_weight"] == "balanced"

    q_conf = selection["test_data_quarantine_confirmation"]
    assert q_conf["test_set_used_for_model_selection"] is False
    assert q_conf["test_set_evaluated_only_once_after_selection"] is True
    assert q_conf["quarantined_test_rows"] == 1022
    assert q_conf["training_rows"] == 4088


def test_34_no_test_refits_final_model():
    """34. Verify final model artifact on disk is read-only and remains identical."""
    disk_model = joblib.load(MODEL_PATH)
    cached_model = get_model()

    disk_coef = disk_model.named_steps["classifier"].coef_
    cached_coef = cached_model.named_steps["classifier"].coef_
    np.testing.assert_array_equal(disk_coef, cached_coef)
