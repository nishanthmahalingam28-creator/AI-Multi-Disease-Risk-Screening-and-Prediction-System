"""
Automated Test Suite for Diabetes Prediction Model
AI Multi-Disease Risk Screening and Prediction System

Validates:
1. Model artifact existence and loading via joblib.
2. Pipeline structure, components, and preprocessing stages.
3. Feature schema integrity, feature ordering, and target definition.
4. Input validation (missing, extra, non-numeric, NaN, infinite, type checks).
5. Output format, probability calibration bounds [0, 1], and neutral descriptions.
6. Non-mutation of input containers and model parameters (no retraining).
7. Metrics and model selection report integrity (sample counts, quarantine confirmation).
8. Batch inference on raw dataset records.
"""

import os
import sys
import json
import copy
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Path configurations
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIABETES_DIR = os.path.dirname(TESTS_DIR)
WORKSPACE_ROOT = os.path.abspath(os.path.join(DIABETES_DIR, "..", "..", ".."))

MODEL_PATH = os.path.join(DIABETES_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(DIABETES_DIR, "feature_schema.json")
METRICS_PATH = os.path.join(DIABETES_DIR, "metrics.json")
SELECTION_PATH = os.path.join(DIABETES_DIR, "model_selection.json")
DATASET_PATH = os.path.join(WORKSPACE_ROOT, "datasets", "diabetes", "diabetes.csv")

# Ensure diabetes directory is in sys.path
if DIABETES_DIR not in sys.path:
    sys.path.insert(0, DIABETES_DIR)

from predict import predict_diabetes, predict, get_model, get_feature_schema, EXPECTED_FEATURES


@pytest.fixture
def valid_input():
    """Provides a canonical valid single-sample input dictionary."""
    return {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }


@pytest.fixture
def valid_input_class_0():
    """Provides a second canonical valid input dictionary."""
    return {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31
    }


# ==============================================================================
# 1. Artifact Existence & Pipeline Architecture Tests
# ==============================================================================

def test_model_artifact_exists():
    """Test 1: Verify model.joblib exists at the expected path."""
    assert os.path.exists(MODEL_PATH), f"Model artifact missing at: {MODEL_PATH}"
    assert os.path.getsize(MODEL_PATH) > 0, "Model artifact file is empty"


def test_model_artifact_loads_successfully():
    """Test 2: Verify model artifact can be loaded using joblib."""
    loaded_model = joblib.load(MODEL_PATH)
    assert loaded_model is not None, "Failed to load model artifact"


def test_loaded_artifact_is_sklearn_pipeline():
    """Test 3: Verify the loaded artifact is a scikit-learn Pipeline instance."""
    loaded_model = joblib.load(MODEL_PATH)
    assert isinstance(loaded_model, Pipeline), f"Expected Pipeline, got {type(loaded_model).__name__}"


def test_pipeline_contains_expected_stages():
    """Test 4: Verify the pipeline contains the required preprocessing and model stages."""
    loaded_model = joblib.load(MODEL_PATH)
    step_names = [step[0] for step in loaded_model.steps]
    expected_steps = ["zero_to_nan", "imputer", "scaler", "classifier"]
    assert step_names == expected_steps, f"Pipeline steps {step_names} != {expected_steps}"


# ==============================================================================
# 2. Schema and Feature Definition Tests
# ==============================================================================

def test_feature_schema_exists():
    """Test 5: Verify feature_schema.json exists."""
    assert os.path.exists(SCHEMA_PATH), f"Feature schema missing at: {SCHEMA_PATH}"


def test_feature_schema_contains_exact_features_in_order():
    """Test 6: Verify schema contains the exact 8 features in the canonical order."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert "exact_feature_names_in_order" in schema, "Key 'exact_feature_names_in_order' missing from schema"
    assert schema["exact_feature_names_in_order"] == EXPECTED_FEATURES, (
        f"Feature order mismatch: {schema['exact_feature_names_in_order']} != {EXPECTED_FEATURES}"
    )
    assert schema["feature_count"] == 8, f"Feature count != 8, got {schema.get('feature_count')}"


def test_target_variable_is_outcome():
    """Test 7: Verify target variable is 'Outcome'."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["target_column"] == "Outcome", f"Target column is '{schema.get('target_column')}', expected 'Outcome'"


# ==============================================================================
# 3. Prediction Module & Core Inference Tests
# ==============================================================================

def test_prediction_module_importable():
    """Test 8: Verify prediction functions are callable and available."""
    assert callable(predict_diabetes), "predict_diabetes is not callable"
    assert callable(predict), "predict alias is not callable"


def test_valid_input_prediction_success(valid_input):
    """Test 9: Verify a valid input dictionary produces a successful prediction result."""
    result = predict_diabetes(valid_input)
    assert isinstance(result, dict), "Result is not a dictionary"
    assert result["status"] == "success", f"Prediction status != 'success': {result}"


def test_predicted_class_is_binary(valid_input, valid_input_class_0):
    """Test 10: Verify predicted_class is strictly 0 or 1."""
    res1 = predict_diabetes(valid_input)
    res2 = predict_diabetes(valid_input_class_0)
    assert res1["predicted_class"] in [0, 1], f"predicted_class not in [0, 1]: {res1['predicted_class']}"
    assert res2["predicted_class"] in [0, 1], f"predicted_class not in [0, 1]: {res2['predicted_class']}"


def test_predicted_probability_bounded_in_unit_interval(valid_input):
    """Test 11: Verify predicted_probability is bounded between 0.0 and 1.0."""
    result = predict_diabetes(valid_input)
    prob = result["predicted_probability"]
    assert isinstance(prob, (float, np.floating)), f"Probability is not float: {type(prob)}"
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds [0.0, 1.0]"


def test_class_probabilities_exist_for_both_classes(valid_input):
    """Test 12: Verify class_probabilities dict contains keys '0' and '1'."""
    result = predict_diabetes(valid_input)
    assert "class_probabilities" in result, "Key 'class_probabilities' missing"
    assert "0" in result["class_probabilities"], "Key '0' missing in class_probabilities"
    assert "1" in result["class_probabilities"], "Key '1' missing in class_probabilities"


def test_class_probabilities_sum_to_one(valid_input):
    """Test 13: Verify probabilities for Class 0 and Class 1 sum to 1.0."""
    result = predict_diabetes(valid_input)
    cp = result["class_probabilities"]
    prob_sum = cp["0"] + cp["1"]
    assert abs(prob_sum - 1.0) < 1e-3, f"Class probabilities do not sum to 1.0: {prob_sum}"


def test_predicted_class_corresponds_to_higher_probability(valid_input, valid_input_class_0):
    """Test 14: Verify predicted class corresponds to the class with larger probability."""
    for sample in [valid_input, valid_input_class_0]:
        res = predict_diabetes(sample)
        pred_cls = str(res["predicted_class"])
        other_cls = "1" if pred_cls == "0" else "0"
        assert res["class_probabilities"][pred_cls] >= res["class_probabilities"][other_cls], (
            f"Predicted class {pred_cls} does not have highest probability: {res['class_probabilities']}"
        )


# ==============================================================================
# 4. Strict Input Validation & Boundary Rejection Tests
# ==============================================================================

def test_missing_feature_raises_value_error(valid_input):
    """Test 15: Verify missing required feature raises ValueError."""
    for col in EXPECTED_FEATURES:
        incomplete = valid_input.copy()
        del incomplete[col]
        with pytest.raises(ValueError, match="Missing required feature"):
            predict_diabetes(incomplete)


def test_extra_feature_raises_value_error(valid_input):
    """Test 16: Verify unexpected/extra feature raises ValueError."""
    extra = valid_input.copy()
    extra["UnexpectedColumn"] = 100
    with pytest.raises(ValueError, match="Unexpected extra feature"):
        predict_diabetes(extra)


def test_non_numeric_feature_value_raises_value_error(valid_input):
    """Test 17: Verify non-numeric values (strings, booleans, None) raise ValueError."""
    # Test string
    bad_str = valid_input.copy()
    bad_str["Glucose"] = "invalid_string"
    with pytest.raises(ValueError, match="Non-numeric value"):
        predict_diabetes(bad_str)

    # Test boolean
    bad_bool = valid_input.copy()
    bad_bool["Age"] = True
    with pytest.raises(ValueError, match="Non-numeric value"):
        predict_diabetes(bad_bool)

    # Test None
    bad_none = valid_input.copy()
    bad_none["BMI"] = None
    with pytest.raises(ValueError, match="Non-numeric value"):
        predict_diabetes(bad_none)


def test_nan_input_raises_value_error(valid_input):
    """Test 18: Verify NaN input provided by caller raises ValueError."""
    bad_nan = valid_input.copy()
    bad_nan["Glucose"] = float("nan")
    with pytest.raises(ValueError, match="Invalid non-finite or NaN"):
        predict_diabetes(bad_nan)


def test_infinite_input_raises_value_error(valid_input):
    """Test 19: Verify infinite input provided by caller raises ValueError."""
    bad_inf = valid_input.copy()
    bad_inf["BMI"] = float("inf")
    with pytest.raises(ValueError, match="Invalid non-finite or NaN"):
        predict_diabetes(bad_inf)


# ==============================================================================
# 5. Tabular Input Container Compatibility Tests
# ==============================================================================

def test_single_row_dataframe_accepted(valid_input):
    """Test 20: Verify a valid single-row pandas DataFrame is accepted."""
    df_single = pd.DataFrame([valid_input])
    result = predict_diabetes(df_single)
    assert result["status"] == "success"
    assert result["predicted_class"] in [0, 1]


def test_pandas_series_accepted(valid_input):
    """Test 21: Verify a pandas Series with required features is accepted."""
    series_input = pd.Series(valid_input)
    result = predict_diabetes(series_input)
    assert result["status"] == "success"
    assert result["predicted_class"] in [0, 1]


def test_multi_row_dataframe_rejected(valid_input):
    """Test 22: Verify multi-row DataFrame is rejected with ValueError."""
    df_multi = pd.DataFrame([valid_input, valid_input])
    with pytest.raises(ValueError, match="Expected a single sample row"):
        predict_diabetes(df_multi)


# ==============================================================================
# 6. Immutability & Non-Mutation Tests
# ==============================================================================

def test_prediction_does_not_modify_input_dict(valid_input):
    """Test 23: Verify prediction does not mutate the caller's input dictionary."""
    orig_copy = copy.deepcopy(valid_input)
    _ = predict_diabetes(valid_input)
    assert valid_input == orig_copy, "Input dictionary was modified during prediction!"


def test_prediction_does_not_modify_input_dataframe(valid_input):
    """Test 24: Verify prediction does not mutate the caller's input DataFrame."""
    df_input = pd.DataFrame([valid_input])
    df_copy = df_input.copy(deep=True)
    _ = predict_diabetes(df_input)
    pd.testing.assert_frame_equal(df_input, df_copy)


def test_output_structure_fields(valid_input):
    """Test 25: Verify prediction output contains all expected standard fields."""
    res = predict_diabetes(valid_input)
    expected_keys = [
        "status",
        "model_name",
        "predicted_class",
        "predicted_probability",
        "class_probabilities",
        "screening_estimate"
    ]
    for k in expected_keys:
        assert k in res, f"Expected key '{k}' missing from prediction response"


def test_model_artifact_not_retrained_during_prediction(valid_input):
    """Test 26: Verify model parameters remain identical before and after inference (no retraining)."""
    model = get_model()
    # Record parameters of classifier and imputer
    classifier = model.named_steps["classifier"]
    imputer = model.named_steps["imputer"]

    coef_before = classifier.coef_.copy()
    intercept_before = classifier.intercept_.copy()
    imputer_stats_before = imputer.statistics_.copy()

    # Run predictions multiple times
    for _ in range(5):
        _ = predict_diabetes(valid_input)

    assert np.array_equal(classifier.coef_, coef_before), "Classifier coef_ modified during inference!"
    assert np.array_equal(classifier.intercept_, intercept_before), "Classifier intercept_ modified during inference!"
    assert np.array_equal(imputer.statistics_, imputer_stats_before), "Imputer statistics_ modified during inference!"


# ==============================================================================
# 7. Metrics and Model Selection Report Verification Tests
# ==============================================================================

def test_metrics_file_exists_and_contains_required_fields():
    """Test 27: Verify metrics.json exists and contains complete evaluation keys."""
    assert os.path.exists(METRICS_PATH), f"metrics.json missing at: {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "selected_model" in metrics, "Missing 'selected_model' in metrics.json"
    assert "test_metrics" in metrics, "Missing 'test_metrics' in metrics.json"

    tm = metrics["test_metrics"]
    required_metric_keys = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "confusion_matrix"
    ]
    for k in required_metric_keys:
        assert k in tm, f"Metric '{k}' missing in test_metrics"


def test_confusion_matrix_sums_to_154():
    """Test 28: Verify confusion matrix values in metrics.json sum to 154."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = metrics["test_metrics"]["confusion_matrix"]
    cm_sum = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
    assert cm_sum == 154, f"Confusion matrix total {cm_sum} != 154"


def test_confusion_matrix_class_marginal_totals():
    """Test 29: Verify TP + FN == actual Class 1 (54) and TN + FP == actual Class 0 (100)."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = metrics["test_metrics"]["confusion_matrix"]
    actual_class_1 = cm["true_positives"] + cm["false_negatives"]
    actual_class_0 = cm["true_negatives"] + cm["false_positives"]

    assert actual_class_1 == 54, f"Actual Class 1 total {actual_class_1} != 54"
    assert actual_class_0 == 100, f"Actual Class 0 total {actual_class_0} != 100"


def test_model_selection_json_content():
    """Test 30: Verify model_selection.json confirms quarantined test set and model selection."""
    assert os.path.exists(SELECTION_PATH), f"model_selection.json missing at: {SELECTION_PATH}"
    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        sel = json.load(f)

    assert "selected_model" in sel, "Missing 'selected_model' in model_selection.json"
    assert sel["selected_model"] == "Logistic_Regression"
    assert sel["quarantine_confirmation"]["test_set_used_during_selection"] is False
    assert sel["quarantine_confirmation"]["test_rows_quarantined"] == 154


# ==============================================================================
# 8. Batch Inference on Real Dataset Records Tests
# ==============================================================================

def test_batch_prediction_on_real_dataset_rows():
    """Test 31: Test batch prediction on 5 actual rows loaded from diabetes.csv."""
    assert os.path.exists(DATASET_PATH), f"Dataset file missing at: {DATASET_PATH}"
    raw_df = pd.read_csv(DATASET_PATH)
    sample_batch = raw_df[EXPECTED_FEATURES].iloc[:5]

    model = get_model()
    batch_preds = model.predict(sample_batch)
    batch_probas = model.predict_proba(sample_batch)

    assert len(batch_preds) == 5, f"Batch prediction length {len(batch_preds)} != 5"
    assert len(batch_probas) == 5, f"Batch probabilities length {len(batch_probas)} != 5"


def test_batch_prediction_properties():
    """Test 32: Test batch prediction probabilities are finite and bounded in [0, 1]."""
    raw_df = pd.read_csv(DATASET_PATH)
    sample_batch = raw_df[EXPECTED_FEATURES].iloc[:10]

    model = get_model()
    batch_preds = model.predict(sample_batch)
    batch_probas = model.predict_proba(sample_batch)

    # Check finite
    assert not np.isnan(batch_preds).any(), "NaN found in batch predictions"
    assert not np.isnan(batch_probas).any(), "NaN found in batch probabilities"
    assert not np.isinf(batch_probas).any(), "Inf found in batch probabilities"

    # Check bounds
    assert (batch_probas >= 0.0).all() and (batch_probas <= 1.0).all(), "Batch probabilities out of [0, 1] range"
    # Check probabilities sum to 1
    row_sums = batch_probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Batch probabilities do not sum to 1.0 per row"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
