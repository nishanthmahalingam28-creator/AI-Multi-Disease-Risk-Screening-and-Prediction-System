"""Comprehensive Integration Test Suite for Breast Cancer Model Workflow.
Member 1: AI Multi-Disease Risk Screening and Prediction System

Verifies all 20 end-to-end integration and boundary requirements:
1. Dataset exists and loads successfully.
2. Expected 30 model features exist in dataset.
3. feature_schema.json exists and matches canonical feature list.
4. model.joblib exists and loads successfully via joblib.
5. preprocessor.joblib exists and loads successfully via joblib.
6. Model pipeline contains ColumnTransformer + LogisticRegression.
7. predict.py imports successfully.
8. Valid 30-feature input produces a valid prediction.
9. predicted_class is strictly in {0, 1}.
10. predicted_probability is strictly bounded in [0.0, 1.0].
11. Class probabilities are valid and sum approximately to 1.0.
12. Missing feature input is rejected with ValueError.
13. Unknown feature input is rejected with ValueError.
14. Non-numeric input is rejected with ValueError.
15. NaN input is rejected with ValueError.
16. Infinite input is rejected with ValueError.
17. Feature order strictly matches feature_schema.json.
18. metrics.json exists and contains complete final evaluation information.
19. Final model metrics recorded in metrics.json remain unchanged and exact.
20. Original dataset maintains intact shape of exactly 569 rows and 33 columns.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXCLUDED_COLUMNS,
)


def run_integration_tests():
    print("=" * 80)
    print("STEP 10: BREAST CANCER MODEL END-TO-END INTEGRATION TEST SUITE")
    print("=" * 80)

    dataset_path = os.path.abspath(
        os.path.join(current_dir, "..", "..", "..", "datasets", "breast_cancer", "breast.csv")
    )
    schema_path = os.path.join(current_dir, "feature_schema.json")
    model_path = os.path.join(current_dir, "model.joblib")
    prep_path = os.path.join(current_dir, "preprocessor.joblib")
    metrics_path = os.path.join(current_dir, "metrics.json")

    # -------------------------------------------------------------------------
    # Test 1: Dataset exists and can be loaded
    # -------------------------------------------------------------------------
    print("\n[Test 01] Verifying dataset existence and loadability...")
    assert os.path.exists(dataset_path), f"Dataset not found at: {dataset_path}"
    df_raw = pd.read_csv(dataset_path)
    assert not df_raw.empty, "Dataset loaded as empty DataFrame!"
    print(f"  -> PASSED: Dataset loaded successfully ({len(df_raw)} rows).")

    # -------------------------------------------------------------------------
    # Test 2: Expected 30 model features exist
    # -------------------------------------------------------------------------
    print("\n[Test 02] Verifying all 30 model features exist in raw CSV...")
    for feat in FEATURE_COLUMNS:
        assert feat in df_raw.columns, f"Feature '{feat}' missing from dataset!"
    assert len(FEATURE_COLUMNS) == 30, f"Expected 30 features, found {len(FEATURE_COLUMNS)}"
    print("  -> PASSED: All 30 canonical features present in dataset.")

    # -------------------------------------------------------------------------
    # Test 3: feature_schema.json exists and matches canonical feature list
    # -------------------------------------------------------------------------
    print("\n[Test 03] Verifying feature_schema.json matches canonical list...")
    assert os.path.exists(schema_path), f"Schema file not found at: {schema_path}"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["feature_count"] == 30, f"Schema feature_count mismatch: {schema['feature_count']}"
    assert schema["feature_names"] == FEATURE_COLUMNS, "Schema feature_names do not match FEATURE_COLUMNS!"
    assert schema["feature_order"] == FEATURE_COLUMNS, "Schema feature_order does not match FEATURE_COLUMNS!"
    assert schema["disease"] == "breast_cancer", f"Schema disease key incorrect: {schema.get('disease')}"
    print("  -> PASSED: feature_schema.json matches canonical feature list exactly.")

    # -------------------------------------------------------------------------
    # Test 4: model.joblib exists and loads successfully
    # -------------------------------------------------------------------------
    print("\n[Test 04] Verifying model.joblib exists and loads...")
    assert os.path.exists(model_path), f"model.joblib not found at: {model_path}"
    pipeline = joblib.load(model_path)
    assert pipeline is not None, "Loaded pipeline is None!"
    print(f"  -> PASSED: model.joblib loaded successfully ({os.path.getsize(model_path)} bytes).")

    # -------------------------------------------------------------------------
    # Test 5: preprocessor.joblib exists and loads successfully
    # -------------------------------------------------------------------------
    print("\n[Test 05] Verifying preprocessor.joblib exists and loads...")
    assert os.path.exists(prep_path), f"preprocessor.joblib not found at: {prep_path}"
    preprocessor = joblib.load(prep_path)
    assert preprocessor is not None, "Loaded preprocessor is None!"
    print(f"  -> PASSED: preprocessor.joblib loaded successfully ({os.path.getsize(prep_path)} bytes).")

    # -------------------------------------------------------------------------
    # Test 6: Model pipeline contains preprocessing and Logistic Regression
    # -------------------------------------------------------------------------
    print("\n[Test 06] Verifying pipeline structure (ColumnTransformer + LogisticRegression)...")
    assert isinstance(pipeline, Pipeline), "model.joblib is not a scikit-learn Pipeline instance!"
    assert "preprocessor" in pipeline.named_steps, "Pipeline missing 'preprocessor' step!"
    assert "classifier" in pipeline.named_steps, "Pipeline missing 'classifier' step!"
    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer), (
        "Pipeline preprocessor step is not a ColumnTransformer!"
    )
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression), (
        "Pipeline classifier step is not a LogisticRegression!"
    )
    assert isinstance(preprocessor, ColumnTransformer), (
        "preprocessor.joblib is not a ColumnTransformer!"
    )
    print("  -> PASSED: Pipeline architecture verified as [ColumnTransformer -> LogisticRegression].")

    # -------------------------------------------------------------------------
    # Test 7: predict.py imports successfully
    # -------------------------------------------------------------------------
    print("\n[Test 07] Verifying predict.py imports successfully...")
    try:
        import predict
        from predict import predict as run_prediction, format_input_dataframe, validate_input
        print("  -> PASSED: predict.py imported successfully.")
    except Exception as e:
        raise AssertionError(f"Failed to import predict.py: {e}")

    # Prepare real row sample (without using target label to influence inference)
    sample_input = df_raw.iloc[0][FEATURE_COLUMNS].to_dict()

    # -------------------------------------------------------------------------
    # Test 8: Valid 30-feature input produces a valid prediction
    # -------------------------------------------------------------------------
    print("\n[Test 08] Verifying valid 30-feature input produces prediction...")
    res = run_prediction(sample_input)
    assert isinstance(res, dict), f"Prediction output is not a dict: {type(res)}"
    assert res.get("status") == "success", f"Prediction status != 'success': {res.get('status')}"
    assert res.get("disease") == "breast_cancer", f"Disease key != 'breast_cancer': {res.get('disease')}"
    print("  -> PASSED: Valid input produces structured success dictionary.")

    # -------------------------------------------------------------------------
    # Test 9: predicted_class is 0 or 1
    # -------------------------------------------------------------------------
    print("\n[Test 09] Verifying predicted_class is strictly 0 or 1...")
    p_class = res.get("predicted_class")
    assert p_class in [0, 1], f"predicted_class {p_class} not in {{0, 1}}!"
    print(f"  -> PASSED: predicted_class is {p_class}.")

    # -------------------------------------------------------------------------
    # Test 10: predicted_probability is between 0 and 1
    # -------------------------------------------------------------------------
    print("\n[Test 10] Verifying predicted_probability is in [0.0, 1.0]...")
    p_prob = res.get("predicted_probability")
    assert isinstance(p_prob, float), f"predicted_probability not float: {type(p_prob)}"
    assert 0.0 <= p_prob <= 1.0, f"predicted_probability {p_prob} out of bounds [0, 1]!"
    print(f"  -> PASSED: predicted_probability is {p_prob:.6f}.")

    # -------------------------------------------------------------------------
    # Test 11: Class probabilities are valid and sum approximately to 1
    # -------------------------------------------------------------------------
    print("\n[Test 11] Verifying class probabilities bounds and sum to 1.0...")
    c_probs = res.get("class_probabilities")
    assert isinstance(c_probs, dict), "class_probabilities not a dict!"
    assert "0" in c_probs and "1" in c_probs, "Keys '0' and '1' must be in class_probabilities!"
    assert 0.0 <= c_probs["0"] <= 1.0, f"p(0) {c_probs['0']} out of bounds!"
    assert 0.0 <= c_probs["1"] <= 1.0, f"p(1) {c_probs['1']} out of bounds!"
    assert np.isclose(c_probs["0"] + c_probs["1"], 1.0, atol=1e-4), "Class probabilities do not sum to 1.0!"
    print(f"  -> PASSED: Class probabilities valid (p0={c_probs['0']:.4f}, p1={c_probs['1']:.4f}).")

    # -------------------------------------------------------------------------
    # Test 12: Missing feature input is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 12] Verifying missing feature input is rejected...")
    bad_sample_missing = sample_input.copy()
    bad_sample_missing.pop("radius_mean")
    rejected_missing = False
    try:
        run_prediction(bad_sample_missing)
    except ValueError:
        rejected_missing = True
    assert rejected_missing, "Failed to reject input with missing feature!"
    print("  -> PASSED: Missing feature rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 13: Unknown feature input is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 13] Verifying unknown feature input is rejected...")
    bad_sample_unknown = sample_input.copy()
    bad_sample_unknown["unauthorized_feature_123"] = 42.0
    rejected_unknown = False
    try:
        run_prediction(bad_sample_unknown)
    except ValueError:
        rejected_unknown = True
    assert rejected_unknown, "Failed to reject input with unknown feature!"
    print("  -> PASSED: Unknown feature rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 14: Non-numeric input is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 14] Verifying non-numeric input is rejected...")
    bad_sample_non_num = sample_input.copy()
    bad_sample_non_num["texture_mean"] = "non_numeric_str"
    rejected_non_num = False
    try:
        run_prediction(bad_sample_non_num)
    except ValueError:
        rejected_non_num = True
    assert rejected_non_num, "Failed to reject non-numeric input!"
    print("  -> PASSED: Non-numeric input rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 15: NaN input is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 15] Verifying NaN input is rejected...")
    bad_sample_nan = sample_input.copy()
    bad_sample_nan["area_mean"] = float("nan")
    rejected_nan = False
    try:
        run_prediction(bad_sample_nan)
    except ValueError:
        rejected_nan = True
    assert rejected_nan, "Failed to reject NaN input!"
    print("  -> PASSED: NaN input rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 16: Infinite input is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 16] Verifying Infinite input is rejected...")
    bad_sample_inf = sample_input.copy()
    bad_sample_inf["perimeter_mean"] = float("inf")
    rejected_inf = False
    try:
        run_prediction(bad_sample_inf)
    except ValueError:
        rejected_inf = True
    assert rejected_inf, "Failed to reject infinite input!"
    print("  -> PASSED: Infinite input rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 17: Feature order matches feature_schema.json
    # -------------------------------------------------------------------------
    print("\n[Test 17] Verifying DataFrame construction feature ordering...")
    reversed_dict = {k: sample_input[k] for k in reversed(list(sample_input.keys()))}
    df_constructed = format_input_dataframe(reversed_dict)
    assert list(df_constructed.columns) == schema["feature_order"], (
        "Constructed DataFrame column order does not match feature_schema.json!"
    )
    print("  -> PASSED: Feature order strictly conforms to feature_schema.json.")

    # -------------------------------------------------------------------------
    # Test 18: metrics.json exists and contains final evaluation information
    # -------------------------------------------------------------------------
    print("\n[Test 18] Verifying metrics.json exists and contains evaluation info...")
    assert os.path.exists(metrics_path), f"metrics.json not found at: {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    required_keys = [
        "disease", "dataset", "train_samples", "test_samples", "candidate_models",
        "selection_criteria", "selected_model", "selected_model_parameters",
        "cross_validation_metrics", "test_metrics", "confusion_matrix",
        "false_positives", "false_negatives", "random_state", "test_size"
    ]
    for rk in required_keys:
        assert rk in metrics_data, f"metrics.json missing required key: '{rk}'"
    print("  -> PASSED: metrics.json exists with all required metadata fields.")

    # -------------------------------------------------------------------------
    # Test 19: Final model metrics recorded in metrics.json remain unchanged
    # -------------------------------------------------------------------------
    print("\n[Test 19] Verifying recorded metrics fidelity in metrics.json...")
    tm = metrics_data["test_metrics"]
    cm = metrics_data["confusion_matrix"]
    assert tm["accuracy"] == 0.9649, f"Unexpected accuracy: {tm['accuracy']}"
    assert tm["precision"] == 0.9750, f"Unexpected precision: {tm['precision']}"
    assert tm["recall_sensitivity"] == 0.9286, f"Unexpected recall: {tm['recall_sensitivity']}"
    assert tm["f1_score"] == 0.9512, f"Unexpected f1_score: {tm['f1_score']}"
    assert tm["roc_auc"] == 0.9960, f"Unexpected roc_auc: {tm['roc_auc']}"
    assert tm["pr_auc"] == 0.9943, f"Unexpected pr_auc: {tm['pr_auc']}"
    assert cm["true_negatives"] == 71, f"Unexpected true_negatives: {cm['true_negatives']}"
    assert cm["false_positives"] == 1, f"Unexpected false_positives: {cm['false_positives']}"
    assert cm["false_negatives"] == 3, f"Unexpected false_negatives: {cm['false_negatives']}"
    assert cm["true_positives"] == 39, f"Unexpected true_positives: {cm['true_positives']}"
    print("  -> PASSED: Final evaluation metrics verified as exact and uncorrupted.")

    # -------------------------------------------------------------------------
    # Test 20: Original dataset still has expected 569 rows and 33 raw columns
    # -------------------------------------------------------------------------
    print("\n[Test 20] Verifying dataset integrity (569 rows, 33 columns)...")
    assert df_raw.shape == (569, 33), f"Dataset shape modified! Expected (569, 33), got {df_raw.shape}"
    assert TARGET_COLUMN in df_raw.columns, f"Target '{TARGET_COLUMN}' missing from raw CSV!"
    for excl in EXCLUDED_COLUMNS:
        assert excl in df_raw.columns, f"Excluded column '{excl}' missing from raw CSV!"
    print(f"  -> PASSED: Original CSV verified intact with {df_raw.shape[0]} rows and {df_raw.shape[1]} columns.")

    print("\n" + "=" * 80)
    print("ALL 20 INTEGRATION TESTS PASSED SUCCESSFULLY (20/20).")
    print("CONFIRMATION: ZERO RE-TRAINING PERFORMED DURING TESTS.")
    print("=" * 80)


if __name__ == "__main__":
    run_integration_tests()
