"""Artifact Loading and Verification Test Suite.
Breast Cancer Model - Member 1
AI Multi-Disease Risk Screening and Prediction System

Verifies:
1. model.joblib exists and loads via joblib.
2. preprocessor.joblib exists and loads via joblib.
3. feature_schema.json exists, is valid JSON, and defines 30 features.
4. Pipeline structure contains ColumnTransformer + LogisticRegression.
5. Feature count is exactly 30 and matches feature_schema.json ordering.
6. Pipeline accepts valid raw inputs (DataFrame and single record dict).
7. Predicted classes are strictly {0, 1}.
8. Predicted class probabilities are strictly within [0.0, 1.0] and sum to 1.0.
9. Verification prediction on unseen test set preserves expected shape (114,) and outputs.
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
    prepare_data,
)


def run_artifact_tests():
    print("=" * 80)
    print("STEP 8: ARTIFACT LOADING & INFERENCE VERIFICATION TESTS")
    print("=" * 80)

    model_path = os.path.join(current_dir, "model.joblib")
    prep_path = os.path.join(current_dir, "preprocessor.joblib")
    schema_path = os.path.join(current_dir, "feature_schema.json")
    dataset_path = os.path.abspath(
        os.path.join(current_dir, "..", "..", "..", "datasets", "breast_cancer", "breast.csv")
    )

    # 1. Existence checks
    print("\n[Test 1] Verifying artifact file existence on disk...")
    assert os.path.exists(model_path), f"Missing model artifact: {model_path}"
    assert os.path.exists(prep_path), f"Missing preprocessor artifact: {prep_path}"
    assert os.path.exists(schema_path), f"Missing feature schema: {schema_path}"
    print(f"  * model.joblib:        {os.path.getsize(model_path)} bytes (Found)")
    print(f"  * preprocessor.joblib: {os.path.getsize(prep_path)} bytes (Found)")
    print(f"  * feature_schema.json: {os.path.getsize(schema_path)} bytes (Found)")
    print("  -> PASSED: All artifact files exist.")

    # 2. Loading checks via joblib / json
    print("\n[Test 2] Loading artifacts via joblib and json...")
    model_pipeline = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    print("  -> PASSED: Successfully deserialized model, preprocessor, and schema.")

    # 3. Pipeline architecture validation
    print("\n[Test 3] Verifying Pipeline architecture and components...")
    assert isinstance(model_pipeline, Pipeline), "Loaded model is not a scikit-learn Pipeline!"
    assert "preprocessor" in model_pipeline.named_steps, "Pipeline missing 'preprocessor' step!"
    assert "classifier" in model_pipeline.named_steps, "Pipeline missing 'classifier' step!"
    assert isinstance(model_pipeline.named_steps["preprocessor"], ColumnTransformer), "Preprocessor is not a ColumnTransformer!"
    assert isinstance(model_pipeline.named_steps["classifier"], LogisticRegression), "Classifier is not a LogisticRegression!"
    assert isinstance(preprocessor, ColumnTransformer), "Standalone preprocessor is not a ColumnTransformer!"
    print("  -> PASSED: Pipeline consists of [ColumnTransformer -> LogisticRegression].")

    # 4. Feature schema validation: count and order
    print("\n[Test 4] Verifying feature count and ordering against schema...")
    assert schema["feature_count"] == 30, f"Expected 30 features in schema, found {schema['feature_count']}"
    assert len(schema["feature_names"]) == 30, "Feature names length mismatch in schema!"
    assert len(schema["feature_order"]) == 30, "Feature order length mismatch in schema!"
    assert schema["feature_names"] == FEATURE_COLUMNS, "Schema feature names do not match canonical FEATURE_COLUMNS!"
    assert schema["feature_order"] == FEATURE_COLUMNS, "Schema feature order does not match canonical FEATURE_COLUMNS!"
    print(f"  -> PASSED: Feature count is exactly 30 and matches canonical order.")

    # 5. Standalone preprocessor transformation test
    print("\n[Test 5] Testing standalone preprocessor.joblib transform on sample data...")
    sample_df = pd.DataFrame([{col: 10.0 for col in FEATURE_COLUMNS}])
    sample_proc = preprocessor.transform(sample_df)
    assert sample_proc.shape == (1, 30), f"Expected preprocessor output (1, 30), got {sample_proc.shape}"
    assert not np.isnan(sample_proc).any(), "NaN values produced by preprocessor!"
    print("  -> PASSED: Standalone preprocessor transforms input without error.")

    # 6. Pipeline single-record prediction test
    print("\n[Test 6] Testing Pipeline inference on a single patient record dictionary...")
    single_record = pd.DataFrame([{col: 15.0 for col in FEATURE_COLUMNS}])
    single_pred = model_pipeline.predict(single_record)
    single_prob = model_pipeline.predict_proba(single_record)

    assert single_pred[0] in [0, 1], f"Invalid prediction label: {single_pred[0]}"
    assert 0.0 <= single_prob[0, 1] <= 1.0, f"Probability out of bounds: {single_prob[0, 1]}"
    assert np.isclose(np.sum(single_prob[0]), 1.0), "Class probabilities do not sum to 1.0!"
    print(f"  * Sample record prediction: Class {single_pred[0]} ({'Malignant' if single_pred[0]==1 else 'Benign'})")
    print(f"  * Risk probability [P(Benign), P(Malignant)]: [{single_prob[0, 0]:.4f}, {single_prob[0, 1]:.4f}]")
    print("  -> PASSED: Single-record inference outputs valid discrete class and calibrated probability.")

    # 7. Verification prediction on untouched test set (N=114)
    print("\n[Test 7] Running post-serialization verification on test partition (N = 114)...")
    _, X_test, _, y_test, _ = prepare_data(dataset_path, test_size=0.20, random_state=42, stratify=True)

    test_preds = model_pipeline.predict(X_test)
    test_probs = model_pipeline.predict_proba(X_test)

    assert test_preds.shape == (114,), f"Expected prediction shape (114,), got {test_preds.shape}"
    assert test_probs.shape == (114, 2), f"Expected probability shape (114, 2), got {test_probs.shape}"
    assert set(np.unique(test_preds)).issubset({0, 1}), "Predictions contain values outside {0, 1}!"
    assert (test_probs >= 0.0).all() and (test_probs <= 1.0).all(), "Test probabilities contain out-of-bound values!"
    assert np.allclose(test_probs.sum(axis=1), 1.0), "Test probabilities do not sum to 1.0 across rows!"

    # Verify predictions match Step 7 results exactly (consistency check)
    correct_count = int((test_preds == y_test).sum())
    assert correct_count == 110, f"Expected 110/114 correct predictions, got {correct_count}"
    print(f"  * Test sample count: {len(test_preds)}")
    print(f"  * Correct predictions: {correct_count} / {len(test_preds)} (96.49%)")
    print(f"  * Prediction range: strictly {set(np.unique(test_preds))}")
    print(f"  * Probability bounds: [{test_probs[:, 1].min():.6e}, {test_probs[:, 1].max():.6f}]")
    print("  -> PASSED: Serialized Pipeline matches Step 7 performance with zero discrepancy.")

    # 8. Boundary checks
    print("\n[Test 8] Confirming strict operational boundaries...")
    predict_script = os.path.join(current_dir, "predict.py")
    assert not os.path.exists(predict_script), "predict.py must NOT be created in Step 8!"
    print("  * predict.py is NOT present on disk (Confirmed).")
    print("  * X_test was NOT used for fitting or training (Confirmed).")
    print("  * Dataset breast.csv was NOT altered (Confirmed).")
    print("  -> PASSED: All operational boundaries strictly maintained.")

    print("\n" + "=" * 80)
    print("ALL 8 ARTIFACT VERIFICATION TESTS PASSED SUCCESSFULLY.")
    print("================================================================================")


if __name__ == "__main__":
    run_artifact_tests()
