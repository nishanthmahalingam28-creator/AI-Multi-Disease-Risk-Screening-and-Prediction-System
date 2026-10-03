"""Inference Test Suite for Breast Cancer Prediction Module.
Member 1: AI Multi-Disease Risk Screening and Prediction System

Executes 10 required inference verification tests:
Test 1: predict.py imports successfully.
Test 2: model.joblib loads successfully.
Test 3: valid 30-feature input produces a prediction.
Test 4: predicted_class is either 0 or 1.
Test 5: predicted_probability is between 0 and 1.
Test 6: class probabilities are between 0 and 1 and sum approximately to 1.
Test 7: missing feature is rejected.
Test 8: unknown feature is rejected.
Test 9: non-numeric feature is rejected.
Test 10: feature order follows feature_schema.json.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def run_tests():
    print("=" * 80)
    print("STEP 9: PREDICTION MODULE VERIFICATION TEST SUITE")
    print("=" * 80)

    dataset_path = os.path.abspath(
        os.path.join(current_dir, "..", "..", "..", "datasets", "breast_cancer", "breast.csv")
    )
    raw_df = pd.read_csv(dataset_path)

    # -------------------------------------------------------------------------
    # Test 1: predict.py imports successfully
    # -------------------------------------------------------------------------
    print("\n[Test 1] Testing predict.py module import...")
    try:
        import predict
        from predict import (
            predict as run_predict,
            validate_input,
            get_model,
            get_feature_schema,
            format_input_dataframe,
        )
        print("  -> PASSED: predict.py imported successfully.")
    except Exception as e:
        raise AssertionError(f"Test 1 Failed: Could not import predict.py: {e}")

    # -------------------------------------------------------------------------
    # Test 2: model.joblib loads successfully
    # -------------------------------------------------------------------------
    print("\n[Test 2] Testing model.joblib loading via get_model()...")
    model = get_model()
    assert model is not None, "Loaded model is None!"
    assert hasattr(model, "predict"), "Loaded model has no predict method!"
    assert hasattr(model, "predict_proba"), "Loaded model has no predict_proba method!"
    print("  -> PASSED: model.joblib loaded successfully.")

    # Prepare real row sample (without using target label to influence inference)
    schema = get_feature_schema()
    canonical_features = schema["feature_names"]
    real_sample = raw_df.iloc[0][canonical_features].to_dict()

    # -------------------------------------------------------------------------
    # Test 3: Valid 30-feature input produces a prediction
    # -------------------------------------------------------------------------
    print("\n[Test 3] Testing valid 30-feature input prediction...")
    result = run_predict(real_sample)
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    assert result.get("status") == "success", f"Expected status 'success', got {result.get('status')}"
    assert result.get("disease") == "breast_cancer", f"Expected disease 'breast_cancer', got {result.get('disease')}"
    print("  -> PASSED: Valid 30-feature input produces structured prediction result.")

    # -------------------------------------------------------------------------
    # Test 4: predicted_class is either 0 or 1
    # -------------------------------------------------------------------------
    print("\n[Test 4] Testing that predicted_class is strictly in {0, 1}...")
    pred_class = result.get("predicted_class")
    assert pred_class in [0, 1], f"predicted_class must be 0 or 1, got {pred_class}"
    print(f"  -> PASSED: predicted_class is valid ({pred_class}).")

    # -------------------------------------------------------------------------
    # Test 5: predicted_probability is between 0 and 1
    # -------------------------------------------------------------------------
    print("\n[Test 5] Testing predicted_probability is bounded in [0.0, 1.0]...")
    pred_prob = result.get("predicted_probability")
    assert isinstance(pred_prob, float), f"predicted_probability must be float, got {type(pred_prob)}"
    assert 0.0 <= pred_prob <= 1.0, f"predicted_probability {pred_prob} out of bounds [0, 1]!"
    print(f"  -> PASSED: predicted_probability is bounded ({pred_prob:.6f}).")

    # -------------------------------------------------------------------------
    # Test 6: Class probabilities are between 0 and 1 and sum approximately to 1
    # -------------------------------------------------------------------------
    print("\n[Test 6] Testing class_probabilities sum to 1.0 and bounds...")
    class_probs = result.get("class_probabilities")
    assert isinstance(class_probs, dict), "class_probabilities must be a dictionary!"
    assert "0" in class_probs and "1" in class_probs, "class_probabilities must contain keys '0' and '1'!"
    p0 = class_probs["0"]
    p1 = class_probs["1"]
    assert 0.0 <= p0 <= 1.0, f"p(0) {p0} out of bounds [0, 1]!"
    assert 0.0 <= p1 <= 1.0, f"p(1) {p1} out of bounds [0, 1]!"
    prob_sum = p0 + p1
    assert np.isclose(prob_sum, 1.0, atol=1e-4), f"Probabilities do not sum to 1.0: {prob_sum}"
    print(f"  -> PASSED: Class probabilities p(0)={p0:.6f}, p(1)={p1:.6f}, sum={prob_sum:.6f}.")

    # -------------------------------------------------------------------------
    # Test 7: Missing feature is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 7] Testing rejection when a feature is missing...")
    incomplete_sample = real_sample.copy()
    incomplete_sample.pop("radius_mean")
    missing_rejected = False
    try:
        run_predict(incomplete_sample)
    except ValueError as e:
        missing_rejected = True
        print(f"  Expected error caught: {e}")
    assert missing_rejected, "Validation failed to reject input with missing feature!"
    print("  -> PASSED: Missing feature was rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 8: Unknown feature is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 8] Testing rejection when an unknown/unexpected feature is present...")
    unknown_sample = real_sample.copy()
    unknown_sample["spurious_biomarker_xyz"] = 999.0
    unknown_rejected = False
    try:
        run_predict(unknown_sample)
    except ValueError as e:
        unknown_rejected = True
        print(f"  Expected error caught: {e}")
    assert unknown_rejected, "Validation failed to reject input with unknown feature!"
    print("  -> PASSED: Unknown feature was rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 9: Non-numeric feature is rejected
    # -------------------------------------------------------------------------
    print("\n[Test 9] Testing rejection when a non-numeric feature value is provided...")
    corrupt_sample = real_sample.copy()
    corrupt_sample["texture_mean"] = "corrupt_string_value"
    non_numeric_rejected = False
    try:
        run_predict(corrupt_sample)
    except ValueError as e:
        non_numeric_rejected = True
        print(f"  Expected error caught: {e}")
    assert non_numeric_rejected, "Validation failed to reject non-numeric feature value!"
    print("  -> PASSED: Non-numeric value was rejected with ValueError.")

    # -------------------------------------------------------------------------
    # Test 10: Feature order follows feature_schema.json
    # -------------------------------------------------------------------------
    print("\n[Test 10] Testing canonical feature ordering construction...")
    # Permute dictionary keys in reverse order
    reversed_sample = {k: real_sample[k] for k in reversed(list(real_sample.keys()))}
    df_constructed = format_input_dataframe(reversed_sample)
    assert list(df_constructed.columns) == schema["feature_order"], (
        "Constructed DataFrame column order does not match feature_schema.json!"
    )
    assert list(df_constructed.columns) == canonical_features, (
        "Constructed DataFrame column order does not match canonical features!"
    )
    # Check values match corresponding keys
    for col in canonical_features:
        assert df_constructed[col].iloc[0] == real_sample[col], f"Value mismatch for column {col}"
    print("  -> PASSED: DataFrame column ordering strictly conforms to feature_schema.json.")

    print("\n" + "=" * 80)
    print("ALL 10 INFERENCE VERIFICATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
