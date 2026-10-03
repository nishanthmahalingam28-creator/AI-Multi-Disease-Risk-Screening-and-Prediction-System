"""
Prediction Module for Stroke Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 8 — Stroke Prediction Module
Provides stateless, reproducible inference that:
1. Loads the serialized scikit-learn Pipeline artifact (model.joblib).
2. Validates incoming patient records against feature_schema.json.
3. Passes raw feature values directly to the loaded pipeline (which encapsulates
   continuous robust scaling, discrete passthrough, one-hot encoding, and Logistic Regression classification).
4. Generates predicted class, predicted probability, and neutral screening estimates.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import joblib

# Robust path handling using pathlib
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
SCHEMA_PATH = BASE_DIR / "feature_schema.json"

# Ensure local module directory is in sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Canonical 10 predictor features in exact expected order from feature_schema.json
EXPECTED_FEATURES: List[str] = [
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

NUMERIC_FEATURES: List[str] = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

# Cached module-level artifacts
_MODEL = None
_SCHEMA = None


def get_feature_schema() -> Dict[str, Any]:
    """Loads and caches the feature schema."""
    global _SCHEMA
    if _SCHEMA is None:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Feature schema not found at: {SCHEMA_PATH}")
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


def get_model():
    """Loads and caches the serialized scikit-learn Pipeline artifact."""
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at: {MODEL_PATH}. Ensure final_evaluation.py has been run."
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def validate_input(input_data: Any) -> Dict[str, Any]:
    """
    Validates patient input against canonical feature schema requirements.
    
    Args:
        input_data: Dictionary, pandas Series, or single-row pandas DataFrame
                    containing the 10 required predictor features.
                    
    Returns:
        Validated dictionary mapping feature names to sanitized values.
        
    Raises:
        TypeError: If input is not a dict, pd.Series, or 1-row pd.DataFrame,
                   or if a numeric feature is provided as a non-numeric type.
        ValueError: If features are missing, unexpected/extra features are present,
                    or values are null, NaN, or non-finite.
    """
    # 1. Container type validation and structure check
    if isinstance(input_data, pd.DataFrame):
        if len(input_data) == 0:
            raise ValueError("Input validation error: Empty DataFrame provided.")
        if len(input_data) != 1:
            raise ValueError(
                f"Input validation error: Expected a single sample row, got DataFrame with {len(input_data)} rows."
            )
        raw_dict = input_data.iloc[0].to_dict()
    elif isinstance(input_data, pd.Series):
        if len(input_data) == 0:
            raise ValueError("Input validation error: Empty Series provided.")
        raw_dict = input_data.to_dict()
    elif isinstance(input_data, dict):
        if len(input_data) == 0:
            raise ValueError("Input validation error: Empty dictionary provided.")
        raw_dict = input_data.copy()
    else:
        raise TypeError(
            f"Input validation error: Invalid input type: {type(input_data).__name__}. "
            f"Expected dict, pd.Series, or 1-row pd.DataFrame."
        )

    input_keys = set(raw_dict.keys())
    required_keys = set(EXPECTED_FEATURES)

    # 2. Check for missing features
    missing = [feat for feat in EXPECTED_FEATURES if feat not in input_keys]
    if missing:
        raise ValueError(
            f"Input validation error: Missing required feature(s): {missing}. "
            f"Expected all 10 features: {EXPECTED_FEATURES}"
        )

    # 3. Check for unexpected / extra features (including target 'stroke' and 'id')
    extra = [k for k in input_keys if k not in required_keys]
    if extra:
        raise ValueError(
            f"Input validation error: Unexpected feature(s) provided: {extra}. "
            f"Expected only the 10 canonical features: {EXPECTED_FEATURES}"
        )

    # 4. Value-level validation: Numeric, finite, non-null, categorical non-empty
    validated: Dict[str, Any] = {}
    for feat in EXPECTED_FEATURES:
        val = raw_dict[feat]

        # Check for None / null / pandas NA
        if val is None or pd.isna(val):
            raise ValueError(f"Input validation error: Feature '{feat}' cannot be null or NaN.")

        # Numeric features validation
        if feat in NUMERIC_FEATURES:
            # Reject boolean types masquerading as integers
            if isinstance(val, bool):
                raise TypeError(
                    f"Input validation error: Numeric feature '{feat}' cannot be boolean (got {val})."
                )

            # Type conversion to float
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                raise TypeError(
                    f"Input validation error: Feature '{feat}' must be numeric. "
                    f"Got value '{val}' of type '{type(val).__name__}'."
                )

            # Check for NaN and Inf
            if math.isnan(num_val):
                raise ValueError(f"Input validation error: Feature '{feat}' cannot be NaN.")
            if math.isinf(num_val):
                raise ValueError(f"Input validation error: Feature '{feat}' cannot be infinite.")

            validated[feat] = num_val

        # Categorical features validation
        else:
            if not isinstance(val, str):
                # Only string representation allowed for categoricals
                raise TypeError(
                    f"Input validation error: Categorical feature '{feat}' must be a string. "
                    f"Got value '{val}' of type '{type(val).__name__}'."
                )
            clean_str = val.strip()
            if not clean_str:
                raise ValueError(f"Input validation error: Categorical feature '{feat}' cannot be empty.")
            validated[feat] = clean_str

    return validated


def predict_one(input_data: Any) -> Dict[str, Any]:
    """
    Stateless prediction function for single-sample stroke risk screening.
    
    Accepts raw patient data, validates feature completeness and types,
    preserves exact feature order, and executes the serialized production pipeline.
    
    Args:
        input_data: Dictionary, pandas Series, or single-row pandas DataFrame
                    with the 10 required predictor features.
                    
    Returns:
        Structured neutral dictionary containing:
        - status: "success"
        - model_name: "Logistic_Regression"
        - predicted_class: int (0 or 1)
        - predicted_probability: float (predicted probability for Class 1, [0.0, 1.0])
        - class_probabilities: Dict mapping "0" and "1" to probabilities
        - threshold: 0.50
        - model_prediction: "Class 0" or "Class 1"
    """
    # 1. Validate inputs and return sanitized mapping
    validated_dict = validate_input(input_data)

    # 2. Build single-row DataFrame with strict canonical column ordering
    input_df = pd.DataFrame([validated_dict], columns=EXPECTED_FEATURES)

    # 3. Retrieve cached model pipeline
    pipeline = get_model()

    # 4. Generate raw class prediction and probability array
    raw_class = int(pipeline.predict(input_df)[0])
    raw_probas = pipeline.predict_proba(input_df)[0]

    prob_class_0 = round(float(raw_probas[0]), 4)
    prob_class_1 = round(float(raw_probas[1]), 4)

    # In risk screening, predicted_probability reports probability for positive Class 1
    predicted_probability = prob_class_1

    # Neutral screening label representation
    neutral_label = "Class 1" if raw_class == 1 else "Class 0"

    return {
        "status": "success",
        "model_name": "Logistic_Regression",
        "predicted_class": raw_class,
        "predicted_probability": predicted_probability,
        "class_probabilities": {
            "0": prob_class_0,
            "1": prob_class_1
        },
        "threshold": 0.50,
        "model_prediction": neutral_label
    }


def predict(input_data: Any) -> Dict[str, Any]:
    """Alias for predict_one."""
    return predict_one(input_data)


def predict_stroke(input_data: Any) -> Dict[str, Any]:
    """Alias for predict_one."""
    return predict_one(input_data)


def batch_predict(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates predictions for a list of patient records."""
    return [predict_one(record) for record in records]


if __name__ == "__main__":
    print("=" * 80)
    print("STROKE PREDICTION MODULE VERIFICATION (STEP 8)")
    print("=" * 80)

    checks_passed = 0
    checks_failed = 0

    # 1. Load model verification
    print("\n1. Verifying model loading:")
    try:
        pipeline = get_model()
        assert pipeline is not None
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")
        print("   [PASS] Saved model loaded successfully from model.joblib.")
        checks_passed += 1
    except Exception as e:
        print(f"   [FAIL] Failed to load model: {e}")
        checks_failed += 1

    # Valid sample from actual stroke dataset (row 0: Male, 67.0, 0, 1, Yes, Private, Urban, 228.69, 36.6, formerly smoked)
    sample_record = {
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

    # 2. Valid dataset row through predict_one()
    print("\n2. Testing with valid patient sample:")
    try:
        result = predict_one(sample_record)
        print("   Result output:")
        print(json.dumps(result, indent=4))
        print("   [PASS] Valid sample processed successfully.")
        checks_passed += 1
    except Exception as e:
        print(f"   [FAIL] Error processing valid sample: {e}")
        checks_failed += 1

    # 3. Required output fields
    print("\n3. Verifying required output fields:")
    required_fields = ["status", "model_name", "predicted_class", "predicted_probability", "class_probabilities", "threshold", "model_prediction"]
    missing_fields = [f for f in required_fields if f not in result]
    if not missing_fields:
        print(f"   [PASS] All required output fields present: {required_fields}")
        checks_passed += 1
    else:
        print(f"   [FAIL] Missing fields: {missing_fields}")
        checks_failed += 1

    # 4. Predicted class is 0 or 1
    print("\n4. Verifying predicted class:")
    if result["predicted_class"] in [0, 1]:
        print(f"   [PASS] Predicted class is valid: {result['predicted_class']}")
        checks_passed += 1
    else:
        print(f"   [FAIL] Invalid predicted class: {result['predicted_class']}")
        checks_failed += 1

    # 5. Predicted probability in [0, 1] and finite
    print("\n5. Verifying predicted probability:")
    prob = result["predicted_probability"]
    if 0.0 <= prob <= 1.0 and math.isfinite(prob):
        print(f"   [PASS] Predicted probability is finite and within [0, 1]: {prob}")
        checks_passed += 1
    else:
        print(f"   [FAIL] Invalid predicted probability: {prob}")
        checks_failed += 1

    # 6. Missing feature input rejected
    print("\n6. Testing rejection of missing features:")
    missing_sample = {k: v for k, v in sample_record.items() if k != "smoking_status"}
    try:
        predict_one(missing_sample)
        print("   [FAIL] Missing feature was NOT rejected!")
        checks_failed += 1
    except ValueError as e:
        print(f"   [PASS] Missing feature correctly rejected: {e}")
        checks_passed += 1

    # 7. Extra feature input rejected (e.g. 'stroke' or arbitrary field)
    print("\n7. Testing rejection of extra features:")
    extra_sample = {**sample_record, "stroke": 1}
    try:
        predict_one(extra_sample)
        print("   [FAIL] Extra feature was NOT rejected!")
        checks_failed += 1
    except ValueError as e:
        print(f"   [PASS] Extra feature correctly rejected: {e}")
        checks_passed += 1

    # 8. NaN input rejected
    print("\n8. Testing rejection of NaN input:")
    nan_sample = {**sample_record, "bmi": float("nan")}
    try:
        predict_one(nan_sample)
        print("   [FAIL] NaN value was NOT rejected!")
        checks_failed += 1
    except ValueError as e:
        print(f"   [PASS] NaN value correctly rejected: {e}")
        checks_passed += 1

    # 9. Infinite numeric input rejected
    print("\n9. Testing rejection of infinite numeric input:")
    inf_sample = {**sample_record, "avg_glucose_level": float("inf")}
    try:
        predict_one(inf_sample)
        print("   [FAIL] Infinite value was NOT rejected!")
        checks_failed += 1
    except ValueError as e:
        print(f"   [PASS] Infinite value correctly rejected: {e}")
        checks_passed += 1

    # 10. Non-numeric input for a numeric feature rejected
    print("\n10. Testing rejection of non-numeric input for numeric feature:")
    non_numeric_sample = {**sample_record, "age": "not_an_age"}
    try:
        predict_one(non_numeric_sample)
        print("   [FAIL] Non-numeric value was NOT rejected!")
        checks_failed += 1
    except TypeError as e:
        print(f"   [PASS] Non-numeric value correctly rejected: {e}")
        checks_passed += 1

    # 11. Feature order preserved
    print("\n11. Verifying feature order preservation:")
    # Scramble the input dictionary order
    scrambled_record = {
        "smoking_status": sample_record["smoking_status"],
        "age": sample_record["age"],
        "gender": sample_record["gender"],
        "bmi": sample_record["bmi"],
        "Residence_type": sample_record["Residence_type"],
        "work_type": sample_record["work_type"],
        "avg_glucose_level": sample_record["avg_glucose_level"],
        "ever_married": sample_record["ever_married"],
        "heart_disease": sample_record["heart_disease"],
        "hypertension": sample_record["hypertension"]
    }
    scrambled_result = predict_one(scrambled_record)
    if scrambled_result == result:
        print("   [PASS] Scrambled feature dictionary produces identical prediction; column order preserved.")
        checks_passed += 1
    else:
        print("   [FAIL] Prediction differed when input key order varied!")
        checks_failed += 1

    # 12. Caller's original dictionary/DataFrame is not modified
    print("\n12. Verifying caller immutability:")
    original_dict = {
        "gender": "Female",
        "age": 45.0,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Govt_job",
        "Residence_type": "Rural",
        "avg_glucose_level": 85.0,
        "bmi": 25.0,
        "smoking_status": "never smoked"
    }
    dict_copy = original_dict.copy()
    predict_one(original_dict)
    if original_dict == dict_copy:
        print("   [PASS] Caller's input dictionary was not mutated.")
        checks_passed += 1
    else:
        print("   [FAIL] Caller's dictionary was modified!")
        checks_failed += 1

    # 13. Determinism: calling twice produces identical results
    print("\n13. Verifying determinism across multiple invocations:")
    run1 = predict_one(sample_record)
    run2 = predict_one(sample_record)
    if run1 == run2:
        print("   [PASS] Identical inputs produce identical outputs across calls.")
        checks_passed += 1
    else:
        print("   [FAIL] Inconsistent output across repeated invocations!")
        checks_failed += 1

    # 14. Model is not retrained or refitted during prediction
    print("\n14. Verifying model is not retrained during prediction:")
    clf = pipeline.named_steps["classifier"]
    coef_before = clf.coef_.copy()
    intercept_before = clf.intercept_.copy()

    predict_one(sample_record)
    predict_one(original_dict)

    coef_after = clf.coef_
    intercept_after = clf.intercept_

    if np.array_equal(coef_before, coef_after) and np.array_equal(intercept_before, intercept_after):
        print("   [PASS] Model coefficients and intercept are unchanged; no retraining occurred.")
        checks_passed += 1
    else:
        print("   [FAIL] Model parameters changed during prediction!")
        checks_failed += 1

    # Container flexibility verification
    print("\n15. Additional container format tests:")
    # pd.Series
    series_res = predict_one(pd.Series(sample_record))
    assert series_res["predicted_class"] == result["predicted_class"]
    print("   [PASS] pd.Series accepted.")

    # 1-row pd.DataFrame
    df_res = predict_one(pd.DataFrame([sample_record]))
    assert df_res["predicted_class"] == result["predicted_class"]
    print("   [PASS] 1-row pd.DataFrame accepted.")

    # Multiple-row DataFrame rejection
    try:
        predict_one(pd.DataFrame([sample_record, sample_record]))
        print("   [FAIL] Multi-row DataFrame was NOT rejected!")
    except ValueError:
        print("   [PASS] Multi-row DataFrame correctly rejected.")

    # Empty input rejection
    try:
        predict_one({})
        print("   [FAIL] Empty dictionary was NOT rejected!")
    except ValueError:
        print("   [PASS] Empty dictionary correctly rejected.")

    print("\n" + "=" * 80)
    print(f"VERIFICATION SUMMARY: {checks_passed} checks passed, {checks_failed} checks failed.")
    print("=" * 80)

    if checks_failed > 0:
        sys.exit(1)
