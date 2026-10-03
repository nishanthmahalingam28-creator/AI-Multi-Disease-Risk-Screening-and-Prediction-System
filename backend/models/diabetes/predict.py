"""
Prediction Module for Diabetes Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System

Stateless prediction module that:
1. Loads the serialized scikit-learn Pipeline artifact (model.joblib).
2. Validates incoming patient records against feature_schema.json.
3. Passes raw feature values directly to the loaded pipeline (which encapsulates
   zero-to-nan conversion, median imputation, robust scaling, and logistic regression).
4. Generates predicted class, predicted probability, and neutral screening estimates.
"""

import os
import sys
import json
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure unpickling finds the preprocessing module when loaded from external modules
try:
    import preprocessing
except ImportError:
    from backend.models.diabetes import preprocessing  # type: ignore
    sys.modules.setdefault("preprocessing", preprocessing)

MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

# Canonical 8 predictor features in exact expected order
EXPECTED_FEATURES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Cached module-level artifacts
_MODEL = None
_SCHEMA = None


def get_feature_schema() -> Dict[str, Any]:
    """Loads and caches the feature schema."""
    global _SCHEMA
    if _SCHEMA is None:
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(f"Feature schema not found at: {SCHEMA_PATH}")
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


def get_model():
    """Loads and caches the serialized scikit-learn Pipeline artifact."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at: {MODEL_PATH}. Ensure final_evaluation.py has been run."
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def validate_input(input_data: Any) -> Dict[str, float]:
    """
    Validates patient input against canonical feature schema requirements.
    
    Args:
        input_data: Dictionary, pandas Series, or single-row pandas DataFrame
                    containing the 8 required predictor features.
                    
    Returns:
        Validated dictionary mapping feature names to float values.
        
    Raises:
        TypeError: If input is not a dict, pd.Series, or 1-row pd.DataFrame.
        ValueError: If features are missing, unexpected/extra features are present,
                    or values are non-numeric, null, NaN, or non-finite.
    """
    # 1. Container type validation
    if isinstance(input_data, pd.DataFrame):
        if len(input_data) != 1:
            raise ValueError(f"Expected a single sample row, got DataFrame with {len(input_data)} rows.")
        raw_dict = input_data.iloc[0].to_dict()
    elif isinstance(input_data, pd.Series):
        raw_dict = input_data.to_dict()
    elif isinstance(input_data, dict):
        raw_dict = input_data.copy()
    else:
        raise TypeError(
            f"Invalid input type: {type(input_data).__name__}. Expected dict, pd.Series, or 1-row pd.DataFrame."
        )

    input_keys = set(raw_dict.keys())
    required_keys = set(EXPECTED_FEATURES)

    # 2. Check for missing features
    missing = [feat for feat in EXPECTED_FEATURES if feat not in input_keys]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    # 3. Check for unknown / extra features
    extra = [k for k in input_keys if k not in required_keys]
    if extra:
        raise ValueError(f"Unexpected extra feature(s) rejected: {extra}")

    # 4. Value validation: numeric, non-null, finite
    validated_dict = {}
    for feat in EXPECTED_FEATURES:
        val = raw_dict[feat]

        # Explicitly reject boolean, None, and non-numeric types
        if val is None or isinstance(val, bool):
            raise ValueError(f"Non-numeric value for feature '{feat}': {repr(val)}")

        # Convert to float
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Non-numeric value for feature '{feat}': {repr(val)}")

        # Reject NaN and infinite values supplied by caller
        if np.isnan(val_float) or np.isinf(val_float):
            raise ValueError(f"Invalid non-finite or NaN value for feature '{feat}': {val_float}")

        validated_dict[feat] = val_float

    return validated_dict


def format_input_dataframe(clean_dict: Dict[str, float]) -> pd.DataFrame:
    """Constructs a single-row DataFrame matching the exact canonical feature order."""
    ordered_values = [clean_dict[col] for col in EXPECTED_FEATURES]
    return pd.DataFrame([ordered_values], columns=EXPECTED_FEATURES, dtype=np.float64)


def predict_diabetes(input_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Generates an AI risk screening estimate for a single patient record.
    
    Args:
        input_data: Single record (dict, pd.Series, or 1-row pd.DataFrame)
                    with all 8 required features:
                    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                     
    Returns:
        Structured dictionary containing:
            - predicted_class (int): 0 or 1
            - predicted_probability (float): Probability corresponding to the predicted class
            - class_probabilities (dict): Probabilities for Class 0 and Class 1
            - screening_estimate (str): Neutral description ("Class 0" or "Class 1")
            - model_name (str): "Logistic_Regression"
            - status (str): "success"
    """
    # 1. Validate input strictly
    clean_dict = validate_input(input_data)

    # 2. Format DataFrame strictly following canonical schema order
    df_input = format_input_dataframe(clean_dict)

    # 3. Load pipeline (contains preprocessing + trained classifier)
    pipeline = get_model()

    # 4. Generate predictions directly through pipeline
    pred_class = int(pipeline.predict(df_input)[0])
    probabilities = pipeline.predict_proba(df_input)[0]

    prob_0 = float(probabilities[0])
    prob_1 = float(probabilities[1])
    prob_predicted = float(probabilities[pred_class])

    # 5. Build structured neutral response
    response = {
        "status": "success",
        "model_name": "Logistic_Regression",
        "predicted_class": pred_class,
        "predicted_probability": round(prob_predicted, 4),
        "class_probabilities": {
            "0": round(prob_0, 4),
            "1": round(prob_1, 4)
        },
        "screening_estimate": f"Class {pred_class}"
    }

    return response


# Alias for unified project interface
predict = predict_diabetes


def run_internal_validation_tests():
    """Runs verification checks for input validation and prediction consistency."""
    print("\n--- Running Internal Validation Tests ---")
    
    valid_sample = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    # Test 1: Valid input succeeds
    res = predict_diabetes(valid_sample)
    assert res["status"] == "success", "Valid input failed prediction!"
    assert res["predicted_class"] in [0, 1], "Invalid predicted_class!"
    assert 0.0 <= res["predicted_probability"] <= 1.0, "Probability not in [0, 1]!"
    print("  [Test 1] Valid input prediction: PASSED")

    # Test 2: Missing feature raises ValueError
    sample_missing = valid_sample.copy()
    del sample_missing["Glucose"]
    try:
        predict_diabetes(sample_missing)
        assert False, "Failed to raise ValueError on missing feature!"
    except ValueError as e:
        assert "Missing required feature" in str(e)
        print("  [Test 2] Missing feature rejection: PASSED")

    # Test 3: Unexpected extra feature raises ValueError
    sample_extra = valid_sample.copy()
    sample_extra["Cholesterol"] = 220
    try:
        predict_diabetes(sample_extra)
        assert False, "Failed to raise ValueError on extra feature!"
    except ValueError as e:
        assert "Unexpected extra feature" in str(e)
        print("  [Test 3] Extra feature rejection: PASSED")

    # Test 4: Non-numeric value raises ValueError
    sample_non_num = valid_sample.copy()
    sample_non_num["Glucose"] = "high"
    try:
        predict_diabetes(sample_non_num)
        assert False, "Failed to raise ValueError on non-numeric value!"
    except ValueError as e:
        assert "Non-numeric value" in str(e)
        print("  [Test 4] Non-numeric value rejection: PASSED")

    # Test 5: NaN value raises ValueError
    sample_nan = valid_sample.copy()
    sample_nan["BMI"] = float("nan")
    try:
        predict_diabetes(sample_nan)
        assert False, "Failed to raise ValueError on NaN value!"
    except ValueError as e:
        assert "NaN" in str(e)
        print("  [Test 5] NaN value rejection: PASSED")

    # Test 6: Infinite value raises ValueError
    sample_inf = valid_sample.copy()
    sample_inf["BloodPressure"] = float("inf")
    try:
        predict_diabetes(sample_inf)
        assert False, "Failed to raise ValueError on infinite value!"
    except ValueError as e:
        assert "non-finite" in str(e) or "infinite" in str(e).lower()
        print("  [Test 6] Infinite value rejection: PASSED")

    # Test 7: Output probability within [0, 1]
    assert 0.0 <= res["class_probabilities"]["0"] <= 1.0
    assert 0.0 <= res["class_probabilities"]["1"] <= 1.0
    assert abs(sum(res["class_probabilities"].values()) - 1.0) < 1e-3
    print("  [Test 7] Probability range and normalization [0, 1]: PASSED")

    print("All 7 validation tests passed successfully!")


if __name__ == "__main__":
    print("=" * 80)
    print("STEP 8: DIABETES RISK SCREENING PREDICTION MODULE DEMONSTRATION")
    print("=" * 80)

    # Example 1: Constructed from Row 0 of dataset (Actual Outcome: 1)
    example_1 = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    # Example 2: Constructed from Row 1 of dataset (Actual Outcome: 0)
    example_2 = {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31
    }

    print("\n--- Example 1 (Actual dataset observation 0) ---")
    print(f"Input features: {json.dumps(example_1, indent=2)}")
    res_1 = predict_diabetes(example_1)
    print(f"Predicted Class:       {res_1['predicted_class']}")
    print(f"Predicted Probability: {res_1['predicted_probability']:.4f}")
    print(f"Screening Estimate:    {res_1['screening_estimate']}")
    print(f"Full Result:           {json.dumps(res_1, indent=2)}")

    print("\n--- Example 2 (Actual dataset observation 1) ---")
    print(f"Input features: {json.dumps(example_2, indent=2)}")
    res_2 = predict_diabetes(example_2)
    print(f"Predicted Class:       {res_2['predicted_class']}")
    print(f"Predicted Probability: {res_2['predicted_probability']:.4f}")
    print(f"Screening Estimate:    {res_2['screening_estimate']}")
    print(f"Full Result:           {json.dumps(res_2, indent=2)}")

    # Run internal validation tests
    run_internal_validation_tests()

    print("\n" + "=" * 80)
    print("STEP 8 PREDICTION MODULE DEMONSTRATION COMPLETE")
    print("=" * 80)
