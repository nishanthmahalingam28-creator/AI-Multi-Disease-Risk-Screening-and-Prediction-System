"""
Prediction Module for Heart Disease Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 8 — Heart Disease Prediction Module
Provides stateless, reproducible inference that:
1. Loads the serialized scikit-learn Pipeline artifact (model.joblib).
2. Validates incoming patient records against feature_schema.json.
3. Passes raw feature values directly to the loaded pipeline (which encapsulates
   continuous robust scaling and discrete/binary passthrough, followed by Random Forest classification).
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

# Ensure unpickling finds the preprocessing module when loaded from external packages
try:
    import preprocessing
except ImportError:
    from backend.models.heart_disease import preprocessing  # type: ignore
    sys.modules.setdefault("preprocessing", preprocessing)

# Canonical 13 predictor features in exact expected order from feature_schema.json
EXPECTED_FEATURES: List[str] = [
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


def validate_input(input_data: Any) -> Dict[str, float]:
    """
    Validates patient input against canonical feature schema requirements.
    
    Args:
        input_data: Dictionary, pandas Series, or single-row pandas DataFrame
                    containing the 13 required predictor features.
                    
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
        raise ValueError(
            f"Input validation error: Missing required feature(s): {missing}. "
            f"Expected all 13 features: {EXPECTED_FEATURES}"
        )

    # 3. Check for unexpected / extra features
    extra = [k for k in input_keys if k not in required_keys]
    if extra:
        raise ValueError(
            f"Input validation error: Unexpected feature(s) provided: {extra}. "
            f"Expected only the 13 canonical features: {EXPECTED_FEATURES}"
        )

    # 4. Value-level validation: Numeric, finite, non-null
    validated: Dict[str, float] = {}
    for feat in EXPECTED_FEATURES:
        val = raw_dict[feat]

        # Check for None / null
        if val is None:
            raise ValueError(f"Input validation error: Feature '{feat}' cannot be None.")

        # Check for boolean types masquerading as numbers
        if isinstance(val, bool):
            raise TypeError(
                f"Input validation error: Feature '{feat}' cannot be boolean (got {val}). "
                f"Numeric value expected."
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

    return validated


def predict_heart_disease(input_data: Any) -> Dict[str, Any]:
    """
    Stateless prediction function for heart disease risk screening.
    
    Accepts raw patient data, validates feature completeness and types,
    and executes the serialized production pipeline.
    
    Args:
        input_data: Dictionary, pandas Series, or single-row pandas DataFrame
                    with the 13 required predictor features.
                    
    Returns:
        Structured dictionary containing:
        - status: "success"
        - model_name: "Random_Forest"
        - predicted_class: int (0 or 1)
        - predicted_probability: float (probability of predicted class, [0.0, 1.0])
        - class_probabilities: Dict mapping "0" and "1" to probabilities
        - screening_estimate: "Class 0" or "Class 1"
    """
    # 1. Validate inputs
    validated_dict = validate_input(input_data)

    # 2. Build single-row DataFrame with canonical column ordering
    input_df = pd.DataFrame([validated_dict], columns=EXPECTED_FEATURES)

    # 3. Retrieve loaded model pipeline
    pipeline = get_model()

    # 4. Generate raw class prediction and probability array
    raw_class = int(pipeline.predict(input_df)[0])
    raw_probas = pipeline.predict_proba(input_df)[0]

    prob_class_0 = round(float(raw_probas[0]), 4)
    prob_class_1 = round(float(raw_probas[1]), 4)

    # Map probability corresponding to the predicted class
    predicted_probability = prob_class_1 if raw_class == 1 else prob_class_0

    # Neutral screening estimate representation
    screening_label = "Class 1" if raw_class == 1 else "Class 0"

    return {
        "status": "success",
        "model_name": "Random_Forest",
        "predicted_class": raw_class,
        "predicted_probability": round(predicted_probability, 4),
        "class_probabilities": {
            "0": prob_class_0,
            "1": prob_class_1
        },
        "screening_estimate": screening_label
    }


def predict(input_data: Any) -> Dict[str, Any]:
    """Alias for predict_heart_disease."""
    return predict_heart_disease(input_data)


def batch_predict(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates predictions for a list of patient records."""
    return [predict_heart_disease(record) for record in records]


if __name__ == "__main__":
    print("=" * 80)
    print("HEART DISEASE PREDICTION MODULE VERIFICATION")
    print("=" * 80)

    # Valid sample test (Actual row from heart.csv)
    sample_record = {
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

    print("\n1. Testing with valid patient sample:")
    result = predict_heart_disease(sample_record)
    print("Result:")
    print(json.dumps(result, indent=2))

    assert result["status"] == "success"
    assert result["model_name"] == "Random_Forest"
    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert "0" in result["class_probabilities"] and "1" in result["class_probabilities"]
    prob_sum = result["class_probabilities"]["0"] + result["class_probabilities"]["1"]
    assert abs(prob_sum - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {prob_sum}"
    assert result["screening_estimate"] in ["Class 0", "Class 1"]
    print("   [PASS] Valid sample output structure and probability ranges verified.")

    # Container type compatibility tests
    print("\n2. Testing container type flexibility:")
    # pd.Series
    series_result = predict_heart_disease(pd.Series(sample_record))
    assert series_result["predicted_class"] == result["predicted_class"]
    print("   [PASS] pd.Series accepted.")

    # 1-row pd.DataFrame
    df_result = predict_heart_disease(pd.DataFrame([sample_record]))
    assert df_result["predicted_class"] == result["predicted_class"]
    print("   [PASS] 1-row pd.DataFrame accepted.")

    # Multiple-row DataFrame rejection
    try:
        predict_heart_disease(pd.DataFrame([sample_record, sample_record]))
        print("   [FAIL] Multiple-row DataFrame was NOT rejected!")
    except ValueError as e:
        print(f"   [PASS] Multiple-row DataFrame correctly rejected: {e}")

    # Invalid input tests
    print("\n3. Testing invalid input rejection:")
    # Missing feature
    missing_sample = {k: v for k, v in sample_record.items() if k != "Thallium"}
    try:
        predict_heart_disease(missing_sample)
        print("   [FAIL] Missing feature was NOT rejected!")
    except ValueError as e:
        print(f"   [PASS] Missing feature correctly rejected: {e}")

    # Extra feature
    extra_sample = {**sample_record, "UnexpectedField": 99}
    try:
        predict_heart_disease(extra_sample)
        print("   [FAIL] Extra feature was NOT rejected!")
    except ValueError as e:
        print(f"   [PASS] Extra feature correctly rejected: {e}")

    # Non-numeric value
    non_numeric_sample = {**sample_record, "Cholesterol": "high"}
    try:
        predict_heart_disease(non_numeric_sample)
        print("   [FAIL] Non-numeric value was NOT rejected!")
    except TypeError as e:
        print(f"   [PASS] Non-numeric value correctly rejected: {e}")

    # NaN value
    nan_sample = {**sample_record, "BP": float("nan")}
    try:
        predict_heart_disease(nan_sample)
        print("   [FAIL] NaN value was NOT rejected!")
    except ValueError as e:
        print(f"   [PASS] NaN value correctly rejected: {e}")

    # Infinite value
    inf_sample = {**sample_record, "Max HR": float("inf")}
    try:
        predict_heart_disease(inf_sample)
        print("   [FAIL] Infinite value was NOT rejected!")
    except ValueError as e:
        print(f"   [PASS] Infinite value correctly rejected: {e}")

    # Boolean value
    bool_sample = {**sample_record, "Sex": True}
    try:
        predict_heart_disease(bool_sample)
        print("   [FAIL] Boolean value was NOT rejected!")
    except TypeError as e:
        print(f"   [PASS] Boolean value correctly rejected: {e}")

    print("\n" + "=" * 80)
    print("PREDICTION MODULE VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
