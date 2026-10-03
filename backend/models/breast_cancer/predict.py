"""Breast Cancer Risk-Screening Inference Module.
Member 1: AI Multi-Disease Risk Screening and Prediction System

Stateless prediction module that:
1. Loads the serialized scikit-learn Pipeline (model.joblib).
2. Validates incoming patient records against feature_schema.json.
3. Automatically applies the embedded preprocessing (StandardScaler).
4. Generates discrete class predictions and calibrated probability estimates.
"""

import json
import os
import sys
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

# Module-level cached artifacts
_MODEL = None
_SCHEMA = None


def get_feature_schema() -> Dict[str, Any]:
    """Loads and caches feature schema metadata."""
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
                f"Model artifact not found at: {MODEL_PATH}. Run serialization first."
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def validate_input(input_data: Any) -> Dict[str, float]:
    """Validates patient input against canonical feature schema requirements.

    Args:
        input_data: Dictionary, pandas Series, or single-row pandas DataFrame
                    containing the 30 continuous features.

    Returns:
        Validated dictionary of float feature values.

    Raises:
        TypeError: If input is not a dict, Series, or DataFrame.
        ValueError: If features are missing, unexpected/unknown features are present,
                    or values are non-numeric, null, or non-finite.
    """
    schema = get_feature_schema()
    canonical_features = schema["feature_names"]

    # 1. Validate container type
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

    # 2. Check for missing required features
    input_keys = set(raw_dict.keys())
    missing = [feat for feat in canonical_features if feat not in input_keys]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    # 3. Check for unknown / unexpected features
    unknown = [k for k in input_keys if k not in canonical_features]
    if unknown:
        raise ValueError(f"Unknown or unexpected feature(s) rejected: {unknown}")

    # 4. Check for non-numeric, null, or non-finite values
    validated_dict = {}
    for feat in canonical_features:
        val = raw_dict[feat]

        # Explicitly reject None, booleans, and empty strings
        if val is None or isinstance(val, (bool, str)):
            # If string, test if convertible to float
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    raise ValueError(f"Non-numeric value for feature '{feat}': {repr(raw_dict[feat])}")
            else:
                raise ValueError(f"Non-numeric value for feature '{feat}': {repr(raw_dict[feat])}")

        try:
            val_float = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Non-numeric value for feature '{feat}': {repr(val)}")

        if np.isnan(val_float) or np.isinf(val_float):
            raise ValueError(f"Invalid non-finite or NaN value for feature '{feat}': {val_float}")

        validated_dict[feat] = val_float

    return validated_dict


def format_input_dataframe(clean_dict: Dict[str, float]) -> pd.DataFrame:
    """Constructs a single-row DataFrame using the exact canonical feature order."""
    schema = get_feature_schema()
    canonical_order = schema["feature_order"]
    ordered_values = [clean_dict[col] for col in canonical_order]
    return pd.DataFrame([ordered_values], columns=canonical_order, dtype=np.float64)


def predict(input_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """Generates an AI risk screening estimate for a single patient sample.

    Args:
        input_data: Single patient record (dict, pd.Series, or 1-row pd.DataFrame)
                    with all 30 required continuous cytological features.

    Returns:
        Structured dictionary containing:
            - status (str): "success"
            - disease (str): "breast_cancer"
            - predicted_class (int): 0 or 1
            - predicted_probability (float): Model-estimated probability of Class 1
            - class_probabilities (dict): Probabilities for Class 0 and Class 1
            - screening_estimate (str): Neutral description ("Class 0" or "Class 1")
            - disclaimer (str): Educational/screening boundary statement

    Raises:
        ValueError: If input validation fails.
        TypeError: If input data type is unsupported.
    """
    # 1. Validate input strictly
    clean_dict = validate_input(input_data)

    # 2. Format DataFrame strictly following canonical schema order
    df_input = format_input_dataframe(clean_dict)

    # 3. Load model pipeline (preprocessing + classifier)
    pipeline = get_model()

    # 4. Generate predictions
    pred_class = int(pipeline.predict(df_input)[0])
    probabilities = pipeline.predict_proba(df_input)[0]

    prob_0 = float(probabilities[0])
    prob_1 = float(probabilities[1])

    # 5. Build structured neutral response
    response = {
        "status": "success",
        "disease": "breast_cancer",
        "predicted_class": pred_class,
        "predicted_probability": round(prob_1, 6),
        "class_probabilities": {
            "0": round(prob_0, 6),
            "1": round(prob_1, 6),
        },
        "screening_estimate": "Class 1" if pred_class == 1 else "Class 0",
        "disclaimer": "This is an AI screening estimate for risk evaluation and does not constitute a medical diagnosis.",
    }

    return response


if __name__ == "__main__":
    # Simple manual demonstration
    dataset_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "..", "..", "datasets", "breast_cancer", "breast.csv")
    )
    if os.path.exists(dataset_path):
        schema = get_feature_schema()
        raw_df = pd.read_csv(dataset_path)
        sample_record = raw_df.iloc[0][schema["feature_names"]].to_dict()
        print("Manual Demonstration Inference:")
        result = predict(sample_record)
        print(json.dumps(result, indent=2))
