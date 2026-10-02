"""Asthma Risk-Screening Inference Interface.

Provides a stateless, reproducible prediction interface that loads the serialized
preprocessor and model artifacts, validates input against the feature schema,
and returns risk screening predictions and calibrated probabilities.
"""

import json
import os
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd

from preprocessing import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    format_single_input,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

# Module-level cached artifacts
_MODEL = None
_PREPROCESSOR = None
_SCHEMA = None


def get_artifacts():
    """Load model, preprocessor, and schema lazily with caching."""
    global _MODEL, _PREPROCESSOR, _SCHEMA
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}. Train model first.")
        _MODEL = joblib.load(MODEL_PATH)

    if _PREPROCESSOR is None:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {PREPROCESSOR_PATH}.")
        _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

    if _SCHEMA is None:
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                _SCHEMA = json.load(f)

    return _MODEL, _PREPROCESSOR, _SCHEMA


def compute_heuristic_risk_tier(probability: float) -> str:
    """Categorize model probability into a software-defined heuristic risk tier.

    NOTE: The tiers ('Low', 'Moderate', 'High') are software-defined heuristic categories
    derived from statistical probability intervals (<0.30 = Low, 0.30-0.70 = Moderate,
    >=0.70 = High). They are intended strictly for UI grouping and screening prioritization,
    and are NOT clinical diagnoses, medical thresholds, or clinical severity classifications.
    """
    if probability < 0.30:
        return "Low"
    elif probability < 0.70:
        return "Moderate"
    else:
        return "High"


def predict(
    input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate asthma risk screening prediction and confidence probability.

    Args:
        input_data: Single patient feature dictionary, DataFrame, or list of feature dicts.

    Returns:
        Dictionary (or list of dicts) with:
            - prediction (str): 'Has Asthma' or 'No Asthma'
            - class (int): 1 or 0
            - probability (float): Model-estimated probability for positive class [0.0, 1.0]
            - risk_level (str): Software-defined heuristic risk category ('Low', 'Moderate', 'High')
            - disclaimer (str): Clinical safety disclaimer

    Raises:
        ValueError: If input features are missing, invalid, or out of range.
        FileNotFoundError: If model artifacts are missing.
    """
    model, preprocessor, _ = get_artifacts()

    # Handle list of dictionaries
    if isinstance(input_data, list):
        return [predict(item) for item in input_data]

    # Handle single dictionary input
    if isinstance(input_data, dict):
        # Strict validation: Check for prohibited leakage columns
        for excl in EXCLUDED_COLUMNS:
            if excl in input_data:
                # Silently ignore or drop to prevent any downstream usage
                input_data = {k: v for k, v in input_data.items() if k not in EXCLUDED_COLUMNS}

        df_input = format_single_input(input_data)

        # Transform through fitted preprocessor (zero leakage)
        X_proc = preprocessor.transform(df_input)

        # Predict class and probability
        pred_class = int(model.predict(X_proc)[0])
        probabilities = model.predict_proba(X_proc)[0]
        pos_prob = float(probabilities[1])

        return {
            "prediction": "Has Asthma" if pred_class == 1 else "No Asthma",
            "class": pred_class,
            "probability": round(pos_prob, 4),
            "risk_level": compute_heuristic_risk_tier(pos_prob),
            "risk_tier_basis": "Software-defined heuristic category (<0.30 Low, 0.30-0.70 Moderate, >=0.70 High), not a clinical threshold.",
            "disclaimer": "This model is an AI-based asthma risk-screening research component and is not a clinically validated diagnostic system. Consult a qualified pulmonologist for diagnosis.",
        }

    # Handle DataFrame input
    if isinstance(input_data, pd.DataFrame):
        results = []
        for _, row in input_data.iterrows():
            results.append(predict(row.to_dict()))
        return results

    raise ValueError(f"Unsupported input type: {type(input_data)}. Expected dict, list of dicts, or pandas DataFrame.")


if __name__ == "__main__":
    sample_patient = {
        "Age": 45,
        "Gender": "Female",
        "BMI": 28.5,
        "Smoking_Status": "Former",
        "Family_History": 1,
        "Allergies": "Pollen",
        "Air_Pollution_Level": "High",
        "Physical_Activity_Level": "Sedentary",
        "Occupation_Type": "Indoor",
        "Comorbidities": "None",
        "Medication_Adherence": 0.65,
        "Number_of_ER_Visits": 1,
        "Peak_Expiratory_Flow": 350.0,
        "FeNO_Level": 35.0,
    }
    print("Testing asthma inference interface with sample patient:")
    res = predict(sample_patient)
    print(json.dumps(res, indent=2))
