"""Parkinson's Disease Risk-Screening Inference Interface.

Provides a stateless, reproducible prediction interface that loads the serialized
preprocessor and model artifacts, validates input against the feature schema,
and returns risk screening predictions, calibrated probabilities, and software-defined heuristic risk tiers.
"""

import json
import os
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd

import importlib.util

_current_dir = os.path.dirname(os.path.abspath(__file__))
_prep_path = os.path.join(_current_dir, "preprocessing.py")
_spec = importlib.util.spec_from_file_location("parkinsons_preprocessing", _prep_path)
_prep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep_mod)

EXCLUDED_COLUMNS = _prep_mod.EXCLUDED_COLUMNS
FEATURE_COLUMNS = _prep_mod.FEATURE_COLUMNS
TARGET_COLUMN = _prep_mod.TARGET_COLUMN
format_single_input = _prep_mod.format_single_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

# Module-level cached artifacts
_MODEL = None
_PREPROCESSOR = None
_SCHEMA = None


def get_artifacts():
    """Load model, preprocessor, and schema lazily with module caching."""
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

    CRITICAL NOTE:
    These tiers ('Low', 'Moderate', 'High') are software-defined heuristic intervals
    (<0.30 = Low, 0.30-0.70 = Moderate, >=0.70 = High) for UI display and screening prioritization.
    They are NOT clinically validated thresholds, medical severity classifications, or neurological diagnoses.
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
    """Generate Parkinson's risk screening prediction and confidence probability.

    Args:
        input_data: Single voice measurement dictionary, DataFrame, or list of feature dicts.

    Returns:
        Dictionary (or list of dicts) with:
            - prediction (str): 'Parkinson\'s Disease' or 'Healthy / No Parkinson\'s Detected'
            - class (int): 1 or 0
            - probability (float): Model-estimated probability for Parkinson's disease [0.0, 1.0]
            - risk_level (str): Software-defined heuristic risk category ('Low', 'Moderate', 'High')
            - risk_tier_basis (str): Clarification of software-defined heuristic definition
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
        # Prevent leakage: remove administrative or target fields if passed
        cleaned_dict = {
            k: v for k, v in input_data.items()
            if k not in EXCLUDED_COLUMNS and k != TARGET_COLUMN
        }

        # Format and validate features against domain bounds
        df_input = format_single_input(cleaned_dict)

        # Scale features through fitted preprocessor (zero leakage)
        X_proc = preprocessor.transform(df_input)

        # Predict class and probability
        pred_class = int(model.predict(X_proc)[0])
        probabilities = model.predict_proba(X_proc)[0]
        pos_prob = float(probabilities[1])

        return {
            "prediction": "Parkinson's Disease" if pred_class == 1 else "Healthy / No Parkinson's Detected",
            "class": pred_class,
            "probability": round(pos_prob, 4),
            "risk_level": compute_heuristic_risk_tier(pos_prob),
            "risk_tier_basis": "Software-defined heuristic category (<0.30 Low, 0.30-0.70 Moderate, >=0.70 High), not a clinical threshold.",
            "disclaimer": "This tool provides an AI-based risk screening assessment derived from vocal perturbation measurements. It is not a clinical neurological diagnosis. Consult a qualified neurologist for definitive evaluation.",
        }

    # Handle DataFrame input
    if isinstance(input_data, pd.DataFrame):
        results = []
        for _, row in input_data.iterrows():
            results.append(predict(row.to_dict()))
        return results

    raise ValueError(f"Unsupported input type: {type(input_data)}. Expected dict, list of dicts, or pandas DataFrame.")


if __name__ == "__main__":
    sample_voice = {
        "MDVP:Fo(Hz)": 119.992,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.0037,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.426,
        "Shimmer:APQ3": 0.02182,
        "Shimmer:APQ5": 0.0313,
        "MDVP:APQ": 0.02971,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02211,
        "HNR": 21.033,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654,
    }
    print("Testing Parkinson's inference interface with sample recording:")
    res = predict(sample_voice)
    print(json.dumps(res, indent=2))
