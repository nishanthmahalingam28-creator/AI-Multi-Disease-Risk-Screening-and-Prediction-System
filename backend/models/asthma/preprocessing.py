"""Asthma Dataset Preprocessing Module.

Handles NA-safe data ingestion, feature group isolation, target leakage prevention,
and scikit-learn ColumnTransformer pipeline definition.
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN: str = "Has_Asthma"

EXCLUDED_COLUMNS: List[str] = [
    "Patient_ID",
    "Asthma_Control_Level",
]

NUMERICAL_FEATURES: List[str] = [
    "Age",
    "BMI",
    "Medication_Adherence",
    "Number_of_ER_Visits",
    "Peak_Expiratory_Flow",
    "FeNO_Level",
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Smoking_Status",
    "Allergies",
    "Air_Pollution_Level",
    "Physical_Activity_Level",
    "Occupation_Type",
    "Comorbidities",
]

BINARY_FEATURES: List[str] = [
    "Family_History",
]

FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

VALID_CATEGORIES: Dict[str, List[str]] = {
    "Gender": ["Female", "Male", "Other"],
    "Smoking_Status": ["Never", "Former", "Current"],
    "Allergies": ["None", "Dust", "Pollen", "Pets", "Multiple"],
    "Air_Pollution_Level": ["Low", "Moderate", "High"],
    "Physical_Activity_Level": ["Sedentary", "Moderate", "Active"],
    "Occupation_Type": ["Indoor", "Outdoor"],
    "Comorbidities": ["None", "Diabetes", "Hypertension", "Both"],
}

NUMERICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Age": (1.0, 120.0),
    "BMI": (10.0, 65.0),
    "Medication_Adherence": (0.0, 1.0),
    "Number_of_ER_Visits": (0.0, 20.0),
    "Peak_Expiratory_Flow": (50.0, 900.0),
    "FeNO_Level": (0.0, 150.0),
}


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for numerical scaling and categorical encoding.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_scaler", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bin_passthrough", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load the asthma dataset safely, preserving literal 'None' and 'N/A' strings.

    Args:
        csv_path: Absolute or relative path to synthetic_asthma_dataset.csv.

    Returns:
        X (pd.DataFrame): 14-feature predictor matrix (strictly excluding Patient_ID and Asthma_Control_Level).
        y (pd.Series): Binary target column Has_Asthma (0 = No Asthma, 1 = Has Asthma).
        metadata (dict): Ingestion summary statistics and leakage check confirmation.

    Raises:
        ValueError: If unexpected schema or missing target is encountered.
    """
    # Use keep_default_na=False so that literal 'None' and 'N/A' remain valid strings
    raw_df = pd.read_csv(csv_path, keep_default_na=False)

    if raw_df.shape[0] != 10000:
        raise ValueError(f"Expected 10,000 rows in dataset, got {raw_df.shape[0]}.")

    # Verify target presence and values
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' not found in CSV.")

    target_values = set(raw_df[TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        raise ValueError(f"Target column '{TARGET_COLUMN}' must contain only {0, 1}, got {target_values}.")

    # Leakage check: Confirm excluded columns are explicitly quarantined
    for excl in EXCLUDED_COLUMNS:
        if excl not in raw_df.columns:
            raise ValueError(f"Expected column '{excl}' not found in dataset.")

    # Check for unexpected truly blank or whitespace cells
    for col in FEATURE_COLUMNS:
        if col not in raw_df.columns:
            raise ValueError(f"Expected predictor column '{col}' missing from dataset.")
        # Ensure no empty strings
        empty_mask = raw_df[col].astype(str).str.strip() == ""
        if empty_mask.any():
            raise ValueError(f"Column '{col}' contains {empty_mask.sum()} empty string cells.")

    # Build X and y
    X = raw_df[FEATURE_COLUMNS].copy()
    y = raw_df[TARGET_COLUMN].astype(int).copy()

    # Ensure types
    for col in NUMERICAL_FEATURES:
        X[col] = X[col].astype(float)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).str.strip()
    for col in BINARY_FEATURES:
        X[col] = X[col].astype(int)

    metadata = {
        "raw_shape": list(raw_df.shape),
        "predictor_shape": list(X.shape),
        "features": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "target": TARGET_COLUMN,
        "target_distribution": {
            "negative_count_0": int((y == 0).sum()),
            "positive_count_1": int((y == 1).sum()),
            "positive_rate": float((y == 1).mean()),
        },
        "leakage_verification": "Patient_ID and Asthma_Control_Level successfully excluded from feature matrix.",
    }

    return X, y, metadata


def format_single_input(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """Validate and format a single patient dictionary into a standardized DataFrame row.

    Args:
        raw_input: Dictionary of patient attributes.

    Returns:
        pd.DataFrame with 1 row matching FEATURE_COLUMNS.

    Raises:
        ValueError: If required features are missing or invalid.
    """
    # Clean keys and strip whitespace
    cleaned_input: Dict[str, Any] = {}
    for k, v in raw_input.items():
        cleaned_input[str(k).strip()] = v

    # Omit excluded columns if provided (prevent accidental leakage or failure)
    for excl in EXCLUDED_COLUMNS:
        cleaned_input.pop(excl, None)

    # Check for missing required features
    missing = [f for f in FEATURE_COLUMNS if f not in cleaned_input]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    formatted_row: Dict[str, Any] = {}

    # Validate numerical features
    for col in NUMERICAL_FEATURES:
        val = cleaned_input[col]
        try:
            val_float = float(val)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Feature '{col}' must be a valid number, got: {val}") from err

        low, high = NUMERICAL_BOUNDS[col]
        if not (low <= val_float <= high):
            raise ValueError(f"Feature '{col}' value {val_float} is outside plausible range [{low}, {high}].")
        formatted_row[col] = val_float

    # Validate categorical features
    for col in CATEGORICAL_FEATURES:
        val_str = str(cleaned_input[col]).strip()
        allowed = VALID_CATEGORIES[col]
        # Match case-insensitively to permitted categories
        matched = next((opt for opt in allowed if opt.lower() == val_str.lower()), None)
        if matched is None:
            raise ValueError(f"Invalid category for '{col}': '{val_str}'. Allowed options: {allowed}")
        formatted_row[col] = matched

    # Validate binary features
    for col in BINARY_FEATURES:
        val = cleaned_input[col]
        try:
            val_int = int(val)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Feature '{col}' must be 0 or 1, got: {val}") from err
        if val_int not in (0, 1):
            raise ValueError(f"Feature '{col}' must be 0 or 1, got: {val_int}")
        formatted_row[col] = val_int

    return pd.DataFrame([formatted_row], columns=FEATURE_COLUMNS)
