"""Parkinson's Disease Dataset Preprocessing Module.

Handles Excel data ingestion, subject identifier extraction for patient-level grouping,
target leakage prevention, and scikit-learn ColumnTransformer pipeline definition.
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN: str = "status"

EXCLUDED_COLUMNS: List[str] = [
    "name",
    "subject_id",
]

PREDICTOR_FEATURES: List[str] = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

FEATURE_COLUMNS: List[str] = PREDICTOR_FEATURES

FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "MDVP:Fo(Hz)": (50.0, 400.0),
    "MDVP:Fhi(Hz)": (50.0, 700.0),
    "MDVP:Flo(Hz)": (40.0, 350.0),
    "MDVP:Jitter(%)": (0.0, 0.1),
    "MDVP:Jitter(Abs)": (0.0, 0.001),
    "MDVP:RAP": (0.0, 0.1),
    "MDVP:PPQ": (0.0, 0.1),
    "Jitter:DDP": (0.0, 0.3),
    "MDVP:Shimmer": (0.0, 0.5),
    "MDVP:Shimmer(dB)": (0.0, 5.0),
    "Shimmer:APQ3": (0.0, 0.3),
    "Shimmer:APQ5": (0.0, 0.3),
    "MDVP:APQ": (0.0, 0.5),
    "Shimmer:DDA": (0.0, 1.0),
    "NHR": (0.0, 1.0),
    "HNR": (0.0, 60.0),
    "RPDE": (0.0, 1.0),
    "DFA": (0.0, 1.5),
    "spread1": (-15.0, 0.0),
    "spread2": (0.0, 1.0),
    "D2": (0.5, 5.0),
    "PPE": (0.0, 1.0),
}


def extract_subject_id(name_str: str) -> str:
    """Extract patient/subject identifier from the raw recording name.

    Example:
        'phon_R01_S01_1' -> 'S01'

    Args:
        name_str: Raw recording identifier string.

    Returns:
        Subject identifier string.
    """
    parts = str(name_str).strip().split("_")
    if len(parts) >= 3:
        return parts[2]
    return str(name_str).strip()


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for standard scaling of all continuous features.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    return preprocessor


def load_dataset(file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Load the Parkinson's dataset safely, extracting subject IDs for grouping and isolating target.

    Args:
        file_path: Absolute or relative path to Parkinsons_Disease_Dataset.xlsx.

    Returns:
        X (pd.DataFrame): 22-feature predictor matrix (strictly excluding name, subject_id, status).
        y (pd.Series): Binary target column status (0 = Healthy, 1 = Parkinson's disease).
        groups (pd.Series): Subject IDs extracted from name for grouped partitioning.
        metadata (dict): Ingestion summary statistics and leakage verification.

    Raises:
        ValueError: If unexpected schema, row count, or missing values are encountered.
    """
    raw_df = pd.read_excel(file_path, sheet_name="Parkinsons Dataset")

    if raw_df.shape[0] != 195:
        raise ValueError(f"Expected 195 rows in dataset, got {raw_df.shape[0]}.")

    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' not found in Excel dataset.")

    target_values = set(raw_df[TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        raise ValueError(f"Target column '{TARGET_COLUMN}' must contain only {{0, 1}}, got {target_values}.")

    if "name" not in raw_df.columns:
        raise ValueError("Expected 'name' column missing from dataset.")

    # Check for missing values across all required predictor features
    for col in FEATURE_COLUMNS:
        if col not in raw_df.columns:
            raise ValueError(f"Expected predictor column '{col}' missing from dataset.")
        null_count = raw_df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Predictor column '{col}' contains {null_count} null/missing values.")

    # Extract subject grouping variable (for patient-level grouping ONLY)
    groups = raw_df["name"].apply(extract_subject_id)

    # Build X and y
    X = raw_df[FEATURE_COLUMNS].astype(float).copy()
    y = raw_df[TARGET_COLUMN].astype(int).copy()

    unique_subjects = groups.nunique()
    subject_targets = raw_df.groupby(groups)[TARGET_COLUMN].nunique()
    conflicts = (subject_targets > 1).sum()
    if conflicts > 0:
        raise ValueError(f"Inconsistent target labels found across {conflicts} subjects.")

    metadata = {
        "raw_shape": list(raw_df.shape),
        "predictor_shape": list(X.shape),
        "features": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "target": TARGET_COLUMN,
        "total_records": len(raw_df),
        "unique_subjects": int(unique_subjects),
        "record_level_distribution": {
            "healthy_count_0": int((y == 0).sum()),
            "parkinsons_count_1": int((y == 1).sum()),
            "parkinsons_rate": float((y == 1).mean()),
        },
        "subject_level_distribution": {
            "healthy_subjects": int((raw_df.groupby(groups)[TARGET_COLUMN].first() == 0).sum()),
            "parkinsons_subjects": int((raw_df.groupby(groups)[TARGET_COLUMN].first() == 1).sum()),
        },
        "leakage_verification": "name, subject_id, and status strictly excluded from feature matrix X.",
    }

    return X, y, groups, metadata


def format_single_input(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """Validate and format a single patient voice recording dictionary into a standardized DataFrame row.

    Args:
        raw_input: Dictionary of acoustic measurement features.

    Returns:
        pd.DataFrame with 1 row matching FEATURE_COLUMNS.

    Raises:
        ValueError: If required features are missing, non-numeric, or outside plausible bounds.
    """
    cleaned_input: Dict[str, Any] = {}
    for k, v in raw_input.items():
        cleaned_input[str(k).strip()] = v

    # Remove excluded columns if passed
    for excl in EXCLUDED_COLUMNS + [TARGET_COLUMN]:
        cleaned_input.pop(excl, None)

    # Check for missing required features
    missing = [f for f in FEATURE_COLUMNS if f not in cleaned_input]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    formatted_row: Dict[str, float] = {}

    for col in FEATURE_COLUMNS:
        val = cleaned_input[col]
        try:
            val_float = float(val)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Feature '{col}' must be a valid number, got: {val}") from err

        low, high = FEATURE_BOUNDS[col]
        if not (low <= val_float <= high):
            raise ValueError(
                f"Feature '{col}' value {val_float} is outside plausible acoustic range [{low}, {high}]."
            )
        formatted_row[col] = val_float

    return pd.DataFrame([formatted_row], columns=FEATURE_COLUMNS)
