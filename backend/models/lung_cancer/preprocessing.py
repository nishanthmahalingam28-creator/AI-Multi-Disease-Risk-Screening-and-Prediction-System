"""Lung Cancer Dataset Preprocessing Module.

Handles data loading, column name normalization, feature encoding,
duplicate detection/handling, and scikit-learn preprocessor pipeline definition.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Column mapping from raw CSV headers to standardized internal feature names
RAW_TO_CLEAN_MAP: Dict[str, str] = {
    "GENDER": "gender",
    "AGE": "age",
    "SMOKING": "smoking",
    "YELLOW_FINGERS": "yellow_fingers",
    "ANXIETY": "anxiety",
    "PEER_PRESSURE": "peer_pressure",
    "CHRONIC DISEASE": "chronic_disease",
    "FATIGUE ": "fatigue",
    "ALLERGY ": "allergy",
    "WHEEZING": "wheezing",
    "ALCOHOL CONSUMING": "alcohol_consuming",
    "COUGHING": "coughing",
    "SHORTNESS OF BREATH": "shortness_of_breath",
    "SWALLOWING DIFFICULTY": "swallowing_difficulty",
    "CHEST PAIN": "chest_pain",
    "LUNG_CANCER": "lung_cancer",
}

FEATURE_COLUMNS: List[str] = [
    "gender",
    "age",
    "smoking",
    "yellow_fingers",
    "anxiety",
    "peer_pressure",
    "chronic_disease",
    "fatigue",
    "allergy",
    "wheezing",
    "alcohol_consuming",
    "coughing",
    "shortness_of_breath",
    "swallowing_difficulty",
    "chest_pain",
]

TARGET_COLUMN: str = "lung_cancer"


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and convert column names to standardized snake_case."""
    df_copy = df.copy()
    new_cols = {}
    for col in df_copy.columns:
        if col in RAW_TO_CLEAN_MAP:
            new_cols[col] = RAW_TO_CLEAN_MAP[col]
        else:
            # Fallback sanitization: strip and snake_case
            clean_name = col.strip().lower().replace(" ", "_")
            new_cols[col] = clean_name
    return df_copy.rename(columns=new_cols)


def encode_gender(val: Union[str, int, float]) -> int:
    """Encode gender to integer: MALE/male -> 1, FEMALE/female -> 0."""
    if isinstance(val, (int, float, np.integer)):
        if int(val) in (0, 1):
            return int(val)
        raise ValueError(f"Invalid numeric value for gender: {val}. Expected 0 (Female) or 1 (Male).")
    val_str = str(val).strip().upper()
    if val_str in ("MALE", "M", "1"):
        return 1
    if val_str in ("FEMALE", "F", "0"):
        return 0
    raise ValueError(f"Unrecognized gender value: {val}. Expected 'MALE' or 'FEMALE'.")


def encode_target(val: Union[str, int, float]) -> int:
    """Encode target: YES/yes -> 1, NO/no -> 0."""
    if isinstance(val, (int, float, np.integer)):
        if int(val) in (0, 1):
            return int(val)
        raise ValueError(f"Invalid numeric value for target: {val}. Expected 0 or 1.")
    val_str = str(val).strip().upper()
    if val_str in ("YES", "Y", "POSITIVE", "1"):
        return 1
    if val_str in ("NO", "N", "NEGATIVE", "0"):
        return 0
    raise ValueError(f"Unrecognized target value: {val}. Expected 'YES' or 'NO'.")


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that standardizes AGE and passes through binary features.
    
    Returns:
        Fitted or unfitted ColumnTransformer instance.
    """
    binary_cols = [c for c in FEATURE_COLUMNS if c != "age"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_scaler", StandardScaler(), ["age"]),
            ("binary_passthrough", "passthrough", binary_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def handle_duplicates(df: pd.DataFrame, drop_exact: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect and handle exact and feature-only duplicates according to explicit policy.
    
    Policy:
    1. Exact full-row duplicates (where all features + target are identical)
       are removed when drop_exact=True. This avoids identical records leaking
       across cross-validation and test partitions.
    2. Feature-only duplicates with conflicting labels are preserved, documented,
       and monitored because they reflect real-world clinical uncertainty (patients
       with identical survey symptoms having different biopsy-verified outcomes).
    
    Args:
        df: Cleaned dataframe with standardized column names.
        drop_exact: If True, drops duplicate rows across all columns.
        
    Returns:
        (processed_df, duplicate_report)
    """
    total_raw = len(df)
    exact_duplicate_mask = df.duplicated()
    exact_duplicate_count = int(exact_duplicate_mask.sum())
    
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    feature_duplicate_mask = df.duplicated(subset=feature_cols, keep=False)
    feature_duplicate_count = int(feature_duplicate_mask.sum())
    
    # Check conflicting feature profiles
    conflicting_profiles = []
    if feature_duplicate_count > 0:
        grouped = df[feature_duplicate_mask].groupby(feature_cols)[TARGET_COLUMN].nunique()
        conflict_groups = grouped[grouped > 1]
        for idx in conflict_groups.index:
            conflicting_profiles.append(idx)
            
    if drop_exact:
        df_processed = df.drop_duplicates().copy().reset_index(drop=True)
    else:
        df_processed = df.copy().reset_index(drop=True)
        
    report = {
        "total_rows_before": total_raw,
        "exact_duplicates_detected": exact_duplicate_count,
        "exact_duplicates_removed": exact_duplicate_count if drop_exact else 0,
        "total_rows_after": len(df_processed),
        "conflicting_feature_profiles_count": len(conflicting_profiles),
        "policy": "Remove exact full-row duplicates prior to train/test split to prevent data leakage; preserve conflicting profile as genuine clinical ambiguity."
    }
    
    return df_processed, report


def load_dataset(csv_path: str, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load and preprocess the dataset from CSV file.
    
    Args:
        csv_path: Path to survey_lung_cancer.csv.
        drop_duplicates: Whether to drop exact duplicate rows.
        
    Returns:
        X (pd.DataFrame): 15 feature columns.
        y (pd.Series): Binary target (0 = NO, 1 = YES).
        meta (dict): Ingestion and preprocessing metadata report.
    """
    raw_df = pd.read_csv(csv_path)
    clean_df = standardize_column_names(raw_df)
    
    # Apply duplicate policy
    dedup_df, dup_report = handle_duplicates(clean_df, drop_exact=drop_duplicates)
    
    # Encode categorical features
    dedup_df["gender"] = dedup_df["gender"].apply(encode_gender)
    dedup_df[TARGET_COLUMN] = dedup_df[TARGET_COLUMN].apply(encode_target)
    
    # Ensure binary features are integer 0 or 1
    for col in FEATURE_COLUMNS:
        if col != "gender" and col != "age":
            dedup_df[col] = dedup_df[col].astype(int)
    dedup_df["age"] = dedup_df["age"].astype(float)
    
    X = dedup_df[FEATURE_COLUMNS].copy()
    y = dedup_df[TARGET_COLUMN].copy()
    
    metadata = {
        "raw_shape": raw_df.shape,
        "processed_shape": dedup_df.shape,
        "duplicate_handling": dup_report,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "target_distribution": {
            "negative_NO_0": int((y == 0).sum()),
            "positive_YES_1": int((y == 1).sum()),
            "positive_rate": float((y == 1).mean()),
        },
    }
    
    return X, y, metadata


def format_single_input(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """Format and validate a single input dictionary into a DataFrame suitable for the preprocessor.
    
    Args:
        raw_input: Dictionary containing feature values.
        
    Returns:
        pd.DataFrame with 1 row and standardized columns.
        
    Raises:
        ValueError: If required features are missing or invalid.
    """
    normalized_input: Dict[str, Any] = {}
    for k, v in raw_input.items():
        clean_k = RAW_TO_CLEAN_MAP.get(k, k.strip().lower().replace(" ", "_"))
        normalized_input[clean_k] = v
        
    # Check for missing features
    missing = [f for f in FEATURE_COLUMNS if f not in normalized_input]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
        
    # Encode and validate values
    row: Dict[str, Any] = {}
    row["gender"] = encode_gender(normalized_input["gender"])
    
    try:
        age_val = float(normalized_input["age"])
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid age value '{normalized_input['age']}': must be a valid number.") from err
        
    if not (1 <= age_val <= 120):
        raise ValueError(f"Age {age_val} is out of realistic clinical range [1, 120].")
    row["age"] = age_val
    
    for col in FEATURE_COLUMNS:
        if col in ("gender", "age"):
            continue
        val = normalized_input[col]
        try:
            val_int = int(val)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid value for binary feature '{col}': {val}. Must be 0 or 1.") from err
        if val_int not in (0, 1):
            raise ValueError(f"Value for binary feature '{col}' must be 0 or 1, got {val_int}.")
        row[col] = val_int
        
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
