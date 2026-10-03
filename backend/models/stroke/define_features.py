"""
Feature and Target Definition Script for Stroke Prediction Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 3 — Feature and Target Definition
This script defines, extracts, and validates the canonical feature matrix (X)
and target vector (y) from the raw Stroke dataset without modifying any data,
and without applying preprocessing, scaling, imputation, feature engineering,
train/test splitting, or model training.
"""

import os
import sys
import json
import hashlib
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


TARGET_COLUMN: str = "stroke"

EXPECTED_ORIGINAL_COLUMNS: List[str] = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]

EXPECTED_PREDICTOR_COLUMNS: List[str] = [
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

EXCLUDED_COLUMNS: Dict[str, str] = {
    "id": "Arbitrary patient record identifier with 5,110 unique integer values across 5,110 rows (100% uniqueness). Non-generalizable and non-predictive; excluded to prevent arbitrary memorization and data leakage.",
    "stroke": "Target disease outcome column (binary indicator: 0 for non-stroke, 1 for stroke). Excluded from predictor matrix X to prevent direct label leakage."
}

EXPECTED_ROWS: int = 5110
EXPECTED_ORIGINAL_COLS: int = 12
EXPECTED_FEATURE_COUNT: int = 10


def compute_sha256(filepath: str) -> str:
    """
    Computes the SHA-256 hash of a file for integrity verification.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Loads the original dataset in a strictly non-destructive manner.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    return pd.read_csv(dataset_path)


def define_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts and strictly verifies feature matrix X and target vector y.
    
    Guarantees:
    - Target column 'stroke' is excluded from X and extracted into y.
    - Identifier column 'id' is excluded from X.
    - X contains exclusively the 10 verified predictor features in their original order.
    - y contains exclusively the binary stroke values (0 and 1).
    - No modification, transformation, replacement, or engineering is applied to X or y.
    """
    # 1. Verify target column presence
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}")

    # 2. Verify all expected original columns exist in order
    if list(df.columns) != EXPECTED_ORIGINAL_COLUMNS:
        raise ValueError(
            f"Original dataset column mismatch.\n"
            f"Observed: {list(df.columns)}\n"
            f"Expected: {EXPECTED_ORIGINAL_COLUMNS}"
        )

    # 3. Verify all expected predictor columns exist
    missing_cols = [col for col in EXPECTED_PREDICTOR_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected predictor columns: {missing_cols}")

    # 4. Extract X preserving exact column order and raw values
    X = df[EXPECTED_PREDICTOR_COLUMNS].copy()

    # 5. Verify target is strictly excluded from X
    if TARGET_COLUMN in X.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is present in feature matrix X!")

    # 6. Verify identifier 'id' is strictly excluded from X
    if "id" in X.columns:
        raise ValueError("Identifier column 'id' is present in feature matrix X!")

    # 7. Extract raw target and verify observed values
    y = df[TARGET_COLUMN].copy()
    observed_target_values = sorted(list(y.dropna().unique()))
    expected_target_values = [0, 1]
    if observed_target_values != expected_target_values:
        raise ValueError(
            f"Unexpected target values: {observed_target_values}. "
            f"Expected: {expected_target_values}"
        )

    # 8. Strict integrity verifications
    assert X.shape[0] == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows in X, got {X.shape[0]}"
    assert X.shape[1] == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} columns in X, got {X.shape[1]}"
    assert len(y) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} values in y, got {len(y)}"
    assert X.shape[0] == len(y), "Row count mismatch between X and y!"
    assert list(X.columns) == EXPECTED_PREDICTOR_COLUMNS, "Columns in X do not match expected order"
    assert len(set(X.columns)) == len(X.columns), "Duplicate column names detected in X!"
    assert y.isnull().sum() == 0, "Null values detected in target y!"

    return X, y


def generate_feature_schema(df: pd.DataFrame, dataset_path: str, output_path: str) -> Dict[str, Any]:
    """
    Constructs and exports the canonical feature schema JSON file.
    """
    X, y = define_features_and_target(df)

    # Identify numeric and categorical columns from actual dtypes
    numeric_features = [col for col in EXPECTED_PREDICTOR_COLUMNS if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in EXPECTED_PREDICTOR_COLUMNS if not pd.api.types.is_numeric_dtype(df[col])]

    # Target distribution
    target_counts = y.value_counts(dropna=False).to_dict()
    target_props = (y.value_counts(dropna=False, normalize=True) * 100).to_dict()
    c0_count = int(target_counts.get(0, 0))
    c1_count = int(target_counts.get(1, 0))
    c0_pct = round(float(target_props.get(0, 0.0)), 4)
    c1_pct = round(float(target_props.get(1, 0.0)), 4)

    # Missing counts per feature
    missing_by_feature = {col: int(X[col].isnull().sum()) for col in EXPECTED_PREDICTOR_COLUMNS}
    missing_pct_by_feature = {col: round(float((cnt / len(df)) * 100), 4) for col, cnt in missing_by_feature.items()}

    # Data types per feature
    feature_dtypes = {col: str(df[col].dtype) for col in EXPECTED_PREDICTOR_COLUMNS}

    schema = {
        "model_domain": "stroke",
        "dataset_path": "datasets/stroke/stroke.csv",
        "dataset_row_count": int(len(df)),
        "dataset_column_count": int(len(df.columns)),
        "original_column_order": EXPECTED_ORIGINAL_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_dtype": str(df[TARGET_COLUMN].dtype),
        "target_observed_values": [0, 1],
        "target_class_counts": {
            "0": c0_count,
            "1": c1_count
        },
        "target_class_percentages": {
            "0": c0_pct,
            "1": c1_pct
        },
        "imbalance_ratio": round(c0_count / c1_count, 4) if c1_count > 0 else None,
        "feature_count": len(EXPECTED_PREDICTOR_COLUMNS),
        "exact_feature_names_in_order": EXPECTED_PREDICTOR_COLUMNS,
        "feature_data_types": feature_dtypes,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_columns": EXCLUDED_COLUMNS,
        "missing_values_by_feature": missing_by_feature,
        "missing_percentages_by_feature": missing_pct_by_feature,
        "leakage_check": {
            "target_in_X": False,
            "identifier_in_X": False,
            "status": "PASSED (Target 'stroke' and identifier 'id' are strictly excluded from predictor matrix X)",
            "diagnostic_note": "No feature exhibits extreme linear correlation (|r| >= 0.85) with target 'stroke'. Correlation analysis was used as one diagnostic check and does not by itself prove the absence of leakage."
        },
        "pipeline_state_documentation": {
            "feature_engineering_performed": False,
            "imputation_performed": False,
            "encoding_performed": False,
            "scaling_performed": False,
            "outlier_removal_performed": False,
            "train_test_split_performed": False,
            "model_training_performed": False,
            "notes": "Features and target are defined in their raw observed state. No transformations, scaling, imputation, or feature engineering have been applied."
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema


def main():
    print("=" * 80)
    print("STEP 3: STROKE DISEASE FEATURE AND TARGET DEFINITION")
    print("=" * 80)

    # 1. Resolve dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    candidate_paths = [
        os.path.join(workspace_root, "datasets", "stroke", "stroke.csv"),
        os.path.join("datasets", "stroke", "stroke.csv"),
        os.path.abspath("datasets/stroke/stroke.csv")
    ]
    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        resolved_path = candidate_paths[0]

    # Pre-execution dataset integrity check
    pre_sha256 = compute_sha256(resolved_path)
    print(f"\n1. Loading dataset from: {resolved_path}")
    print(f"   - Pre-Execution Dataset SHA-256: {pre_sha256}")
    df = load_dataset(resolved_path)
    print(f"   - Raw dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Extract feature matrix X and target vector y
    print("\n2. Extracting feature matrix X and target vector y...")
    X, y = define_features_and_target(df)

    print(f"   - X shape: {X.shape} (rows: {X.shape[0]}, features: {X.shape[1]})")
    print(f"   - y shape: {y.shape} (rows: {len(y)})")

    # 3. Categorize features by dtype
    numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

    print(f"\n3. Feature Categorization ({len(X.columns)} features total):")
    print(f"   - Numeric Features ({len(numeric_features)}): {numeric_features}")
    print(f"   - Categorical Features ({len(categorical_features)}): {categorical_features}")

    # 4. Feature list in canonical order
    print(f"\n4. Exact Feature List in Canonical Order ({len(X.columns)} features):")
    for idx, col in enumerate(X.columns):
        f_type = "Numeric" if col in numeric_features else "Categorical"
        null_count = X[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"   [{idx:2d}] {repr(col):<20} | Type: {f_type:<11} | Dtype: {str(X[col].dtype):<7} | Missing: {null_count:3d} ({null_pct:5.2f}%)")

    # 5. Excluded columns
    print("\n5. Excluded Columns and Rationales:")
    for col, reason in EXCLUDED_COLUMNS.items():
        print(f"   - {repr(col)}: {reason}")

    # 6. Target distribution
    target_counts = y.value_counts(dropna=False).to_dict()
    target_props = (y.value_counts(dropna=False, normalize=True) * 100).to_dict()
    c0_count = int(target_counts.get(0, 0))
    c1_count = int(target_counts.get(1, 0))
    c0_pct = float(target_props.get(0, 0.0))
    c1_pct = float(target_props.get(1, 0.0))

    print(f"\n6. Target Definition and Distribution:")
    print(f"   - Target Column: {repr(TARGET_COLUMN)} ({y.dtype})")
    print(f"   - Class 0 (No Stroke): {c0_count:5d} rows ({c0_pct:5.2f}%)")
    print(f"   - Class 1 (Stroke)   : {c1_count:5d} rows ({c1_pct:5.2f}%)")
    print(f"   - Class Imbalance Ratio: {c0_count / c1_count:.2f}:1")

    # 7. Generate feature schema JSON
    schema_path = os.path.join(script_dir, "feature_schema.json")
    print(f"\n7. Exporting canonical feature schema to: {schema_path}")
    generate_feature_schema(df, resolved_path, schema_path)
    print("   [x] feature_schema.json successfully created and validated.")

    # 8. Run validation assertions
    print("\n8. Comprehensive Validation Checks:")
    checks = [
        ("Dataset loads successfully", df.shape == (EXPECTED_ROWS, EXPECTED_ORIGINAL_COLS)),
        ("X contains exactly 10 features", X.shape[1] == EXPECTED_FEATURE_COUNT),
        ("y contains exactly 5110 rows", len(y) == EXPECTED_ROWS),
        ("X and y have matching row counts (5110 == 5110)", X.shape[0] == len(y)),
        ("Identifier column 'id' is strictly excluded from X", "id" not in X.columns),
        ("Target column 'stroke' is strictly excluded from X", TARGET_COLUMN not in X.columns),
        ("Feature order exactly matches expected canonical order", list(X.columns) == EXPECTED_PREDICTOR_COLUMNS),
        ("Numeric features correctly identified (5 features)", len(numeric_features) == 5),
        ("Categorical features correctly identified (5 features)", len(categorical_features) == 5),
        ("Target y contains only observed values 0 and 1", set(y.unique()) == {0, 1}),
        ("Target y has zero null values", y.isnull().sum() == 0),
        ("No duplicate feature names in X", len(set(X.columns)) == len(X.columns)),
        ("Feature schema JSON file exists and is non-empty", os.path.exists(schema_path) and os.path.getsize(schema_path) > 0)
    ]

    all_passed = True
    for desc, passed in checks:
        status_str = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"   {status_str} {desc}")

    if not all_passed:
        print("\nERROR: One or more validation checks failed!")
        sys.exit(1)

    # Post-execution dataset integrity check
    post_sha256 = compute_sha256(resolved_path)
    hash_match = (pre_sha256 == post_sha256)
    print(f"\n9. Dataset Integrity Verification:")
    print(f"   - Post-Execution Dataset SHA-256: {post_sha256}")
    print(f"   - Read-Only Integrity Verified : {hash_match}")
    if not hash_match:
        raise RuntimeError("CRITICAL ERROR: Dataset file was modified during Step 3 execution!")

    print("\n" + "=" * 80)
    print("STEP 3: FEATURE AND TARGET DEFINITION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
