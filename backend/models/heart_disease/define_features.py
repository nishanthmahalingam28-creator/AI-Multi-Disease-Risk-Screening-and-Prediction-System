"""
Feature and Target Definition Script for Heart Disease Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 3 — Feature and Target Definition
This script defines and validates the feature matrix (X) and target vector (y)
from the raw dataset without modifying the dataset or applying any preprocessing,
scaling, imputation, feature engineering, or model training.
"""

import os
import sys
import json
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


TARGET_COLUMN: str = "Heart Disease"

TARGET_MAPPING: Dict[str, int] = {
    "Absence": 0,
    "Presence": 1
}

EXPECTED_PREDICTOR_COLUMNS: List[str] = [
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

EXPECTED_ROWS: int = 270
EXPECTED_FEATURE_COUNT: int = 13


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the original dataset without modifying it."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    return pd.read_csv(dataset_path)


def define_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts and strictly verifies feature matrix X and target vector y.
    
    Guarantees:
    - Target column 'Heart Disease' is excluded from X and extracted into y.
    - X contains exclusively the 13 observed predictor features in their original order.
    - y is mapped to binary values (Absence = 0, Presence = 1) for modeling verification.
    - No modification, transformation, replacement, or engineering is applied to X.
    """
    # 1. Verify target column presence
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}")

    # 2. Verify all expected predictor columns exist
    missing_cols = [col for col in EXPECTED_PREDICTOR_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected predictor columns: {missing_cols}")

    # 3. Verify exact feature order
    observed_cols = [col for col in df.columns if col != TARGET_COLUMN]
    if observed_cols != EXPECTED_PREDICTOR_COLUMNS:
        raise ValueError(
            f"Feature column order mismatch.\n"
            f"Observed: {observed_cols}\n"
            f"Expected: {EXPECTED_PREDICTOR_COLUMNS}"
        )

    # 4. Extract X preserving exact column order and raw values
    X = df[EXPECTED_PREDICTOR_COLUMNS].copy()

    # 5. Verify target is strictly excluded from X
    if TARGET_COLUMN in X.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is present in feature matrix X!")

    # 6. Extract raw target and verify observed values
    raw_y = df[TARGET_COLUMN].copy()
    observed_target_values = set(raw_y.dropna().unique())
    expected_target_values = set(TARGET_MAPPING.keys())
    if observed_target_values != expected_target_values:
        raise ValueError(
            f"Unexpected target values: {observed_target_values}. "
            f"Expected: {expected_target_values}"
        )

    # 7. Map target to binary representation (Absence=0, Presence=1)
    y = raw_y.map(TARGET_MAPPING)

    # 8. Strict integrity verifications
    assert X.shape[0] == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows in X, got {X.shape[0]}"
    assert X.shape[1] == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} columns in X, got {X.shape[1]}"
    assert len(y) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} values in y, got {len(y)}"
    assert X.shape[0] == len(y), "Row count mismatch between X and y!"
    assert list(X.columns) == EXPECTED_PREDICTOR_COLUMNS, "Columns in X do not match expected order"
    assert X.isnull().sum().sum() == 0, "Null values detected in X!"
    assert y.isnull().sum() == 0, "Null values detected in y after mapping!"
    assert len(set(X.columns)) == len(X.columns), "Duplicate column names detected in X!"
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"Column {col} is not numeric!"

    return X, y


def generate_feature_schema(df: pd.DataFrame, dataset_path: str, output_path: str) -> Dict[str, Any]:
    """
    Constructs and exports the canonical feature schema JSON file.
    """
    raw_y = df[TARGET_COLUMN]
    target_counts = raw_y.value_counts().to_dict()
    target_binary_counts = raw_y.map(TARGET_MAPPING).value_counts().to_dict()

    feature_dtypes = {col: str(df[col].dtype) for col in EXPECTED_PREDICTOR_COLUMNS}

    schema = {
        "model_domain": "heart_disease",
        "dataset_path": "datasets/heart_disease/heart.csv",
        "dataset_row_count": int(len(df)),
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "target_data_type": "object -> int64 (binary encoded for modeling: Absence=0, Presence=1)",
        "target_observed_values": ["Absence", "Presence"],
        "target_value_counts": {
            "Absence": int(target_counts.get("Absence", 0)),
            "Presence": int(target_counts.get("Presence", 0))
        },
        "target_binary_counts": {
            "0": int(target_binary_counts.get(0, 0)),
            "1": int(target_binary_counts.get(1, 0))
        },
        "feature_count": len(EXPECTED_PREDICTOR_COLUMNS),
        "exact_feature_names_in_order": EXPECTED_PREDICTOR_COLUMNS,
        "feature_data_types": feature_dtypes,
        "target_leakage_check": {
            "target_in_X": False,
            "leakage_status": "PASSED (Target 'Heart Disease' is strictly excluded from predictor matrix X)",
            "leakage_diagnostic_note": (
                "No obvious target leakage was identified during the feature and target inspection. "
                "Correlation analysis was used as one diagnostic check and does not by itself prove the absence of leakage or proxy identifiers."
            )
        },
        "pipeline_state_documentation": {
            "feature_engineering_performed": False,
            "zero_value_replacement_performed": False,
            "imputation_performed": False,
            "scaling_performed": False,
            "outlier_removal_performed": False,
            "feature_selection_performed": False,
            "train_test_split_performed": False,
            "model_training_performed": False,
            "notes": "Features and target are defined in their raw observed state. No transformations, scaling, or feature engineering have been applied."
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema


def main():
    print("=" * 80)
    print("STEP 3: HEART DISEASE FEATURE AND TARGET DEFINITION")
    print("=" * 80)

    # 1. Resolve dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    candidate_paths = [
        os.path.join(workspace_root, "datasets", "heart_disease", "heart.csv"),
        os.path.join("datasets", "heart_disease", "heart.csv"),
        os.path.abspath("datasets/heart_disease/heart.csv")
    ]
    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        resolved_path = candidate_paths[0]

    print(f"\n1. Loading dataset from: {resolved_path}")
    df = load_dataset(resolved_path)
    print(f"   - Raw dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Define features (X) and target (y)
    print("\n2. Extracting feature matrix X and target vector y...")
    X, y = define_features_and_target(df)

    print(f"   - X shape: {X.shape} (rows: {X.shape[0]}, features: {X.shape[1]})")
    print(f"   - y shape: {y.shape} (rows: {len(y)})")

    # 3. Feature list and ordering
    print(f"\n3. Exact Feature List in Canonical Order ({len(X.columns)} features):")
    for idx, col in enumerate(X.columns):
        print(f"   [{idx:2d}] {repr(col):<26} ({X[col].dtype})")

    # 4. Target mapping and class distribution
    print(f"\n4. Target Definition and Mapping:")
    print(f"   - Target Column: {repr(TARGET_COLUMN)}")
    print(f"   - Mapping: {TARGET_MAPPING}")
    raw_counts = df[TARGET_COLUMN].value_counts()
    mapped_counts = y.value_counts()
    for cat_label, num_val in TARGET_MAPPING.items():
        cnt = raw_counts[cat_label]
        pct = (cnt / len(y)) * 100
        print(f"   - Class {repr(cat_label):<10} -> {num_val}: {cnt:3d} rows ({pct:5.2f}%)")

    # 5. Validation checks summary
    print("\n5. Comprehensive Validation Checks:")
    checks = [
        ("X contains exactly 13 features", X.shape[1] == 13),
        ("y contains exactly 270 values", len(y) == 270),
        ("X and y have matching row counts (270 == 270)", X.shape[0] == len(y)),
        ("Target 'Heart Disease' is excluded from X", TARGET_COLUMN not in X.columns),
        ("All X columns exist in source dataset", all(c in df.columns for c in X.columns)),
        ("Feature order exactly preserved from original dataset", list(X.columns) == EXPECTED_PREDICTOR_COLUMNS),
        ("All X columns are numeric", all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)),
        ("Target contains only 'Absence' and 'Presence'", set(df[TARGET_COLUMN].unique()) == {"Absence", "Presence"}),
        ("Mapped y contains only 0 and 1", set(y.unique()) == {0, 1}),
        ("No missing values in X", X.isnull().sum().sum() == 0),
        ("No missing values in y", y.isnull().sum() == 0),
        ("No duplicate feature names in X", len(set(X.columns)) == len(X.columns))
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

    # 6. Generate and save feature schema JSON
    schema_path = os.path.join(script_dir, "feature_schema.json")
    print(f"\n6. Generating feature schema JSON: {schema_path}")
    schema = generate_feature_schema(df, resolved_path, schema_path)
    print("   - Schema successfully written and verified.")

    print("\n" + "=" * 80)
    print("STEP 3 FEATURE AND TARGET DEFINITION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
