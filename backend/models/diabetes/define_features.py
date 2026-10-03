"""
Feature and Target Definition Script for Diabetes Model
AI Multi-Disease Risk Screening and Prediction System

Dataset: datasets/diabetes/diabetes.csv
Target: Outcome (Observed values: 0, 1)
Predictors (8 observed features in original order):
  1. Pregnancies
  2. Glucose
  3. BloodPressure
  4. SkinThickness
  5. Insulin
  6. BMI
  7. DiabetesPedigreeFunction
  8. Age
"""

import os
import json
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


TARGET_COLUMN: str = "Outcome"

EXPECTED_PREDICTOR_COLUMNS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

EXPECTED_TARGET_VALUES: List[int] = [0, 1]
EXPECTED_ROWS: int = 768
EXPECTED_COLS: int = 8


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the original dataset without modifying it."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return df


def define_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts and strictly verifies feature matrix X and target vector y.
    
    Guarantees:
    - Target column 'Outcome' is extracted into y.
    - X contains exclusively the 8 observed predictor features in their original order.
    - Target is strictly excluded from X.
    - No modification, transformation, replacement, or engineering is applied.
    """
    # 1. Target column validation
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}")

    y = df[TARGET_COLUMN].copy()

    # Confirm y observed values are strictly 0 and 1
    observed_y_values = sorted(list(y.unique()))
    if observed_y_values != EXPECTED_TARGET_VALUES:
        raise ValueError(f"Target contains unexpected values: {observed_y_values}. Expected: {EXPECTED_TARGET_VALUES}")

    # 2. Predictor column validation
    for col in EXPECTED_PREDICTOR_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Expected predictor '{col}' not found in dataset columns: {list(df.columns)}")

    # Extract X with exact expected order
    X = df[EXPECTED_PREDICTOR_COLUMNS].copy()

    # 3. Shape and integrity validations
    assert TARGET_COLUMN not in X.columns, f"CRITICAL: Target '{TARGET_COLUMN}' must NOT be in X!"
    assert list(X.columns) == EXPECTED_PREDICTOR_COLUMNS, "Column order mismatch in X!"
    assert len(set(X.columns)) == len(EXPECTED_PREDICTOR_COLUMNS), "Duplicate columns detected in X!"
    assert X.shape == (EXPECTED_ROWS, EXPECTED_COLS), f"X shape {X.shape} != expected ({EXPECTED_ROWS}, {EXPECTED_COLS})"
    assert y.shape == (EXPECTED_ROWS,), f"y shape {y.shape} != expected ({EXPECTED_ROWS},)"

    # Confirm all columns in X are numeric
    non_numeric_cols = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    assert len(non_numeric_cols) == 0, f"Non-numeric columns in X: {non_numeric_cols}"

    return X, y


def generate_feature_schema(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, output_path: str) -> Dict[str, Any]:
    """
    Constructs and saves the canonical feature schema for the Diabetes model.
    """
    data_types = {col: str(X[col].dtype) for col in X.columns}
    target_val_counts = {str(k): int(v) for k, v in y.value_counts().sort_index().to_dict().items()}
    target_observed_values = [int(v) for v in sorted(y.unique())]

    schema = {
        "model_domain": "diabetes",
        "dataset_path": "datasets/diabetes/diabetes.csv",
        "dataset_row_count": len(df),
        "target_column": TARGET_COLUMN,
        "target_data_type": str(y.dtype),
        "target_observed_values": target_observed_values,
        "target_value_counts": target_val_counts,
        "feature_count": len(EXPECTED_PREDICTOR_COLUMNS),
        "exact_feature_names_in_order": EXPECTED_PREDICTOR_COLUMNS,
        "feature_data_types": data_types,
        "target_leakage_check": {
            "target_in_X": TARGET_COLUMN in X.columns,
            "leakage_status": "PASSED (Target 'Outcome' is strictly excluded from predictor matrix X)"
        },
        "pipeline_state_documentation": {
            "feature_engineering_performed": False,
            "zero_value_replacement_performed": False,
            "imputation_performed": False,
            "scaling_performed": False,
            "outlier_removal_performed": False,
            "feature_selection_performed": False,
            "notes": "Features and target are preserved in their raw observed state. No transformations have been applied."
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema


def run_feature_definition(dataset_path: str, schema_path: str):
    print("=" * 80)
    print("STEP 3: DEFINE DIABETES MODEL FEATURES (X) AND TARGET (y)")
    print("=" * 80)

    # 1. Load dataset
    abs_dataset_path = os.path.abspath(dataset_path)
    print(f"Loading dataset from: {abs_dataset_path}")
    df = load_dataset(abs_dataset_path)
    print(f"Original dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Define X and y
    X, y = define_features_and_target(df)
    print("\nTarget (y) Definition:")
    print(f"   - Target column: '{TARGET_COLUMN}'")
    print(f"   - Target observed values: {[int(v) for v in sorted(list(y.unique()))]}")
    print(f"   - Target value counts: {y.value_counts().to_dict()}")
    print(f"   - Target data type: {y.dtype}")
    print(f"   - Target shape: {y.shape}")

    print("\nFeature Matrix (X) Definition:")
    print(f"   - Number of features: {X.shape[1]}")
    print(f"   - Feature matrix shape: {X.shape}")
    print("   - Exact feature names and data types in order:")
    for idx, col in enumerate(X.columns):
        print(f"     [{idx}] '{col}' ({X[col].dtype})")

    # 3. Confirmations
    print("\n--- Integrity & Validation Confirmations ---")
    print(f"   [1] X shape == (768, 8): {X.shape == (768, 8)} (Actual: {X.shape})")
    print(f"   [2] y shape == (768,): {y.shape == (768,)} (Actual: {y.shape})")
    print(f"   [3] All columns in X are numeric: {all(np.issubdtype(X[c].dtype, np.number) for c in X.columns)}")
    print(f"   [4] y contains only observed values 0 and 1: {set(y.unique()) == {0, 1}}")
    print(f"   [5] No feature is duplicated: {len(X.columns) == len(set(X.columns))}")
    print(f"   [6] Exact original column order preserved: {list(X.columns) == EXPECTED_PREDICTOR_COLUMNS}")
    print(f"   [7] Target leakage check (Outcome in X): {TARGET_COLUMN in X.columns} (Target is excluded)")

    # 4. Explicit documentation of untouched state
    print("\n--- Explicit State Documentation ---")
    print("   - Feature engineering performed: None")
    print("   - Zero-value replacement performed: None (All raw zeros retained as-is)")
    print("   - Imputation performed: None")
    print("   - Scaling performed: None")
    print("   - Outlier removal performed: None")
    print("   - Feature selection performed: None")

    # 5. Generate and save schema
    schema = generate_feature_schema(df, X, y, schema_path)
    abs_schema_path = os.path.abspath(schema_path)
    print(f"\nFeature schema saved to: {abs_schema_path}")

    print("=" * 80)
    print("STEP 3 COMPLETED SUCCESSFULLY - NO TRAINING OR PREPROCESSING PERFORMED")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "..", "..", "..", "datasets", "diabetes", "diabetes.csv"),
        os.path.join("datasets", "diabetes", "diabetes.csv"),
        os.path.abspath("datasets/diabetes/diabetes.csv")
    ]

    selected_dataset = None
    for p in candidate_paths:
        if os.path.exists(p):
            selected_dataset = p
            break

    if selected_dataset is None:
        selected_dataset = candidate_paths[0]

    out_schema = os.path.join(script_dir, "feature_schema.json")
    run_feature_definition(selected_dataset, out_schema)
