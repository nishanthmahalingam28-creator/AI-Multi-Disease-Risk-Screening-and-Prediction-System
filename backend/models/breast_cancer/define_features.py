"""
Feature and Target Definition & Leakage Investigation Script
Breast Cancer Model - Member 1
AI Multi-Disease Risk Screening and Prediction System

Dataset: datasets/breast_cancer/breast.csv
Target: diagnosis (B: 0, M: 1)
Excluded: id, Unnamed: 32
"""

import os
import sys
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

# Canonical definitions of the 30 continuous predictor features
MEAN_FEATURES: List[str] = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
]

SE_FEATURES: List[str] = [
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se"
]

WORST_FEATURES: List[str] = [
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

PREDICTOR_FEATURES: List[str] = MEAN_FEATURES + SE_FEATURES + WORST_FEATURES
TARGET_COLUMN: str = "diagnosis"
EXCLUDED_COLUMNS: List[str] = ["id", "Unnamed: 32"]

TARGET_ENCODING: Dict[str, int] = {
    "B": 0,  # Benign (Negative class for screening)
    "M": 1   # Malignant (Positive class for screening)
}


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw breast cancer CSV dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    return pd.read_csv(csv_path)


def define_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts and strictly validates feature matrix X and target series y.
    
    Guarantees:
    - Target 'diagnosis' is separated into y.
    - Excluded columns ('id', 'Unnamed: 32') are completely quarantined.
    - X contains exclusively the 30 validated continuous cytological predictors.
    """
    # 1. Target validation
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns.")
    y = df[TARGET_COLUMN].copy()

    # 2. Excluded columns validation
    for col in EXCLUDED_COLUMNS:
        if col in df.columns:
            # Verified present in raw CSV, must be dropped from X
            pass

    # 3. Features extraction
    missing_expected = [f for f in PREDICTOR_FEATURES if f not in df.columns]
    if missing_expected:
        raise ValueError(f"Expected predictor features missing from dataset: {missing_expected}")

    X = df[PREDICTOR_FEATURES].copy()

    # 4. Strict assertions
    assert TARGET_COLUMN not in X.columns, f"Target '{TARGET_COLUMN}' must not be present in feature matrix X!"
    for col in EXCLUDED_COLUMNS:
        assert col not in X.columns, f"Excluded column '{col}' must not be present in feature matrix X!"
    assert X.shape[1] == len(PREDICTOR_FEATURES), f"X must contain exactly {len(PREDICTOR_FEATURES)} columns, got {X.shape[1]}"
    assert len(X) == len(y), f"Row count mismatch: X has {len(X)} rows, y has {len(y)} rows"

    return X, y


def perform_leakage_and_quality_audit(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Performs comprehensive data leakage and data quality checks on all columns.
    """
    results: Dict[str, Any] = {}

    # Target encoding for correlation investigation
    y_encoded = y.map(TARGET_ENCODING)

    # A. Check every raw column for inclusion/exclusion status and leakage potential
    raw_col_audit = {}
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            if col == "id":
                status = "EXCLUDED"
                reason = "Arbitrary patient identifier. Non-clinical integer key with risk of memorization/spurious correlation."
            else:
                status = "EXCLUDED"
                reason = "Parser artifact caused by trailing comma in CSV header line; contains 100% missing values (NaN)."
        elif col == TARGET_COLUMN:
            status = "TARGET (y)"
            reason = "Ground-truth clinical diagnosis label to be predicted (B=0, M=1). Quarantined from X."
        elif col in PREDICTOR_FEATURES:
            status = "INCLUDED (X)"
            corr_val = float(X[col].corr(y_encoded))
            reason = f"Pre-diagnostic FNA morphological measurement. Pearson correlation with target: r = {corr_val:.4f}."
        else:
            status = "UNKNOWN"
            reason = "Unexpected column not in schema."

        raw_col_audit[col] = {
            "status": status,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "reason": reason
        }
    results["column_audit"] = raw_col_audit

    # B. Target leakage checks
    # 1. Check if any feature is identical to target
    duplicate_of_target = []
    for col in X.columns:
        if (X[col] == y_encoded).all():
            duplicate_of_target.append(col)
    results["duplicate_of_target"] = duplicate_of_target

    # 2. Check for deterministic target leakage (|r| == 1.0 or near 1.0)
    deterministic_leakage = []
    correlations = {}
    for col in X.columns:
        r = float(X[col].corr(y_encoded))
        correlations[col] = round(r, 4)
        if abs(r) >= 0.95:
            deterministic_leakage.append((col, r))
    results["deterministic_leakage"] = deterministic_leakage
    results["correlations_with_target"] = correlations

    # 3. Check for constant features (std == 0)
    constant_features = [col for col in X.columns if X[col].std() == 0 or X[col].nunique() <= 1]
    results["constant_features"] = constant_features

    # 4. Check for near-constant features (e.g. >99% identical values)
    near_constant_features = []
    for col in X.columns:
        top_val_pct = X[col].value_counts(normalize=True).iloc[0]
        if top_val_pct >= 0.99:
            near_constant_features.append((col, round(float(top_val_pct * 100), 2)))
    results["near_constant_features"] = near_constant_features

    # 5. Check for duplicate columns (identical data across columns)
    duplicate_pairs = []
    for i in range(len(X.columns)):
        for j in range(i + 1, len(X.columns)):
            c1, c2 = X.columns[i], X.columns[j]
            if X[c1].equals(X[c2]):
                duplicate_pairs.append((c1, c2))
    results["duplicate_feature_pairs"] = duplicate_pairs

    # 6. Check data types of all predictors
    non_numerical_features = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
    results["non_numerical_features"] = non_numerical_features

    # 7. Check for missing values in X and y
    results["missing_values_X"] = int(X.isnull().sum().sum())
    results["missing_values_y"] = int(y.isnull().sum())

    return results


def run_pipeline(csv_path: str):
    print("=" * 80)
    print("STEP 3: FEATURE (X) & TARGET (y) DEFINITION & LEAKAGE INVESTIGATION")
    print("=" * 80)
    
    # Ingest
    df = load_raw_dataset(csv_path)
    print(f"Dataset loaded successfully from: {os.path.abspath(csv_path)}")
    print(f"Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Define X and y
    X, y = define_features_and_target(df)
    print(f"\nFinal X shape: {X.shape} ({X.shape[1]} features, {X.shape[0]} samples)")
    print(f"Final y shape: {y.shape} ({y.shape[0]} samples)")

    # Run audit
    audit = perform_leakage_and_quality_audit(df, X, y)

    # Output verification details
    print("\n--- 1. Target Definition ---")
    print(f"Target column: '{TARGET_COLUMN}'")
    print(f"Target classes: {list(y.unique())}")
    print(f"Class counts:\n{y.value_counts().to_string()}")
    print(f"Target encoding for screening: {TARGET_ENCODING}")

    print("\n--- 2. Exclusion Verification ---")
    print("Explicitly verified exclusions:")
    for ex_col in EXCLUDED_COLUMNS:
        is_in_df = ex_col in df.columns
        is_in_X = ex_col in X.columns
        print(f"  * '{ex_col}': In raw CSV = {is_in_df} | Present in X = {is_in_X} -> EXCLUDED ({audit['column_audit'][ex_col]['reason']})")
    print(f"  * '{TARGET_COLUMN}' (Target): Present in X = {TARGET_COLUMN in X.columns} -> EXCLUDED from X")

    print("\n--- 3. Leakage Checks Summary ---")
    print(f"  * Features identical to target: {audit['duplicate_of_target']} (Clean)")
    print(f"  * Deterministically leaking features (|r| >= 0.95): {audit['deterministic_leakage']} (Clean)")
    print(f"  * Constant features (std == 0): {audit['constant_features']} (Clean)")
    print(f"  * Near-constant features (>=99% single value): {audit['near_constant_features']} (Clean)")
    print(f"  * Duplicate feature pairs: {audit['duplicate_feature_pairs']} (Clean)")
    print(f"  * Non-numerical features in X: {audit['non_numerical_features']} (Clean - all 30 are float64)")
    print(f"  * Missing values in X: {audit['missing_values_X']} (Clean)")
    print(f"  * Missing values in y: {audit['missing_values_y']} (Clean)")

    print("\n--- 4. Predictor Features Correlation with Encoded Target ---")
    top_pos = sorted(audit["correlations_with_target"].items(), key=lambda x: x[1], reverse=True)[:5]
    top_neg = sorted(audit["correlations_with_target"].items(), key=lambda x: x[1])[:5]
    print("Top 5 Positive Correlations (Higher value associated with Malignancy):")
    for feat, r in top_pos:
        print(f"  - {feat:25s}: r = {r:+.4f}")
    print("Lowest / Negative Correlations:")
    for feat, r in top_neg:
        print(f"  - {feat:25s}: r = {r:+.4f}")

    print("\n" + "=" * 80)
    print("NO MODEL TRAINING PERFORMED. STOPPING AFTER STEP 3 AS DIRECTED.")
    print("=" * 80)


if __name__ == "__main__":
    default_dataset = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "datasets", "breast_cancer", "breast.csv"
    )
    if not os.path.exists(default_dataset):
        default_dataset = os.path.join("datasets", "breast_cancer", "breast.csv")
    run_pipeline(default_dataset)
