"""
Train/Test Split Creation and Validation Script
Diabetes Model
AI Multi-Disease Risk Screening and Prediction System

Dataset: datasets/diabetes/diabetes.csv
Split configuration: test_size=0.20, random_state=42, stratify=y
Predictor columns (8):
  Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
Target: Outcome (Observed values: 0, 1)
"""

import os
import sys
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add local directory to path for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from define_features import (
        EXPECTED_PREDICTOR_COLUMNS,
        TARGET_COLUMN,
        load_dataset,
        define_features_and_target,
    )
except ImportError:
    from backend.models.diabetes.define_features import (
        EXPECTED_PREDICTOR_COLUMNS,
        TARGET_COLUMN,
        load_dataset,
        define_features_and_target,
    )


def create_train_test_split(
    dataset_path: str,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Loads dataset, extracts X and y, and performs reproducible 80/20 train/test split.
    
    Guarantees:
    - Zero data leakage: No preprocessing, scaling, or imputation is fitted prior to or during splitting.
    - Stratified partitioning preserves observed target class balance.
    - Train and test partitions are completely disjoint (zero index overlap).
    - Preserves exact 8 predictor columns and their original order.
    """
    # 1. Load dataset without modification
    df = load_dataset(dataset_path)

    # 2. Extract validated X and y
    X, y = define_features_and_target(df)

    # 3. Perform stratified train/test split
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    # 4. Rigorous verification checks
    # Assert column identity and ordering
    assert list(X_train.columns) == list(X_test.columns), "Feature column names or order mismatch between train and test!"
    assert list(X_train.columns) == EXPECTED_PREDICTOR_COLUMNS, "Feature columns do not match canonical schema!"

    # Assert no index overlap (disjoint sets)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)
    assert len(overlap) == 0, f"Critical data leakage: {len(overlap)} overlapping indices detected!"
    assert len(X_train) + len(X_test) == len(df), f"Row loss: {len(X_train)} + {len(X_test)} != {len(df)}"

    # Assert both target classes exist in train and test
    assert set(y_train.unique()) == {0, 1}, "Training set does not contain both target classes!"
    assert set(y_test.unique()) == {0, 1}, "Test set does not contain both target classes!"

    # Assert target is excluded from feature matrices
    assert TARGET_COLUMN not in X_train.columns, f"CRITICAL: Target '{TARGET_COLUMN}' leaked into X_train!"
    assert TARGET_COLUMN not in X_test.columns, f"CRITICAL: Target '{TARGET_COLUMN}' leaked into X_test!"

    # Distribution calculations
    train_counts = {int(k): int(v) for k, v in y_train.value_counts().sort_index().items()}
    test_counts = {int(k): int(v) for k, v in y_test.value_counts().sort_index().items()}
    full_counts = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}

    train_pcts = {int(k): float(round(v, 2)) for k, v in (y_train.value_counts(normalize=True) * 100).sort_index().items()}
    test_pcts = {int(k): float(round(v, 2)) for k, v in (y_test.value_counts(normalize=True) * 100).sort_index().items()}
    full_pcts = {int(k): float(round(v, 2)) for k, v in (y.value_counts(normalize=True) * 100).sort_index().items()}

    split_metrics = {
        "dataset_total_rows": len(df),
        "split_ratio": f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
        "random_state": random_state,
        "stratified": stratify,
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.shape),
            "y_test": list(y_test.shape)
        },
        "target_distribution": {
            "full_dataset": {
                "counts": full_counts,
                "percentages": full_pcts
            },
            "train_partition": {
                "counts": train_counts,
                "percentages": train_pcts
            },
            "test_partition": {
                "counts": test_counts,
                "percentages": test_pcts
            }
        },
        "indices_overlap": len(overlap),
        "features": {
            "count": len(X_train.columns),
            "names_in_order": list(X_train.columns),
            "identical_order_in_test": list(X_train.columns) == list(X_test.columns)
        },
        "leakage_verification": {
            "target_in_X_train": TARGET_COLUMN in X_train.columns,
            "target_in_X_test": TARGET_COLUMN in X_test.columns,
            "preprocessor_fitted": False,
            "test_information_used": False
        }
    }

    return X_train, X_test, y_train, y_test, split_metrics


def run_split(dataset_path: str):
    print("=" * 80)
    print("STEP 4: TRAIN/TEST SPLIT - DIABETES MODEL")
    print("=" * 80)

    abs_path = os.path.abspath(dataset_path)
    print(f"Dataset path: {abs_path}")

    X_train, X_test, y_train, y_test, metrics = create_train_test_split(
        dataset_path=abs_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )

    print("\n1. Partition Shapes:")
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - X_test shape:  {X_test.shape}")
    print(f"   - y_train shape: {y_train.shape}")
    print(f"   - y_test shape:  {y_test.shape}")

    print("\n2. Target Class Counts and Proportions:")
    print("   [Full Dataset (N=768)]")
    for k, v in metrics["target_distribution"]["full_dataset"]["counts"].items():
        pct = metrics["target_distribution"]["full_dataset"]["percentages"][k]
        print(f"     * Class {k}: {v} samples ({pct:.2f}%)")

    print(f"   [Train Partition (N={len(y_train)}, 80%)]")
    for k, v in metrics["target_distribution"]["train_partition"]["counts"].items():
        pct = metrics["target_distribution"]["train_partition"]["percentages"][k]
        print(f"     * Class {k}: {v} samples ({pct:.2f}%)")

    print(f"   [Test Partition (N={len(y_test)}, 20%)]")
    for k, v in metrics["target_distribution"]["test_partition"]["counts"].items():
        pct = metrics["target_distribution"]["test_partition"]["percentages"][k]
        print(f"     * Class {k}: {v} samples ({pct:.2f}%)")

    print("\n3. Feature Names and Ordering:")
    print(f"   - Feature count in X_train: {len(X_train.columns)}")
    print(f"   - Feature count in X_test:  {len(X_test.columns)}")
    print(f"   - Identical feature order in train and test: {metrics['features']['identical_order_in_test']}")
    print("   - Feature order:")
    for idx, col in enumerate(X_train.columns):
        print(f"     [{idx}] '{col}'")

    print("\n4. Index Disjointness and Overlap Check:")
    print(f"   - Number of overlapping row indices: {metrics['indices_overlap']}")
    print(f"   - Total rows combined: {len(X_train) + len(X_test)} == {metrics['dataset_total_rows']}")

    print("\n5. Target Classes Presence & Stratification:")
    print(f"   - Both classes (0 and 1) in y_train: {set(y_train.unique()) == {0, 1}}")
    print(f"   - Both classes (0 and 1) in y_test:  {set(y_test.unique()) == {0, 1}}")
    print("   - Stratification delta from original dataset:")
    for k in [0, 1]:
        orig_p = metrics["target_distribution"]["full_dataset"]["percentages"][k]
        tr_p = metrics["target_distribution"]["train_partition"]["percentages"][k]
        te_p = metrics["target_distribution"]["test_partition"]["percentages"][k]
        print(f"     * Class {k}: Original = {orig_p:.2f}%, Train = {tr_p:.2f}% (delta: {tr_p - orig_p:+.2f}%), Test = {te_p:.2f}% (delta: {te_p - orig_p:+.2f}%)")

    print("\n6. Pipeline and Leakage Confirmations:")
    print(f"   - Target 'Outcome' present in X_train: {metrics['leakage_verification']['target_in_X_train']}")
    print(f"   - Target 'Outcome' present in X_test:  {metrics['leakage_verification']['target_in_X_test']}")
    print(f"   - Preprocessor fitted: {metrics['leakage_verification']['preprocessor_fitted']}")
    print(f"   - Test partition information used for preprocessing or training: {metrics['leakage_verification']['test_information_used']}")
    print("   - Datasets saved as permanent files: False (reproducible dynamically via random_state=42)")

    print("=" * 80)
    print("STEP 4 COMPLETED SUCCESSFULLY - NO PREPROCESSING OR MODEL TRAINING PERFORMED")
    print("=" * 80)

    return X_train, X_test, y_train, y_test, metrics


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

    run_split(selected_dataset)
