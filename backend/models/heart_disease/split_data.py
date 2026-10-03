"""
Train/Test Split Creation and Validation Script
Heart Disease Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 4 — Train/Test Split
Performs reproducible 80/20 stratified partitioning of the Heart Disease dataset.
The test set is quarantined and remains completely untouched by any preprocessing,
transformation, scaling, or training.
"""

import os
import sys
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add local directory to sys.path for standalone and module-level execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from define_features import (
        EXPECTED_PREDICTOR_COLUMNS,
        TARGET_COLUMN,
        TARGET_MAPPING,
        load_dataset,
        define_features_and_target,
    )
except ImportError:
    from backend.models.heart_disease.define_features import (
        EXPECTED_PREDICTOR_COLUMNS,
        TARGET_COLUMN,
        TARGET_MAPPING,
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
    Loads dataset, extracts X and y, and performs reproducible 80/20 stratified train/test split.
    
    Guarantees:
    - Zero data leakage: No preprocessing, scaling, imputation, or engineering is applied.
    - Stratification: Preserves the observed target class distribution (Absence vs Presence).
    - Disjoint partitions: Zero index/row overlap between train and test partitions.
    - Feature integrity: Preserves exact 13 predictor columns in their canonical order.
    - Quarantined test partition: Returned untouched for final evaluation only.
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

    # 4. Rigorous validation checks
    # Shape checks
    assert X_train.shape[0] == 216, f"Expected 216 train samples, got {X_train.shape[0]}"
    assert X_test.shape[0] == 54, f"Expected 54 test samples, got {X_test.shape[0]}"
    assert len(y_train) == 216, f"Expected 216 train targets, got {len(y_train)}"
    assert len(y_test) == 54, f"Expected 54 test targets, got {len(y_test)}"
    assert X_train.shape[1] == 13, f"Expected 13 features in X_train, got {X_train.shape[1]}"
    assert X_test.shape[1] == 13, f"Expected 13 features in X_test, got {X_test.shape[1]}"

    # Index disjointness (zero overlap)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    index_overlap = train_indices.intersection(test_indices)
    assert len(index_overlap) == 0, f"Detected {len(index_overlap)} overlapping rows between train and test!"

    # Feature preservation and ordering
    assert list(X_train.columns) == EXPECTED_PREDICTOR_COLUMNS, "X_train feature columns/order mismatch"
    assert list(X_test.columns) == EXPECTED_PREDICTOR_COLUMNS, "X_test feature columns/order mismatch"

    # Target exclusion
    assert TARGET_COLUMN not in X_train.columns, f"Target '{TARGET_COLUMN}' found in X_train!"
    assert TARGET_COLUMN not in X_test.columns, f"Target '{TARGET_COLUMN}' found in X_test!"

    # Target class distributions
    y_train_counts = y_train.value_counts().to_dict()
    y_test_counts = y_test.value_counts().to_dict()
    y_train_props = (y_train.value_counts(normalize=True) * 100).to_dict()
    y_test_props = (y_test.value_counts(normalize=True) * 100).to_dict()

    # Verify stratified balance
    # Absence (0): 120 in train, 30 in test
    # Presence (1): 96 in train, 24 in test
    assert y_train_counts[0] == 120, f"Expected 120 Class 0 in y_train, got {y_train_counts[0]}"
    assert y_train_counts[1] == 96, f"Expected 96 Class 1 in y_train, got {y_train_counts[1]}"
    assert y_test_counts[0] == 30, f"Expected 30 Class 0 in y_test, got {y_test_counts[0]}"
    assert y_test_counts[1] == 24, f"Expected 24 Class 1 in y_test, got {y_test_counts[1]}"

    # No null values
    assert X_train.isnull().sum().sum() == 0, "Null values detected in X_train!"
    assert X_test.isnull().sum().sum() == 0, "Null values detected in X_test!"
    assert y_train.isnull().sum() == 0, "Null values detected in y_train!"
    assert y_test.isnull().sum() == 0, "Null values detected in y_test!"

    split_metadata = {
        "dataset_path": "datasets/heart_disease/heart.csv",
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size_fraction": test_size,
        "random_state": random_state,
        "stratified": stratify,
        "feature_count": len(EXPECTED_PREDICTOR_COLUMNS),
        "features": EXPECTED_PREDICTOR_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "train_class_counts": {str(k): int(v) for k, v in y_train_counts.items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test_counts.items()},
        "train_class_percentages": {str(k): round(float(v), 4) for k, v in y_train_props.items()},
        "test_class_percentages": {str(k): round(float(v), 4) for k, v in y_test_props.items()},
        "zero_index_overlap": True,
        "preprocessing_performed": False,
        "notes": (
            "Train/test partitioning complete with exact stratified preservation of target classes. "
            "The 54 test samples are strictly quarantined and must remain completely untouched "
            "during candidate model development and cross-validation."
        )
    }

    return X_train, X_test, y_train, y_test, split_metadata


def main():
    print("=" * 80)
    print("STEP 4: HEART DISEASE TRAIN/TEST SPLIT EXECUTION AND VERIFICATION")
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

    print(f"\n1. Split Configuration:")
    print(f"   - Dataset Path : {resolved_path}")
    print(f"   - Split Ratio  : 80% Train, 20% Test (test_size = 0.20)")
    print(f"   - Random State : 42")
    print(f"   - Stratify     : By target variable 'Heart Disease' (y)")

    # 2. Execute Train/Test split
    print(f"\n2. Executing train_test_split...")
    X_train, X_test, y_train, y_test, metadata = create_train_test_split(
        dataset_path=resolved_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )

    # 3. Report Partition Shapes
    print(f"\n3. Partition Shapes:")
    print(f"   - Total records in raw dataset : {metadata['total_samples']}")
    print(f"   - X_train shape                : {X_train.shape} (216 rows, 13 features)")
    print(f"   - X_test shape (quarantined)   : {X_test.shape}  (54 rows, 13 features)")
    print(f"   - y_train shape                : {y_train.shape} (216 labels)")
    print(f"   - y_test shape (quarantined)   : {y_test.shape}  (54 labels)")

    # 4. Report Target Class Distributions
    print(f"\n4. Target Class Distribution Comparison:")
    print(f"   - Target Column: '{TARGET_COLUMN}' (Absence=0, Presence=1)")
    print(f"   - Overall Raw Dataset (270 rows):")
    print(f"     * Class 0 (Absence) : 150 rows (55.56%)")
    print(f"     * Class 1 (Presence): 120 rows (44.44%)")
    print(f"   - Training Set (216 rows, 80.0% of total):")
    print(f"     * Class 0 (Absence) : {metadata['train_class_counts']['0']} rows ({metadata['train_class_percentages']['0']}%)")
    print(f"     * Class 1 (Presence): {metadata['train_class_counts']['1']} rows ({metadata['train_class_percentages']['1']}%)")
    print(f"   - Quarantined Test Set (54 rows, 20.0% of total):")
    print(f"     * Class 0 (Absence) : {metadata['test_class_counts']['0']} rows ({metadata['test_class_percentages']['0']}%)")
    print(f"     * Class 1 (Presence): {metadata['test_class_counts']['1']} rows ({metadata['test_class_percentages']['1']}%)")

    # 5. Report Stratification Verification
    print(f"\n5. Stratification Preservation:")
    train_ratio = metadata['train_class_counts']['0'] / metadata['train_class_counts']['1']
    test_ratio = metadata['test_class_counts']['0'] / metadata['test_class_counts']['1']
    print(f"   - Raw class ratio (Absence : Presence)  : 150 / 120 = 1.25 : 1")
    print(f"   - Train class ratio (Absence : Presence): {metadata['train_class_counts']['0']} / {metadata['train_class_counts']['1']} = {train_ratio:.2f} : 1")
    print(f"   - Test class ratio (Absence : Presence) : {metadata['test_class_counts']['0']} / {metadata['test_class_counts']['1']} = {test_ratio:.2f} : 1")
    print(f"   - Stratification Status: PERFECTLY PRESERVED across train and test partitions.")

    # 6. Row Overlap Verification
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    overlap = train_idx.intersection(test_idx)
    print(f"\n6. Partition Disjointness Check:")
    print(f"   - Overlapping row indices between train and test: {len(overlap)}")
    print(f"   - Overlap Status: ZERO OVERLAP (Train and test are completely disjoint).")

    # 7. Feature Preservation and Ordering
    print(f"\n7. Feature Preservation and Canonical Order:")
    print(f"   - Feature count in X_train : {X_train.shape[1]}")
    print(f"   - Feature count in X_test  : {X_test.shape[1]}")
    order_matches = (list(X_train.columns) == EXPECTED_PREDICTOR_COLUMNS == list(X_test.columns))
    print(f"   - Feature ordering match   : {order_matches}")
    for idx, col in enumerate(EXPECTED_PREDICTOR_COLUMNS):
        print(f"     [{idx:2d}] '{col}'")

    # 8. Data Leakage and Preprocessing Status
    print(f"\n8. Preprocessing and Quarantine Status:")
    print(f"   - Preprocessing performed prior to or during split: NONE")
    print(f"   - Feature scaling applied                         : NONE")
    print(f"   - Imputation applied                              : NONE")
    print(f"   - Zero replacement applied                        : NONE")
    print(f"   - Outlier modification applied                    : NONE")
    print(f"   - Feature engineering applied                     : NONE")
    print(f"   - Feature selection applied                       : NONE")
    print(f"   - SMOTE / Resampling applied                      : NONE")
    print(f"   - Model training performed                        : NONE")
    print(f"   - Test partition quarantine confirmation          : ACTIVE (54 samples quarantined)")

    print("\n" + "=" * 80)
    print("STEP 4 TRAIN/TEST SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
