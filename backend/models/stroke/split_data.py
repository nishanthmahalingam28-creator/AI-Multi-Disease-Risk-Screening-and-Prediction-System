"""
Train/Test Split Creation and Validation Script
Stroke Prediction Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 4 — Train/Test Split
Performs reproducible 80/20 stratified partitioning of the Stroke dataset.
The test set is quarantined and remains completely untouched by any preprocessing,
transformation, scaling, imputation, or training.
"""

import os
import sys
import json
import hashlib
from typing import Tuple, Dict, Any, List
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
        load_dataset,
        define_features_and_target,
    )
except ImportError:
    from backend.models.stroke.define_features import (
        EXPECTED_PREDICTOR_COLUMNS,
        TARGET_COLUMN,
        load_dataset,
        define_features_and_target,
    )

EXPECTED_DATASET_HASH = "aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815"
EXPECTED_TOTAL_ROWS = 5110
EXPECTED_TRAIN_ROWS = 4088
EXPECTED_TEST_ROWS = 1022
EXPECTED_FEATURE_COUNT = 10


def compute_sha256(filepath: str) -> str:
    """
    Computes the SHA-256 hash of a file for integrity verification.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


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
    - Stratification: Preserves the observed target class distribution (0 vs 1).
    - Disjoint partitions: Zero index/row overlap between train and test partitions.
    - Feature integrity: Preserves exact 10 predictor columns in their canonical order.
    - Quarantined test partition: Returned untouched for final evaluation only.
    """
    # 1. Dataset integrity check before split
    initial_hash = compute_sha256(dataset_path)
    assert initial_hash == EXPECTED_DATASET_HASH, (
        f"Dataset hash mismatch!\nExpected: {EXPECTED_DATASET_HASH}\nGot: {initial_hash}"
    )

    # 2. Load dataset without modification
    df = load_dataset(dataset_path)
    total_rows = len(df)
    assert total_rows == EXPECTED_TOTAL_ROWS, f"Expected {EXPECTED_TOTAL_ROWS} rows, got {total_rows}"

    # 3. Extract validated X and y
    X, y = define_features_and_target(df)

    # 4. Perform stratified train/test split
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    # 5. Rigorous validation checks
    # Row and feature counts
    assert X_train.shape[0] == EXPECTED_TRAIN_ROWS, f"Expected {EXPECTED_TRAIN_ROWS} train rows, got {X_train.shape[0]}"
    assert X_test.shape[0] == EXPECTED_TEST_ROWS, f"Expected {EXPECTED_TEST_ROWS} test rows, got {X_test.shape[0]}"
    assert len(y_train) == EXPECTED_TRAIN_ROWS, f"Expected {EXPECTED_TRAIN_ROWS} train labels, got {len(y_train)}"
    assert len(y_test) == EXPECTED_TEST_ROWS, f"Expected {EXPECTED_TEST_ROWS} test labels, got {len(y_test)}"
    assert X_train.shape[1] == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} features in X_train, got {X_train.shape[1]}"
    assert X_test.shape[1] == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} features in X_test, got {X_test.shape[1]}"
    assert total_rows == len(X_train) + len(X_test), "Row sum mismatch between original, train, and test!"
    assert X_train.shape[0] == len(y_train), "Row count mismatch between X_train and y_train!"
    assert X_test.shape[0] == len(y_test), "Row count mismatch between X_test and y_test!"

    # Index disjointness (zero overlap) and complete coverage
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    index_overlap = train_indices.intersection(test_indices)
    assert len(index_overlap) == 0, f"Detected {len(index_overlap)} overlapping rows between train and test!"
    assert train_indices.union(test_indices) == set(df.index), "Train and test indices do not fully cover original dataset!"

    # Feature preservation and ordering
    assert list(X_train.columns) == EXPECTED_PREDICTOR_COLUMNS, "X_train feature columns/order mismatch"
    assert list(X_test.columns) == EXPECTED_PREDICTOR_COLUMNS, "X_test feature columns/order mismatch"

    # Excluded columns check
    assert "id" not in X_train.columns and "id" not in X_test.columns, "Identifier 'id' found in feature matrix!"
    assert TARGET_COLUMN not in X_train.columns and TARGET_COLUMN not in X_test.columns, f"Target '{TARGET_COLUMN}' found in feature matrix!"

    # Target class values
    assert set(y_train.unique()) == {0, 1}, f"Unexpected target values in y_train: {set(y_train.unique())}"
    assert set(y_test.unique()) == {0, 1}, f"Unexpected target values in y_test: {set(y_test.unique())}"
    assert y_train.isnull().sum() == 0, "Null values detected in y_train!"
    assert y_test.isnull().sum() == 0, "Null values detected in y_test!"

    # Class counts and proportions
    orig_y_counts = y.value_counts(dropna=False).to_dict()
    orig_y_props = (y.value_counts(dropna=False, normalize=True) * 100).to_dict()
    y_train_counts = y_train.value_counts().to_dict()
    y_test_counts = y_test.value_counts().to_dict()
    y_train_props = (y_train.value_counts(normalize=True) * 100).to_dict()
    y_test_props = (y_test.value_counts(normalize=True) * 100).to_dict()

    # Exact stratified counts verification
    # Overall: 4861 Class 0, 249 Class 1
    # Train: 3889 Class 0, 199 Class 1
    # Test: 972 Class 0, 50 Class 1
    assert y_train_counts[0] == 3889, f"Expected 3889 Class 0 in y_train, got {y_train_counts[0]}"
    assert y_train_counts[1] == 199, f"Expected 199 Class 1 in y_train, got {y_train_counts[1]}"
    assert y_test_counts[0] == 972, f"Expected 972 Class 0 in y_test, got {y_test_counts[0]}"
    assert y_test_counts[1] == 50, f"Expected 50 Class 1 in y_test, got {y_test_counts[1]}"
    assert y_train_counts[0] + y_test_counts[0] == orig_y_counts[0], "Class 0 count sum mismatch!"
    assert y_train_counts[1] + y_test_counts[1] == orig_y_counts[1], "Class 1 count sum mismatch!"

    # BMI missing values verification (unimputed in Step 4)
    train_bmi_nulls = int(X_train["bmi"].isnull().sum())
    test_bmi_nulls = int(X_test["bmi"].isnull().sum())
    total_bmi_nulls = int(X["bmi"].isnull().sum())
    assert train_bmi_nulls == 170, f"Expected 170 missing bmi in train, got {train_bmi_nulls}"
    assert test_bmi_nulls == 31, f"Expected 31 missing bmi in test, got {test_bmi_nulls}"
    assert train_bmi_nulls + test_bmi_nulls == total_bmi_nulls == 201, "Missing BMI count sum mismatch!"

    # Final dataset integrity check
    final_hash = compute_sha256(dataset_path)
    assert initial_hash == final_hash, "Dataset hash changed during execution!"

    split_metadata = {
        "dataset_path": "datasets/stroke/stroke.csv",
        "dataset_sha256": final_hash,
        "original_row_count": total_rows,
        "train_row_count": len(X_train),
        "test_row_count": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "stratified": stratify,
        "feature_count": len(EXPECTED_PREDICTOR_COLUMNS),
        "exact_feature_names_in_order": EXPECTED_PREDICTOR_COLUMNS,
        "target_name": TARGET_COLUMN,
        "target_observed_values": [0, 1],
        "original_class_counts": {str(k): int(v) for k, v in orig_y_counts.items()},
        "train_class_counts": {str(k): int(v) for k, v in y_train_counts.items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test_counts.items()},
        "original_class_percentages": {str(k): round(float(v), 4) for k, v in orig_y_props.items()},
        "train_class_percentages": {str(k): round(float(v), 4) for k, v in y_train_props.items()},
        "test_class_percentages": {str(k): round(float(v), 4) for k, v in y_test_props.items()},
        "missing_bmi_counts": {
            "total": total_bmi_nulls,
            "train": train_bmi_nulls,
            "test": test_bmi_nulls,
            "train_percentage": round(float((train_bmi_nulls / len(X_train)) * 100), 4),
            "test_percentage": round(float((test_bmi_nulls / len(X_test)) * 100), 4)
        },
        "overlap_check": {
            "index_overlap_count": len(index_overlap),
            "is_disjoint": bool(len(index_overlap) == 0),
            "full_coverage": bool(train_indices.union(test_indices) == set(df.index))
        },
        "preprocessing_applied": False,
        "model_training_applied": False,
        "test_set_quarantine": {
            "status": "QUARANTINED",
            "samples": len(X_test),
            "rules": "Test partition must remain untouched until Step 7 final model evaluation. No fitting, imputation tuning, or threshold selection allowed on test set."
        }
    }

    return X_train, X_test, y_train, y_test, split_metadata


def main():
    print("=" * 80)
    print("STEP 4: STROKE DISEASE TRAIN/TEST SPLIT EXECUTION AND VERIFICATION")
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

    # Pre-split dataset integrity check
    pre_sha256 = compute_sha256(resolved_path)
    print(f"\n1. Split Configuration:")
    print(f"   - Dataset Path       : {resolved_path}")
    print(f"   - Dataset SHA-256    : {pre_sha256}")
    print(f"   - Split Ratio        : 80% Train, 20% Test (test_size = 0.20)")
    print(f"   - Random State       : 42")
    print(f"   - Stratify           : By target variable '{TARGET_COLUMN}' (y)")

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
    print(f"   - Total records in raw dataset : {metadata['original_row_count']}")
    print(f"   - X_train shape                : {X_train.shape} (4088 rows, 10 features)")
    print(f"   - X_test shape (quarantined)   : {X_test.shape}  (1022 rows, 10 features)")
    print(f"   - y_train shape                : {y_train.shape} (4088 labels)")
    print(f"   - y_test shape (quarantined)   : {y_test.shape}  (1022 labels)")

    # 4. Report Target Class Distributions
    print(f"\n4. Target Class Distribution Comparison:")
    print(f"   - Target Column: '{TARGET_COLUMN}' (0 = No Stroke, 1 = Stroke)")
    print(f"   - Overall Raw Dataset ({metadata['original_row_count']} rows):")
    print(f"     * Class 0 (No Stroke): {metadata['original_class_counts']['0']} rows ({metadata['original_class_percentages']['0']}%)")
    print(f"     * Class 1 (Stroke)   : {metadata['original_class_counts']['1']} rows ({metadata['original_class_percentages']['1']}%)")
    print(f"   - Training Set ({metadata['train_row_count']} rows, 80.0% of total):")
    print(f"     * Class 0 (No Stroke): {metadata['train_class_counts']['0']} rows ({metadata['train_class_percentages']['0']}%)")
    print(f"     * Class 1 (Stroke)   : {metadata['train_class_counts']['1']} rows ({metadata['train_class_percentages']['1']}%)")
    print(f"   - Quarantined Test Set ({metadata['test_row_count']} rows, 20.0% of total):")
    print(f"     * Class 0 (No Stroke): {metadata['test_class_counts']['0']} rows ({metadata['test_class_percentages']['0']}%)")
    print(f"     * Class 1 (Stroke)   : {metadata['test_class_counts']['1']} rows ({metadata['test_class_percentages']['1']}%)")

    # 5. Report Stratification Verification
    orig_ratio = metadata['original_class_counts']['0'] / metadata['original_class_counts']['1']
    train_ratio = metadata['train_class_counts']['0'] / metadata['train_class_counts']['1']
    test_ratio = metadata['test_class_counts']['0'] / metadata['test_class_counts']['1']
    print(f"\n5. Stratification Proportions:")
    print(f"   - Raw class ratio   (0 : 1): {metadata['original_class_counts']['0']} / {metadata['original_class_counts']['1']} = {orig_ratio:.2f} : 1")
    print(f"   - Train class ratio (0 : 1): {metadata['train_class_counts']['0']} / {metadata['train_class_counts']['1']} = {train_ratio:.2f} : 1")
    print(f"   - Test class ratio  (0 : 1): {metadata['test_class_counts']['0']} / {metadata['test_class_counts']['1']} = {test_ratio:.2f} : 1")
    print(f"   - Class 1 proportions : Original = {metadata['original_class_percentages']['1']}%, Train = {metadata['train_class_percentages']['1']}%, Test = {metadata['test_class_percentages']['1']}%")

    # 6. Report Missing BMI Distribution in Partitions
    print(f"\n6. Missing 'bmi' Counts (Unimputed in Step 4):")
    print(f"   - Total missing bmi in raw dataset : {metadata['missing_bmi_counts']['total']} (3.93%)")
    print(f"   - Missing bmi in X_train           : {metadata['missing_bmi_counts']['train']} ({metadata['missing_bmi_counts']['train_percentage']}%)")
    print(f"   - Missing bmi in X_test            : {metadata['missing_bmi_counts']['test']} ({metadata['missing_bmi_counts']['test_percentage']}%)")
    print(f"   - Sum verification                 : {metadata['missing_bmi_counts']['train']} + {metadata['missing_bmi_counts']['test']} = {metadata['missing_bmi_counts']['total']}")

    # 7. Row Overlap Verification
    print(f"\n7. Partition Disjointness Check:")
    print(f"   - Overlapping row indices between train and test: {metadata['overlap_check']['index_overlap_count']}")
    print(f"   - Partitions are strictly disjoint             : {metadata['overlap_check']['is_disjoint']}")
    print(f"   - Combined partitions cover all original rows   : {metadata['overlap_check']['full_coverage']}")

    # 8. Feature Preservation and Ordering
    print(f"\n8. Feature Preservation and Canonical Order:")
    print(f"   - Feature count in X_train : {X_train.shape[1]}")
    print(f"   - Feature count in X_test  : {X_test.shape[1]}")
    order_matches = (list(X_train.columns) == EXPECTED_PREDICTOR_COLUMNS == list(X_test.columns))
    print(f"   - Feature ordering match   : {order_matches}")
    for idx, col in enumerate(EXPECTED_PREDICTOR_COLUMNS):
        print(f"     [{idx:2d}] '{col}'")

    # 9. Save Summary JSON
    summary_path = os.path.join(script_dir, "split_summary.json")
    print(f"\n9. Exporting split summary JSON to: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("   [x] split_summary.json successfully created.")

    # 10. Data Leakage and Preprocessing Status
    print(f"\n10. Preprocessing and Quarantine Status:")
    print(f"   - Preprocessing performed prior to or during split: NONE (preprocessing_applied = False)")
    print(f"   - Feature scaling applied                         : NONE")
    print(f"   - Imputation applied                              : NONE")
    print(f"   - Outlier modification applied                    : NONE")
    print(f"   - Categorical encoding applied                    : NONE")
    print(f"   - Feature engineering applied                     : NONE")
    print(f"   - Model training performed                        : NONE (model_training_applied = False)")
    print(f"   - Test partition quarantine confirmation          : ACTIVE (1022 samples quarantined)")

    # 11. Post-Execution Dataset Integrity Check
    post_sha256 = compute_sha256(resolved_path)
    hash_match = (pre_sha256 == post_sha256 == EXPECTED_DATASET_HASH)
    print(f"\n11. Dataset Integrity Verification:")
    print(f"   - Pre-Execution Dataset SHA-256 : {pre_sha256}")
    print(f"   - Post-Execution Dataset SHA-256: {post_sha256}")
    print(f"   - Expected SHA-256              : {EXPECTED_DATASET_HASH}")
    print(f"   - Hash Match Verified           : {hash_match}")
    if not hash_match:
        raise RuntimeError("CRITICAL ERROR: Dataset file was modified during Step 4 execution!")

    print("\n" + "=" * 80)
    print("STEP 4 TRAIN/TEST SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
