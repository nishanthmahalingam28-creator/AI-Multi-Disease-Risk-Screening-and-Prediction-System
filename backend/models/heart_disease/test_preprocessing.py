"""
Unit and Validation Test Suite for Heart Disease Preprocessing Pipeline
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 5 — Preprocessing Tests
Verifies:
1. Raw dataset remains completely unchanged (hash integrity).
2. Train/test split maintains exact 216/54 sample counts.
3. Preprocessor is initially unfitted.
4. Preprocessing fits only on training data (X_train).
5. Transformed training dimensions are correct (216, 13).
6. Transformed test dimensions are correct (54, 13).
7. Transformed data contains no NaN or Inf values.
8. Test transformation does not refit or alter learned parameters.
9. Feature ordering is preserved and tracked.
10. Target is strictly excluded from preprocessing.
11. Binary and discrete features are passed through without modification.
12. No outlier removal or row filtering occurs.
13. No zero replacement occurs in valid binary/discrete features.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

# Add local directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from define_features import (
    EXPECTED_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    define_features_and_target,
)
from split_data import create_train_test_split
from preprocessing import (
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
    DISCRETE_FEATURES,
    FEATURE_COLUMNS,
    build_approach_b_pipeline,
    build_approach_a_pipeline,
    build_preprocessing_pipeline,
)


def get_dataset_path() -> str:
    workspace_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    return os.path.join(workspace_root, "datasets", "heart_disease", "heart.csv")


def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# PYTEST TEST FUNCTIONS
# =============================================================================

def test_raw_dataset_remains_unchanged():
    """Verify raw dataset file is not modified during preprocessing operations."""
    path = get_dataset_path()
    initial_hash = compute_file_sha256(path)

    # Execute full split and preprocessing
    X_train, X_test, y_train, y_test, _ = create_train_test_split(path)
    pipe = build_preprocessing_pipeline("approach_b")
    _ = pipe.fit_transform(X_train)
    _ = pipe.transform(X_test)

    final_hash = compute_file_sha256(path)
    assert initial_hash == final_hash, "Raw dataset hash changed during preprocessing!"


def test_split_dimensions():
    """Verify exact 216/54 row split and 13 feature counts."""
    path = get_dataset_path()
    X_train, X_test, y_train, y_test, meta = create_train_test_split(path)

    assert X_train.shape == (216, 13), f"Expected (216, 13), got {X_train.shape}"
    assert X_test.shape == (54, 13), f"Expected (54, 13), got {X_test.shape}"
    assert len(y_train) == 216, f"Expected 216 targets, got {len(y_train)}"
    assert len(y_test) == 54, f"Expected 54 targets, got {len(y_test)}"


def test_preprocessor_initially_unfitted():
    """Verify preprocessor raises error if transform is called before fit."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    # ColumnTransformer raises NotFittedError or AttributeError when unfitted
    with pytest.raises((NotFittedError, AttributeError)):
        pipe.transform(X_train)


def test_preprocessor_fits_only_on_training_data():
    """Verify preprocessing pipeline fits exclusively on X_train without test set access."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    pipe.fit(X_train)

    # Check learned centers and scales in continuous transformer
    scaler = pipe.named_transformers_["continuous"].named_steps["scaler"]
    imputer = pipe.named_transformers_["continuous"].named_steps["imputer"]

    assert hasattr(scaler, "center_"), "Scaler does not have learned center_ attribute"
    assert hasattr(scaler, "scale_"), "Scaler does not have learned scale_ attribute"
    assert len(scaler.center_) == len(CONTINUOUS_FEATURES)

    # Medians computed on X_train only
    for idx, col in enumerate(CONTINUOUS_FEATURES):
        expected_median = X_train[col].median()
        assert np.isclose(scaler.center_[idx], expected_median), (
            f"Learned center for {col} ({scaler.center_[idx]}) does not match X_train median ({expected_median})"
        )


def test_transformed_training_dimensions():
    """Verify transformed training data has exact dimensions (216, 13)."""
    path = get_dataset_path()
    X_train, _, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    X_train_trans = pipe.fit_transform(X_train)

    assert X_train_trans.shape == (216, 13), f"Expected (216, 13), got {X_train_trans.shape}"


def test_transformed_test_dimensions():
    """Verify transformed test data has exact dimensions (54, 13)."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    pipe.fit(X_train)
    X_test_trans = pipe.transform(X_test)

    assert X_test_trans.shape == (54, 13), f"Expected (54, 13), got {X_test_trans.shape}"


def test_transformed_data_no_nan_or_inf():
    """Verify transformed data contains 0 NaN and 0 Inf values."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    X_tr = pipe.fit_transform(X_train)
    X_te = pipe.transform(X_test)

    assert not np.isnan(X_tr).any(), "NaN found in transformed X_train"
    assert not np.isnan(X_te).any(), "NaN found in transformed X_test"
    assert not np.isinf(X_tr).any(), "Inf found in transformed X_train"
    assert not np.isinf(X_te).any(), "Inf found in transformed X_test"


def test_test_transformation_does_not_refit():
    """Verify transforming X_test does NOT modify learned training parameters (zero leakage)."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    pipe.fit(X_train)

    scaler = pipe.named_transformers_["continuous"].named_steps["scaler"]
    center_before = np.copy(scaler.center_)
    scale_before = np.copy(scaler.scale_)

    # Transform X_test multiple times
    _ = pipe.transform(X_test)
    _ = pipe.transform(X_test)

    center_after = scaler.center_
    scale_after = scaler.scale_

    np.testing.assert_array_equal(center_before, center_after, "Learned centers changed after transforming test data!")
    np.testing.assert_array_equal(scale_before, scale_after, "Learned scales changed after transforming test data!")


def test_feature_order_and_tracking():
    """Verify feature names and tracking through the pipeline."""
    pipe = build_preprocessing_pipeline("approach_b")
    path = get_dataset_path()
    X_train, _, _, _, _ = create_train_test_split(path)
    pipe.fit(X_train)

    feature_names_out = list(pipe.get_feature_names_out())
    assert len(feature_names_out) == 13, f"Expected 13 features out, got {len(feature_names_out)}"
    # All 13 expected features must be present
    for col in EXPECTED_PREDICTOR_COLUMNS:
        assert any(col in name for name in feature_names_out), f"Feature {col} not found in features out: {feature_names_out}"


def test_target_never_included_in_preprocessing():
    """Verify target column 'Heart Disease' is never present in preprocessor inputs or outputs."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns

    pipe = build_preprocessing_pipeline("approach_b")
    pipe.fit(X_train)
    out_names = pipe.get_feature_names_out()
    assert not any(TARGET_COLUMN in name for name in out_names)


def test_binary_and_discrete_features_preserved():
    """Verify binary indicators and discrete features are passed through without scaling distortion."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    X_train_trans = pipe.fit_transform(X_train)

    # In Approach B ColumnTransformer:
    # First 5 columns are continuous, next 3 are binary, next 5 are discrete
    feature_names = list(pipe.get_feature_names_out())

    # Check binary columns
    for col in BINARY_FEATURES:
        col_idx = feature_names.index(col)
        original_vals = X_train[col].values
        transformed_vals = X_train_trans[:, col_idx]
        np.testing.assert_array_equal(original_vals, transformed_vals, f"Binary column {col} was modified!")

    # Check discrete columns
    for col in DISCRETE_FEATURES:
        col_idx = feature_names.index(col)
        original_vals = X_train[col].values
        transformed_vals = X_train_trans[:, col_idx]
        np.testing.assert_array_equal(original_vals, transformed_vals, f"Discrete column {col} was modified!")


def test_no_outlier_removal():
    """Verify no samples were dropped or removed due to outlier flags."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    X_tr = pipe.fit_transform(X_train)
    X_te = pipe.transform(X_test)

    assert X_tr.shape[0] == 216, f"Rows dropped from X_train! Expected 216, got {X_tr.shape[0]}"
    assert X_te.shape[0] == 54, f"Rows dropped from X_test! Expected 54, got {X_te.shape[0]}"


def test_no_zero_replacement():
    """Verify zero counts in binary and discrete features remain strictly identical."""
    path = get_dataset_path()
    X_train, _, _, _, _ = create_train_test_split(path)

    pipe = build_preprocessing_pipeline("approach_b")
    X_train_trans = pipe.fit_transform(X_train)
    feature_names = list(pipe.get_feature_names_out())

    for col in BINARY_FEATURES + DISCRETE_FEATURES:
        col_idx = feature_names.index(col)
        orig_zeros = int((X_train[col] == 0).sum())
        trans_zeros = int((X_train_trans[:, col_idx] == 0).sum())
        assert orig_zeros == trans_zeros, f"Zero count changed for {col}: {orig_zeros} -> {trans_zeros}"


# =============================================================================
# STANDALONE EXECUTION RUNNER
# =============================================================================

def run_all_tests():
    print("=" * 80)
    print("STEP 5: RUNNING PREPROCESSING TEST SUITE (HEART DISEASE)")
    print("=" * 80)

    tests = [
        ("Raw dataset remains unchanged", test_raw_dataset_remains_unchanged),
        ("Split dimensions are exactly 216/54", test_split_dimensions),
        ("Preprocessor initially unfitted", test_preprocessor_initially_unfitted),
        ("Preprocessor fits only on X_train", test_preprocessor_fits_only_on_training_data),
        ("Transformed training dimensions are (216, 13)", test_transformed_training_dimensions),
        ("Transformed test dimensions are (54, 13)", test_transformed_test_dimensions),
        ("Transformed data contains no NaN/Inf", test_transformed_data_no_nan_or_inf),
        ("Test transformation does not refit", test_test_transformation_does_not_refit),
        ("Feature order and tracking preserved", test_feature_order_and_tracking),
        ("Target strictly excluded from preprocessing", test_target_never_included_in_preprocessing),
        ("Binary and discrete features preserved untouched", test_binary_and_discrete_features_preserved),
        ("No outlier removal occurs", test_no_outlier_removal),
        ("No zero replacement occurs", test_no_zero_replacement),
    ]

    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print(f"ALL {len(tests)} PREPROCESSING TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
