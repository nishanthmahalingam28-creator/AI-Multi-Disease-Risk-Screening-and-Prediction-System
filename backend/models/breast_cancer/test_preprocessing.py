"""Validation script for Breast Cancer Preprocessing Pipeline.
Member 1: AI Multi-Disease Risk Screening and Prediction System

Verifies:
1. Feature extraction and exclusion isolation.
2. Preprocessor construction and ColumnTransformer definition.
3. Fitting strictly on X_train (no leakage from X_test).
4. Successful transformation of X_train and X_test.
5. Dimension matching and zero NaN introduction.
6. Non-mutation of original dataset files.
7. Explicit confirmation that no classifier is trained.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXCLUDED_COLUMNS,
    build_preprocessor,
    load_dataset,
    prepare_data,
    fit_and_transform_train,
    transform_test,
)


def run_preprocessing_tests():
    print("=" * 80)
    print("STEP 5: PREPROCESSING PIPELINE VALIDATION AUDIT")
    print("=" * 80)

    dataset_path = os.path.join(
        current_dir, "..", "..", "..", "datasets", "breast_cancer", "breast.csv"
    )
    dataset_path = os.path.abspath(dataset_path)

    # Record original file size and hash/timestamp to verify non-modification
    orig_size = os.path.getsize(dataset_path)
    orig_mtime = os.path.getmtime(dataset_path)
    print(f"Dataset path: {dataset_path}")
    print(f"Original file size: {orig_size} bytes")

    # 1. Test load_dataset
    print("\n[Test 1] Testing load_dataset()...")
    X, y, meta = load_dataset(dataset_path)
    assert X.shape == (569, 30), f"Expected X shape (569, 30), got {X.shape}"
    assert y.shape == (569,), f"Expected y shape (569,), got {y.shape}"
    assert list(X.columns) == FEATURE_COLUMNS, "Feature columns do not match expected schema!"
    assert TARGET_COLUMN not in X.columns, f"Target '{TARGET_COLUMN}' found in X!"
    for excl in EXCLUDED_COLUMNS:
        assert excl not in X.columns, f"Excluded column '{excl}' found in X!"
    print(f"  -> PASSED: Loaded {X.shape[0]} rows, {X.shape[1]} predictor features.")

    # 2. Test prepare_data (Split + Unfitted Preprocessor)
    print("\n[Test 2] Testing prepare_data() split and preprocessor isolation...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        dataset_path, test_size=0.20, random_state=42, stratify=True
    )
    assert X_train.shape == (455, 30), f"Expected X_train (455, 30), got {X_train.shape}"
    assert X_test.shape == (114, 30), f"Expected X_test (114, 30), got {X_test.shape}"
    assert y_train.shape == (455,), f"Expected y_train (455,), got {y_train.shape}"
    assert y_test.shape == (114,), f"Expected y_test (114,), got {y_test.shape}"
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, "Index overlap detected!"
    # Ensure preprocessor is NOT yet fitted
    assert not hasattr(preprocessor, "transformers_"), "Preprocessor must be unfitted upon creation!"
    print(f"  -> PASSED: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}. Preprocessor is unfitted.")

    # 3. Test fitting preprocessor strictly on X_train
    print("\n[Test 3] Testing fit_and_transform_train() on X_train only...")
    X_train_scaled, fitted_preprocessor = fit_and_transform_train(preprocessor, X_train)
    assert hasattr(fitted_preprocessor, "transformers_"), "Preprocessor must be fitted after fit_and_transform_train!"
    assert X_train_scaled.shape == (455, 30), f"Expected scaled X_train (455, 30), got {X_train_scaled.shape}"
    assert not np.isnan(X_train_scaled).any(), "NaN values found in transformed X_train!"
    
    # Verify training scaling properties: mean ~ 0, std ~ 1
    train_means = np.mean(X_train_scaled, axis=0)
    train_stds = np.std(X_train_scaled, axis=0)
    assert np.allclose(train_means, 0.0, atol=1e-7), "Scaled X_train means are not ~ 0.0!"
    assert np.allclose(train_stds, 1.0, atol=1e-7), "Scaled X_train stds are not ~ 1.0!"
    print("  -> PASSED: X_train transformed successfully. Empirical means ~ 0.0, stds ~ 1.0.")

    # 4. Test transform_test() on X_test (without re-fitting)
    print("\n[Test 4] Testing transform_test() on X_test without fitting...")
    # Record scaler internal parameters before test transform to ensure they do not change
    scaler_step = fitted_preprocessor.named_transformers_["standard_scaler"]
    train_fitted_mean = scaler_step.mean_.copy()
    train_fitted_var = scaler_step.var_.copy()

    X_test_scaled = transform_test(fitted_preprocessor, X_test)
    assert X_test_scaled.shape == (114, 30), f"Expected scaled X_test (114, 30), got {X_test_scaled.shape}"
    assert not np.isnan(X_test_scaled).any(), "NaN values found in transformed X_test!"

    # Verify that the scaler parameters did NOT mutate during test transform
    assert np.array_equal(scaler_step.mean_, train_fitted_mean), "Scaler mean mutated during test transform! Data leakage detected."
    assert np.array_equal(scaler_step.var_, train_fitted_var), "Scaler variance mutated during test transform! Data leakage detected."
    print("  -> PASSED: X_test transformed successfully using X_train parameters. No leakage detected.")

    # 5. Dimension and Type Match Confirmation
    print("\n[Test 5] Verifying dimension alignment and data types...")
    assert X_train_scaled.dtype == np.float64, f"Expected float64 dtype, got {X_train_scaled.dtype}"
    assert X_test_scaled.dtype == np.float64, f"Expected float64 dtype, got {X_test_scaled.dtype}"
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1] == 30, "Feature dimension mismatch between train and test!"
    print("  -> PASSED: Both matrices are float64 with exactly 30 columns.")

    # 6. Verify original CSV file was NOT modified
    print("\n[Test 6] Verifying original CSV file integrity...")
    current_size = os.path.getsize(dataset_path)
    current_mtime = os.path.getmtime(dataset_path)
    assert current_size == orig_size, "Original dataset file size was altered!"
    assert current_mtime == orig_mtime, "Original dataset file modification timestamp changed!"
    print(f"  -> PASSED: Original dataset file is completely untouched ({current_size} bytes).")

    print("\n" + "=" * 80)
    print("ALL PREPROCESSING TESTS PASSED SUCCESSFULLY (6/6).")
    print("CONFIRMATION: NO CLASSIFIER / MODEL WAS TRAINED.")
    print("=" * 80)


if __name__ == "__main__":
    run_preprocessing_tests()
