"""
Unit and Validation Tests for Diabetes Preprocessing Pipeline
AI Multi-Disease Risk Screening and Prediction System

Verifies:
1. Original CSV dataset remains unchanged.
2. Split sizes are exactly 614 (train) and 154 (test).
3. Preprocessor is completely unfitted before training data is supplied.
4. Preprocessing can fit X_train successfully.
5. X_train transforms successfully.
6. X_test transforms successfully without refitting.
7. Transformed output dimensions are correct ((614, 8) and (154, 8)).
8. No NaN or Inf remains after transformation in either partition.
9. Feature order is preserved identically in both partitions.
10. Preprocessing does not use target vector y.
11. Test data is not used to fit preprocessing (parameters remain unchanged).
12. Pregnancies=0 is preserved (not treated as missing).
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import (
    FEATURE_COLUMNS,
    ZERO_AS_MISSING_COLUMNS,
    PASSTHROUGH_ZERO_COLUMNS,
    build_preprocessor,
    prepare_data,
    fit_and_transform_train,
    transform_test,
)
from define_features import TARGET_COLUMN


def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_tests():
    print("=" * 80)
    print("STEP 5: PREPROCESSING PIPELINE TEST SUITE - DIABETES MODEL")
    print("=" * 80)

    dataset_path = os.path.join(
        current_dir, "..", "..", "..", "datasets", "diabetes", "diabetes.csv"
    )
    dataset_path = os.path.abspath(dataset_path)

    # 1. Record original dataset state
    orig_hash = compute_file_hash(dataset_path)
    orig_size = os.path.getsize(dataset_path)
    print(f"Dataset path: {dataset_path}")
    print(f"Original file hash (SHA256): {orig_hash[:16]}... | Size: {orig_size} bytes")

    # 2. Test prepare_data split sizes
    print("\n[Test 1 & 2] Testing split sizes and preprocessor initial state...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        dataset_path, test_size=0.20, random_state=42, stratify=True
    )

    # Check split sizes
    assert X_train.shape == (614, 8), f"Expected X_train shape (614, 8), got {X_train.shape}"
    assert X_test.shape == (154, 8), f"Expected X_test shape (154, 8), got {X_test.shape}"
    assert y_train.shape == (614,), f"Expected y_train shape (614,), got {y_train.shape}"
    assert y_test.shape == (154,), f"Expected y_test shape (154,), got {y_test.shape}"
    assert len(X_train) + len(X_test) == 768, "Sum of split sizes != 768"
    print("  -> PASSED: Split sizes are exactly 614 train and 154 test.")

    # 3. Test preprocessor is unfitted before training data is supplied
    print("\n[Test 3] Testing preprocessor is unfitted prior to fitting...")
    imputer_step = preprocessor.named_steps["imputer"]
    scaler_step = preprocessor.named_steps["scaler"]
    assert not hasattr(imputer_step, "statistics_"), "Imputer must not be fitted upon creation!"
    assert not hasattr(scaler_step, "center_"), "Scaler must not be fitted upon creation!"
    print("  -> PASSED: Preprocessor is completely unfitted.")

    # 4. Test preprocessing fits X_train
    print("\n[Test 4 & 5] Testing fit_and_transform_train() on X_train...")
    X_train_transformed, fitted_preprocessor = fit_and_transform_train(preprocessor, X_train)
    assert hasattr(fitted_preprocessor.named_steps["imputer"], "statistics_"), "Imputer must be fitted after fit_transform!"
    assert hasattr(fitted_preprocessor.named_steps["scaler"], "center_"), "Scaler must be fitted after fit_transform!"
    print("  -> PASSED: Preprocessor fitted successfully on X_train.")

    # 5. Verify X_train transformation output dimensions and validity
    print("\n[Test 6 & 7] Testing X_train transformed dimensions and finiteness...")
    assert X_train_transformed.shape == (614, 8), f"Expected (614, 8), got {X_train_transformed.shape}"
    assert not np.isnan(X_train_transformed).any(), "Transformed X_train contains NaN values!"
    assert not np.isinf(X_train_transformed).any(), "Transformed X_train contains Inf values!"
    print("  -> PASSED: X_train transformed successfully to shape (614, 8) with 0 NaNs and 0 Infs.")

    # 6. Test X_test transforms successfully without refitting
    print("\n[Test 8] Testing transform_test() on X_test without refitting...")
    # Cache learned parameters before transforming test
    learned_imputer_stats = fitted_preprocessor.named_steps["imputer"].statistics_.copy()
    learned_scaler_center = fitted_preprocessor.named_steps["scaler"].center_.copy()
    learned_scaler_scale = fitted_preprocessor.named_steps["scaler"].scale_.copy()

    X_test_transformed = transform_test(fitted_preprocessor, X_test)
    assert X_test_transformed.shape == (154, 8), f"Expected (154, 8), got {X_test_transformed.shape}"
    assert not np.isnan(X_test_transformed).any(), "Transformed X_test contains NaN values!"
    assert not np.isinf(X_test_transformed).any(), "Transformed X_test contains Inf values!"
    print("  -> PASSED: X_test transformed successfully to shape (154, 8) with 0 NaNs and 0 Infs.")

    # 7. Test data leakage: Verify parameters did NOT change after test transform
    print("\n[Test 9] Verifying test data was not used to fit preprocessing...")
    post_imputer_stats = fitted_preprocessor.named_steps["imputer"].statistics_
    post_scaler_center = fitted_preprocessor.named_steps["scaler"].center_
    post_scaler_scale = fitted_preprocessor.named_steps["scaler"].scale_
    assert np.array_equal(learned_imputer_stats, post_imputer_stats), "Imputer statistics changed during test transform!"
    assert np.array_equal(learned_scaler_center, post_scaler_center), "Scaler centers changed during test transform!"
    assert np.array_equal(learned_scaler_scale, post_scaler_scale), "Scaler scales changed during test transform!"
    print("  -> PASSED: Preprocessor parameters strictly identical before and after transforming X_test.")

    # 8. Test feature order preservation
    print("\n[Test 10] Testing exact 8-feature order preservation...")
    assert list(X_train.columns) == FEATURE_COLUMNS, "X_train columns do not match canonical feature order!"
    assert list(X_test.columns) == FEATURE_COLUMNS, "X_test columns do not match canonical feature order!"
    assert len(FEATURE_COLUMNS) == 8, f"Expected 8 features, got {len(FEATURE_COLUMNS)}"
    print(f"  -> PASSED: Canonical feature order strictly preserved: {FEATURE_COLUMNS}")

    # 9. Test preprocessing does not use y
    print("\n[Test 11] Testing preprocessing does not use target vector y...")
    # fit_transform and transform do not accept or utilize y
    preproc_new = build_preprocessor()
    res_no_y = preproc_new.fit_transform(X_train)
    assert np.allclose(X_train_transformed, res_no_y), "Preprocessor output differs when y is omitted!"
    print("  -> PASSED: Preprocessing is fully unsupervised and independent of y.")

    # 10. Test Pregnancies=0 is preserved, while physiological zeros are imputed
    print("\n[Test 12] Testing selective zero-value handling...")
    # Prior to scaling, check intermediate imputer output
    zero_step = fitted_preprocessor.named_steps["zero_to_nan"]
    imp_step = fitted_preprocessor.named_steps["imputer"]
    
    # Check on a mock observation with Pregnancies=0 and Glucose=0
    mock_input = pd.DataFrame([{
        "Pregnancies": 0,
        "Glucose": 0,
        "BloodPressure": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "BMI": 0.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 25
    }])
    mock_zeroed = zero_step.transform(mock_input)
    assert mock_zeroed.loc[0, "Pregnancies"] == 0, "Pregnancies=0 must NOT be converted to NaN!"
    assert np.isnan(mock_zeroed.loc[0, "Glucose"]), "Glucose=0 must be converted to NaN!"
    assert np.isnan(mock_zeroed.loc[0, "BloodPressure"]), "BloodPressure=0 must be converted to NaN!"
    assert np.isnan(mock_zeroed.loc[0, "SkinThickness"]), "SkinThickness=0 must be converted to NaN!"
    assert np.isnan(mock_zeroed.loc[0, "Insulin"]), "Insulin=0 must be converted to NaN!"
    assert np.isnan(mock_zeroed.loc[0, "BMI"]), "BMI=0 must be converted to NaN!"

    mock_imputed = imp_step.transform(mock_zeroed)
    # Check that Pregnancies remains 0 after imputer
    assert mock_imputed[0, 0] == 0.0, "Pregnancies must remain 0 after imputation!"
    # Check that Glucose received the training median
    expected_glucose_median = learned_imputer_stats[1]
    assert mock_imputed[0, 1] == expected_glucose_median, f"Glucose should receive training median {expected_glucose_median}, got {mock_imputed[0, 1]}"
    print("  -> PASSED: Pregnancies=0 preserved; physiological zeros imputed with training medians.")

    # 11. Test original dataset non-mutation
    print("\n[Test 13] Verifying original CSV dataset remains unchanged...")
    post_hash = compute_file_hash(dataset_path)
    post_size = os.path.getsize(dataset_path)
    assert orig_hash == post_hash, "Original dataset file hash changed!"
    assert orig_size == post_size, "Original dataset file size changed!"
    print(f"  -> PASSED: Original dataset file hash matches exactly: {post_hash[:16]}...")

    print("\n" + "=" * 80)
    print("ALL 13 PREPROCESSING PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
