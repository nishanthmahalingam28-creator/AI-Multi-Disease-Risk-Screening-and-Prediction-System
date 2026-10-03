"""
Unit and Validation Test Suite for Stroke Preprocessing Pipeline
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 5 — Preprocessing Tests
Verifies:
1. Raw dataset remains completely unchanged (hash integrity).
2. Train/test split maintains exact 4088/1022 sample counts.
3. Feature ordering and feature count are preserved.
4. Target 'stroke' and identifier 'id' are strictly absent.
5. Saved preprocessor artifact loads successfully and is fitted.
6. Pipeline contains expected ColumnTransformer structure.
7. Categorical OneHotEncoder has handle_unknown='ignore'.
8. Categorical categories are learned from training data only.
9. Numeric imputer is fitted with valid statistics.
10. BMI training median exists and is exactly 28.0.
11. Transformed train and test partitions contain 0 NaNs.
12. Transformed train and test partitions contain only finite values.
13. Output dimensions are exactly (4088, 21) and (1022, 21).
14. Transforming test partition does not alter fitted parameters.
15. Test data was never used to fit preprocessing parameters.
16. Preprocessing summary JSON exists and contains valid metadata.
17. Unknown category handling succeeds without errors.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
import pytest
from sklearn.compose import ColumnTransformer

# Add local directory to sys.path for standalone and pytest execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from define_features import (
    EXPECTED_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    define_features_and_target,
)
from split_data import create_train_test_split, compute_sha256
from preprocessing import (
    CONTINUOUS_NUMERIC_FEATURES,
    BINARY_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_stroke_preprocessor,
)

EXPECTED_DATASET_HASH = "aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815"


def get_dataset_path() -> str:
    workspace_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    return os.path.join(workspace_root, "datasets", "stroke", "stroke.csv")


def test_original_dataset_hash_unchanged():
    """Verify raw dataset file is not modified during preprocessing operations."""
    path = get_dataset_path()
    current_hash = compute_sha256(path)
    assert current_hash == EXPECTED_DATASET_HASH, (
        f"Raw dataset hash modified!\nExpected: {EXPECTED_DATASET_HASH}\nGot: {current_hash}"
    )


def test_train_test_row_counts():
    """Verify exact 4088 train rows and 1022 test rows."""
    path = get_dataset_path()
    X_train, X_test, y_train, y_test, meta = create_train_test_split(path)

    assert X_train.shape[0] == 4088, f"Expected 4088 train rows, got {X_train.shape[0]}"
    assert X_test.shape[0] == 1022, f"Expected 1022 test rows, got {X_test.shape[0]}"
    assert len(y_train) == 4088, f"Expected 4088 train targets, got {len(y_train)}"
    assert len(y_test) == 1022, f"Expected 1022 test targets, got {len(y_test)}"
    assert X_train.shape[0] + X_test.shape[0] == 5110, "Row sum mismatch!"


def test_feature_order_and_count():
    """Verify exact 10 predictor features in canonical order."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    assert list(X_train.columns) == EXPECTED_PREDICTOR_COLUMNS
    assert list(X_test.columns) == EXPECTED_PREDICTOR_COLUMNS
    assert len(X_train.columns) == 10


def test_target_and_id_absent():
    """Verify target 'stroke' and identifier 'id' are strictly absent from feature matrices."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    assert "id" not in X_train.columns
    assert "id" not in X_test.columns
    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns


def test_preprocessor_artifact_loads_and_is_fitted():
    """Verify saved preprocessor.joblib loads and is fitted."""
    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    assert os.path.exists(artifact_path), f"preprocessor.joblib not found at: {artifact_path}"

    preprocessor = joblib.load(artifact_path)
    assert isinstance(preprocessor, ColumnTransformer)
    assert hasattr(preprocessor, "transformers_"), "Loaded preprocessor is not fitted!"


def test_expected_pipeline_structure():
    """Verify preprocessor has continuous, binary, and categorical named transformers."""
    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "continuous" in transformer_names
    assert "binary" in transformer_names
    assert "categorical" in transformer_names


def test_categorical_encoder_handle_unknown():
    """Verify OneHotEncoder has handle_unknown='ignore'."""
    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    cat_pipeline = preprocessor.named_transformers_["categorical"]
    ohe = cat_pipeline.named_steps["ohe"]
    assert ohe.handle_unknown == "ignore"


def test_categorical_categories_learned_on_train():
    """Verify learned categories match categories present in training partition."""
    path = get_dataset_path()
    X_train, _, _, _, _ = create_train_test_split(path)

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    cat_pipeline = preprocessor.named_transformers_["categorical"]
    ohe = cat_pipeline.named_steps["ohe"]

    for col_idx, col_name in enumerate(CATEGORICAL_FEATURES):
        learned_cats = set(ohe.categories_[col_idx])
        train_cats = set(X_train[col_name].dropna().unique())
        assert learned_cats == train_cats, f"Category mismatch for {col_name}"


def test_numeric_imputer_is_fitted():
    """Verify continuous and binary numeric imputers are fitted with valid statistics."""
    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps["imputer"]
    assert hasattr(cont_imputer, "statistics_")
    assert len(cont_imputer.statistics_) == len(CONTINUOUS_NUMERIC_FEATURES)

    bin_imputer = preprocessor.named_transformers_["binary"].named_steps["imputer"]
    assert hasattr(bin_imputer, "statistics_")
    assert len(bin_imputer.statistics_) == len(BINARY_NUMERIC_FEATURES)


def test_bmi_training_median_exists():
    """Verify BMI median learned by the preprocessor is exactly 28.0 (from training set)."""
    path = get_dataset_path()
    X_train, _, _, _, _ = create_train_test_split(path)
    expected_bmi_median = float(X_train["bmi"].median())

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps["imputer"]
    bmi_idx = CONTINUOUS_NUMERIC_FEATURES.index("bmi")
    actual_bmi_median = float(cont_imputer.statistics_[bmi_idx])

    assert actual_bmi_median == 28.0, f"Expected 28.0, got {actual_bmi_median}"
    assert actual_bmi_median == expected_bmi_median, "Imputer median does not match X_train median!"


def test_transformed_train_and_test_no_nan():
    """Verify transformed partitions contain zero NaN values."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert int(np.isnan(X_train_trans).sum()) == 0, "NaNs found in transformed X_train!"
    assert int(np.isnan(X_test_trans).sum()) == 0, "NaNs found in transformed X_test!"


def test_transformed_train_and_test_only_finite():
    """Verify transformed partitions contain only finite numeric values (no infs)."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert bool(np.all(np.isfinite(X_train_trans))), "Non-finite values in transformed X_train!"
    assert bool(np.all(np.isfinite(X_test_trans))), "Non-finite values in transformed X_test!"


def test_transformed_dimensions():
    """Verify transformed dimensions are exactly (4088, 21) and (1022, 21)."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert X_train_trans.shape == (4088, 21), f"Expected (4088, 21), got {X_train_trans.shape}"
    assert X_test_trans.shape == (1022, 21), f"Expected (1022, 21), got {X_test_trans.shape}"


def test_transformation_does_not_change_fitted_parameters():
    """Verify transforming test data does not alter any learned training statistics."""
    path = get_dataset_path()
    _, X_test, _, _, _ = create_train_test_split(path)

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps["imputer"]
    cont_scaler = preprocessor.named_transformers_["continuous"].named_steps["scaler"]
    ohe = preprocessor.named_transformers_["categorical"].named_steps["ohe"]

    initial_medians = np.copy(cont_imputer.statistics_)
    initial_scale = np.copy(cont_scaler.scale_)
    initial_center = np.copy(cont_scaler.center_)
    initial_categories = [np.copy(c) for c in ohe.categories_]

    # Transform test set
    _ = preprocessor.transform(X_test)

    np.testing.assert_array_equal(cont_imputer.statistics_, initial_medians)
    np.testing.assert_array_equal(cont_scaler.scale_, initial_scale)
    np.testing.assert_array_equal(cont_scaler.center_, initial_center)
    for cat_before, cat_after in zip(initial_categories, ohe.categories_):
        np.testing.assert_array_equal(cat_before, cat_after)


def test_no_fit_on_test_data():
    """Verify preprocessor parameters strictly equal training set statistics, not test set."""
    path = get_dataset_path()
    X_train, X_test, _, _, _ = create_train_test_split(path)

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps["imputer"]
    train_glucose_median = float(X_train["avg_glucose_level"].median())
    test_glucose_median = float(X_test["avg_glucose_level"].median())

    fitted_glucose_median = float(cont_imputer.statistics_[CONTINUOUS_NUMERIC_FEATURES.index("avg_glucose_level")])

    assert fitted_glucose_median == train_glucose_median
    assert fitted_glucose_median != test_glucose_median or abs(train_glucose_median - test_glucose_median) < 1e-4


def test_summary_json_integrity():
    """Verify preprocessing_summary.json exists, is valid JSON, and has all required fields."""
    summary_path = os.path.join(current_dir, "preprocessing_summary.json")
    assert os.path.exists(summary_path), f"Summary JSON not found at: {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_domain"] == "stroke"
    assert meta["original_feature_count"] == 10
    assert meta["training_imputation_statistics"]["bmi_training_median"] == 28.0
    assert meta["scaler_selection"]["selected_scaler"] == "RobustScaler"
    assert meta["partition_shapes"]["transformed_train"] == [4088, 21]
    assert meta["partition_shapes"]["transformed_test"] == [1022, 21]
    assert meta["integrity_and_quarantine"]["preprocessing_fit_partition"] == "X_train_only"
    assert meta["integrity_and_quarantine"]["test_set_used_only_for_transform"] is True


def test_unknown_category_handling():
    """Verify preprocessor gracefully handles unseen categorical levels using handle_unknown='ignore'."""
    path = get_dataset_path()
    X_train, _, _, _, _ = create_train_test_split(path)

    artifact_path = os.path.join(current_dir, "preprocessor.joblib")
    preprocessor = joblib.load(artifact_path)

    sample = X_train.iloc[[0]].copy()
    sample["gender"] = "NonBinary"  # Unseen category
    sample["work_type"] = "Freelancer"  # Unseen category
    sample["smoking_status"] = "Vaping"  # Unseen category

    transformed = preprocessor.transform(sample)
    assert transformed.shape == (1, 21)
    assert not np.isnan(transformed).any()
    assert np.all(np.isfinite(transformed))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
