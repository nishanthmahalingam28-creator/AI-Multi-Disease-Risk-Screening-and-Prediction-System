"""Breast Cancer Dataset Preprocessing Module.
Member 1: AI Multi-Disease Risk Screening and Prediction System

Handles data loading, feature validation, target isolation, and
scikit-learn ColumnTransformer preprocessor pipeline definition.
"""

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Target variable definition
TARGET_COLUMN: str = "diagnosis"
TARGET_ENCODING: Dict[str, int] = {
    "B": 0,  # Benign (Screening Negative)
    "M": 1   # Malignant (Screening Positive)
}

# Explicitly excluded columns (quarantined to prevent data leakage/memorization)
EXCLUDED_COLUMNS: List[str] = [
    "id",           # Non-clinical arbitrary database key
    "Unnamed: 32"   # CSV parser artifact from trailing comma, 100% NaN
]

# Logical grouping of the 30 continuous cytological predictor features
MEAN_FEATURES: List[str] = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]

SE_FEATURES: List[str] = [
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
]

WORST_FEATURES: List[str] = [
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

# Canonical ordered list of the 30 predictor features
FEATURE_COLUMNS: List[str] = MEAN_FEATURES + SE_FEATURES + WORST_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Builds an unfitted scikit-learn ColumnTransformer for numerical standardization.

    Applies StandardScaler to all 30 continuous morphological features.
    
    Returns:
        Unfitted ColumnTransformer instance ready to be fitted exclusively on X_train.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "standard_scaler",
                StandardScaler(),
                FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Loads and strictly validates the breast cancer dataset.

    Args:
        csv_path: Path to breast.csv.

    Returns:
        X (pd.DataFrame): 30-feature predictor matrix.
        y (pd.Series): Binary target series (0 = Benign, 1 = Malignant).
        metadata (dict): Ingestion summary statistics and validation checks.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing, target values are invalid,
                    or unexpected nulls are found in predictors.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    raw_df = pd.read_csv(csv_path)

    # 1. Row count validation
    if raw_df.shape[0] != 569:
        raise ValueError(f"Expected 569 rows in breast cancer dataset, found {raw_df.shape[0]}.")

    # 2. Target presence and value validation
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' missing from dataset.")

    unique_targets = set(raw_df[TARGET_COLUMN].dropna().unique())
    if unique_targets != {"B", "M"}:
        raise ValueError(f"Target column '{TARGET_COLUMN}' must contain exactly {{'B', 'M'}}, found {unique_targets}.")

    # 3. Quarantined column validation
    for excl in EXCLUDED_COLUMNS:
        if excl in raw_df.columns:
            # Verified present in raw CSV; will be excluded from X
            pass

    # 4. Predictor feature existence and completeness validation
    for col in FEATURE_COLUMNS:
        if col not in raw_df.columns:
            raise ValueError(f"Required predictor feature '{col}' missing from dataset.")
        null_count = raw_df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Predictor feature '{col}' contains {null_count} missing values.")

    # 5. Extract X and y
    X = raw_df[FEATURE_COLUMNS].copy()
    y = raw_df[TARGET_COLUMN].map(TARGET_ENCODING).astype(int).copy()

    # Ensure all feature columns are numerical float64
    for col in FEATURE_COLUMNS:
        X[col] = X[col].astype(np.float64)

    # Strict isolation assertions
    assert TARGET_COLUMN not in X.columns, f"Target '{TARGET_COLUMN}' found inside feature matrix X!"
    for excl in EXCLUDED_COLUMNS:
        assert excl not in X.columns, f"Excluded column '{excl}' found inside feature matrix X!"
    assert X.shape == (569, 30), f"Expected X shape (569, 30), got {X.shape}"
    assert y.shape == (569,), f"Expected y shape (569,), got {y.shape}"

    metadata = {
        "dataset_path": os.path.abspath(csv_path),
        "total_samples": len(raw_df),
        "feature_count": len(FEATURE_COLUMNS),
        "target_counts": {
            "0 (Benign)": int((y == 0).sum()),
            "1 (Malignant)": int((y == 1).sum()),
        },
        "target_proportions": {
            "0 (Benign)": float(round((y == 0).mean() * 100, 2)),
            "1 (Malignant)": float(round((y == 1).mean() * 100, 2)),
        },
        "excluded_columns": EXCLUDED_COLUMNS,
    }

    return X, y, metadata


def prepare_data(
    csv_path: str,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Loads dataset, splits into stratified train/test partitions, and returns an unfitted preprocessor.

    Guarantees zero data leakage: The preprocessor returned is completely unfitted.
    It MUST be fitted only on X_train.

    Args:
        csv_path: Path to breast.csv.
        test_size: Proportion of dataset allocated to the test set (default 0.20).
        random_state: Seed for reproducible pseudorandom splitting (default 42).
        stratify: Whether to preserve class proportions in partitions (default True).

    Returns:
        X_train (pd.DataFrame): Training features (unscaled).
        X_test (pd.DataFrame): Test features (unscaled).
        y_train (pd.Series): Training target labels (0 or 1).
        y_test (pd.Series): Test target labels (0 or 1).
        preprocessor (ColumnTransformer): Unfitted preprocessor.
    """
    X, y, _ = load_dataset(csv_path)

    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Verification assertions
    assert list(X_train.columns) == FEATURE_COLUMNS, "Train column order mismatch!"
    assert list(X_test.columns) == FEATURE_COLUMNS, "Test column order mismatch!"
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, "Index overlap detected!"

    preprocessor = build_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor


def fit_and_transform_train(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> Tuple[np.ndarray, ColumnTransformer]:
    """Fits the preprocessor strictly on X_train and returns the transformed array and fitted object.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training features DataFrame.

    Returns:
        X_train_scaled (np.ndarray): Scaled training features.
        fitted_preprocessor (ColumnTransformer): Fitted preprocessor instance.
    """
    X_train_scaled = preprocessor.fit_transform(X_train)
    return X_train_scaled, preprocessor


def transform_test(
    fitted_preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Transforms test features using a previously fitted preprocessor without fitting.

    Args:
        fitted_preprocessor: Fitted ColumnTransformer instance.
        X_test: Test features DataFrame.

    Returns:
        X_test_scaled (np.ndarray): Scaled test features.
    """
    return fitted_preprocessor.transform(X_test)
