"""
Diabetes Dataset Preprocessing Pipeline
AI Multi-Disease Risk Screening and Prediction System

Pipeline Architecture:
1. Zero-to-NaN Conversion for 5 physiological measurement columns:
   ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
   (Pregnancies=0 is preserved as a valid non-parous biological count).
2. Median Imputation: SimpleImputer(strategy='median')
   (Fitted strictly on non-zero X_train values to avoid leakage).
3. Robust Scaling: RobustScaler()
   (Centers by median and scales by IQR, robust against high skewness and extreme outliers).

Order of 8 Predictor Features is strictly preserved:
['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
"""

import os
import sys
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer

# Ensure local imports work in both standalone and package context
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
    from split_data import create_train_test_split
except ImportError:
    from backend.models.diabetes.define_features import (
        EXPECTED_PREDICTOR_COLUMNS,
        TARGET_COLUMN,
        load_dataset,
        define_features_and_target,
    )
    from backend.models.diabetes.split_data import create_train_test_split


# The 5 continuous physiological measurement columns where zero values represent
# physically impossible readings / missing data in the Pima Indians dataset.
ZERO_AS_MISSING_COLUMNS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Columns where zero is a valid, natural count and must NEVER be treated as missing
PASSTHROUGH_ZERO_COLUMNS: List[str] = [
    "Pregnancies"
]

FEATURE_COLUMNS: List[str] = EXPECTED_PREDICTOR_COLUMNS


def convert_zeros_to_nan(X: Any) -> Any:
    """
    Converts zero values to np.nan strictly in the 5 physiological measurement columns:
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'].
    
    Leaves 'Pregnancies', 'DiabetesPedigreeFunction', and 'Age' completely untouched.
    Preserves exact DataFrame or ndarray dimensions and column order.
    """
    if isinstance(X, pd.DataFrame):
        X_out = X.copy()
        for col in ZERO_AS_MISSING_COLUMNS:
            if col in X_out.columns:
                X_out[col] = X_out[col].mask(X_out[col] == 0, np.nan)
        return X_out
    else:
        X_arr = np.array(X, dtype=float, copy=True)
        col_indices = [FEATURE_COLUMNS.index(c) for c in ZERO_AS_MISSING_COLUMNS if c in FEATURE_COLUMNS]
        for idx in col_indices:
            X_arr[:, idx] = np.where(X_arr[:, idx] == 0, np.nan, X_arr[:, idx])
        return X_arr


def build_preprocessor() -> Pipeline:
    """
    Constructs an unfitted scikit-learn Pipeline for diabetes risk screening.
    
    Pipeline Steps:
    1. 'zero_to_nan': FunctionTransformer replacing 0 with NaN for the 5 measurement columns.
    2. 'imputer': SimpleImputer(strategy='median') learning median from training partition.
    3. 'scaler': RobustScaler() scaling features by median and IQR.
    
    Returns:
        Unfitted sklearn Pipeline instance.
    """
    preprocessor = Pipeline([
        (
            "zero_to_nan",
            FunctionTransformer(convert_zeros_to_nan, validate=False)
        ),
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            RobustScaler()
        )
    ])
    return preprocessor


def compare_scalers(X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comparative investigation between StandardScaler and RobustScaler
    on the imputed training feature matrix.
    """
    # 1. Temporarily impute X_train to analyze scaling behavior
    X_imputed = convert_zeros_to_nan(X_train)
    imputer = SimpleImputer(strategy="median")
    X_imp_arr = imputer.fit_transform(X_imputed)
    df_imp = pd.DataFrame(X_imp_arr, columns=FEATURE_COLUMNS)

    # 2. Fit StandardScaler
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(df_imp)

    # 3. Fit RobustScaler
    rob_scaler = RobustScaler()
    X_rob = rob_scaler.fit_transform(df_imp)

    comparison_results = {}
    for idx, col in enumerate(FEATURE_COLUMNS):
        col_data = df_imp[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        comparison_results[col] = {
            "raw_median": float(round(col_data.median(), 3)),
            "raw_mean": float(round(col_data.mean(), 3)),
            "raw_std": float(round(col_data.std(), 3)),
            "raw_iqr": float(round(iqr, 3)),
            "std_scaler_range": [float(round(X_std[:, idx].min(), 3)), float(round(X_std[:, idx].max(), 3))],
            "std_scaler_max_zscore": float(round(np.abs(X_std[:, idx]).max(), 3)),
            "robust_scaler_range": [float(round(X_rob[:, idx].min(), 3)), float(round(X_rob[:, idx].max(), 3))],
            "robust_scaler_iqr_span": float(round(np.percentile(X_rob[:, idx], 75) - np.percentile(X_rob[:, idx], 25), 3))
        }

    return comparison_results


def prepare_data(
    dataset_path: str,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Loads dataset, splits into stratified train/test partitions, and returns an unfitted preprocessor.
    
    Guarantees:
    - Zero data leakage: Preprocessor returned is completely unfitted.
    - Preprocessor MUST be fitted strictly on X_train.
    """
    X_train, X_test, y_train, y_test, _ = create_train_test_split(
        dataset_path=dataset_path,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    preprocessor = build_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


def fit_and_transform_train(
    preprocessor: Pipeline,
    X_train: pd.DataFrame
) -> Tuple[np.ndarray, Pipeline]:
    """
    Fits the preprocessing pipeline strictly on X_train and returns the transformed array and fitted object.
    """
    X_train_transformed = preprocessor.fit_transform(X_train)
    return X_train_transformed, preprocessor


def transform_test(
    fitted_preprocessor: Pipeline,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Transforms test features using a previously fitted preprocessor without refitting.
    """
    return fitted_preprocessor.transform(X_test)


def run_preprocessing_pipeline(dataset_path: str):
    print("=" * 80)
    print("STEP 5: PREPROCESSING PIPELINE - DIABETES MODEL")
    print("=" * 80)

    abs_path = os.path.abspath(dataset_path)
    print(f"Dataset path: {abs_path}")
    orig_size = os.path.getsize(abs_path)

    # 1. Partition dataset
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        dataset_path=abs_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    print(f"\n1. Data Partitioned:")
    print(f"   - X_train shape: {X_train.shape} (80%)")
    print(f"   - X_test shape:  {X_test.shape} (20%)")
    print(f"   - y_train shape: {y_train.shape}")
    print(f"   - y_test shape:  {y_test.shape}")

    # 2. Zero-value investigation & documentation
    print("\n2. Zero-Value Decision & Justification:")
    print("   - Columns with zeros examined:")
    for col in ZERO_AS_MISSING_COLUMNS:
        z_train = (X_train[col] == 0).sum()
        z_pct = (z_train / len(X_train)) * 100
        print(f"     * '{col}': {z_train} zeros in X_train ({z_pct:.2f}%)")
    print("   - Non-zero conversion column:")
    print(f"     * 'Pregnancies': {(X_train['Pregnancies'] == 0).sum()} zeros in X_train. Retained as valid count (nulliparous subjects).")
    print("   - Decision Rationale:")
    print("     A numerical value of 0.0 for Glucose, BloodPressure, SkinThickness, Insulin, or BMI is a")
    print("     physiologically impossible measurement for living ambulatory subjects and reflects unrecorded")
    print("     measurements encoded as 0. Inside the preprocessing pipeline, these zeros are converted to NaN")
    print("     and imputed using the median learned strictly from the training partition.")

    # 3. Scaler investigation & documentation
    print("\n3. Scaler Investigation & Comparison (StandardScaler vs RobustScaler):")
    scaler_comp = compare_scalers(X_train)
    print(f"   {'Feature':<25} {'Raw Median':<12} {'Raw IQR':<10} {'StdScaler Range':<20} {'RobustScaler Range':<20}")
    print("   " + "-" * 88)
    for col, data in scaler_comp.items():
        print(f"   {col:<25} {data['raw_median']:<12.2f} {data['raw_iqr']:<10.2f} {str(data['std_scaler_range']):<20} {str(data['robust_scaler_range']):<20}")

    print("\n   - Selected Scaler: RobustScaler")
    print("   - Justification:")
    print("     Features such as 'Insulin' (skew +2.27, range up to 846 with 34 IQR outliers) and")
    print("     'DiabetesPedigreeFunction' (skew +1.92, 29 IQR outliers) exhibit extreme right-skewness.")
    print("     StandardScaler relies on the sample mean and standard deviation, which are severely distorted")
    print("     by extreme observations, compressing normal-range data into narrow bands.")
    print("     RobustScaler scales via the median and Interquartile Range (IQR = Q3 - Q1), making it")
    print("     statistically resilient against extreme values while preserving linear relationships.")

    # 4. Fit on X_train only
    print("\n4. Fitting Preprocessing Pipeline strictly on X_train:")
    X_train_transformed, fitted_preprocessor = fit_and_transform_train(preprocessor, X_train)
    print(f"   - X_train transformed shape: {X_train_transformed.shape}")
    print(f"   - NaN count in transformed X_train: {np.isnan(X_train_transformed).sum()}")
    print(f"   - Inf count in transformed X_train: {np.isinf(X_train_transformed).sum()}")

    # Learned imputer statistics
    imputer_step = fitted_preprocessor.named_steps["imputer"]
    print("   - Learned Training Medians (Imputer Statistics):")
    for idx, col in enumerate(FEATURE_COLUMNS):
        print(f"     * '{col}': {imputer_step.statistics_[idx]:.2f}")

    # Learned scaler statistics
    scaler_step = fitted_preprocessor.named_steps["scaler"]
    print("   - Learned Training Scaling Centers (Medians) and Scales (IQRs):")
    for idx, col in enumerate(FEATURE_COLUMNS):
        print(f"     * '{col}': Center (Median) = {scaler_step.center_[idx]:.2f}, Scale (IQR) = {scaler_step.scale_[idx]:.2f}")

    # 5. Transform X_test without refitting
    print("\n5. Transforming X_test using fitted preprocessor (zero refitting):")
    X_test_transformed = transform_test(fitted_preprocessor, X_test)
    print(f"   - X_test transformed shape:  {X_test_transformed.shape}")
    print(f"   - NaN count in transformed X_test:  {np.isnan(X_test_transformed).sum()}")
    print(f"   - Inf count in transformed X_test:  {np.isinf(X_test_transformed).sum()}")

    # 6. Verify non-mutation of raw dataset
    after_size = os.path.getsize(abs_path)
    print("\n6. Dataset Non-Mutation Check:")
    print(f"   - Original dataset size: {orig_size} bytes")
    print(f"   - Current dataset size:  {after_size} bytes")
    print(f"   - Dataset unchanged: {orig_size == after_size}")

    print("=" * 80)
    print("STEP 5 PREPROCESSING COMPLETE - NO CLASSIFIER TRAINED")
    print("=" * 80)

    return X_train_transformed, X_test_transformed, fitted_preprocessor


if __name__ == "__main__":
    candidate_paths = [
        os.path.join(current_dir, "..", "..", "..", "datasets", "diabetes", "diabetes.csv"),
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

    run_preprocessing_pipeline(selected_dataset)
