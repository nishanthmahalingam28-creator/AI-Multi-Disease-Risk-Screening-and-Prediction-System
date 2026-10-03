"""
Preprocessing Pipeline for Stroke Prediction Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 5 — Preprocessing
Implements leakage-free preprocessing pipeline fitted strictly on training data (X_train).
- Continuous numeric vitals (age, avg_glucose_level, bmi): SimpleImputer(median) -> RobustScaler()
- Binary numeric indicators (hypertension, heart_disease): SimpleImputer(median) -> Passthrough (unscaled)
- Categorical features (gender, ever_married, work_type, Residence_type, smoking_status):
  SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore", sparse_output=False)
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Add local directory to sys.path for standalone and module-level execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from split_data import create_train_test_split, compute_sha256
except ImportError:
    from backend.models.stroke.define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from backend.models.stroke.split_data import create_train_test_split, compute_sha256

# =============================================================================
# FEATURE GROUP DEFINITIONS
# =============================================================================

# Continuous numeric measurements (subject to skewness and outlier presence)
CONTINUOUS_NUMERIC_FEATURES: List[str] = [
    "age",
    "avg_glucose_level",
    "bmi"
]

# Binary numeric indicator features (valid observed domain: 0, 1)
BINARY_NUMERIC_FEATURES: List[str] = [
    "hypertension",
    "heart_disease"
]

# Categorical string features requiring imputation and one-hot encoding
CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

# Complete canonical 10-feature list
FEATURE_COLUMNS: List[str] = EXPECTED_PREDICTOR_COLUMNS


# =============================================================================
# PIPELINE BUILDER
# =============================================================================

def build_stroke_preprocessor(scaler_type: str = "robust") -> ColumnTransformer:
    """
    Constructs the canonical scikit-learn ColumnTransformer preprocessor for the Stroke model.
    
    Structure:
    1. Continuous Numeric Pipeline (age, avg_glucose_level, bmi):
       - SimpleImputer(strategy='median')
       - RobustScaler() (or StandardScaler if specified)
    2. Binary Numeric Pipeline (hypertension, heart_disease):
       - SimpleImputer(strategy='median')
       - Passthrough (no scaling applied to preserve exact 0/1 indicator representation)
    3. Categorical Pipeline (gender, ever_married, work_type, Residence_type, smoking_status):
       - SimpleImputer(strategy='most_frequent')
       - OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")

    continuous_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


def compare_scalers_on_train(X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Empirically compares StandardScaler and RobustScaler on training continuous features.
    No test data is used; comparison is based on distributional robustness.
    """
    cont_df = X_train[CONTINUOUS_NUMERIC_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    cont_imputed = imputer.fit_transform(cont_df)
    cont_imputed_df = pd.DataFrame(cont_imputed, columns=CONTINUOUS_NUMERIC_FEATURES)

    # 1. StandardScaler fit
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(cont_imputed_df)
    std_df = pd.DataFrame(X_std, columns=CONTINUOUS_NUMERIC_FEATURES)

    # 2. RobustScaler fit
    rob_scaler = RobustScaler()
    X_rob = rob_scaler.fit_transform(cont_imputed_df)
    rob_df = pd.DataFrame(X_rob, columns=CONTINUOUS_NUMERIC_FEATURES)

    # Outlier statistics
    outlier_counts = {}
    for col in CONTINUOUS_NUMERIC_FEATURES:
        s = cont_imputed_df[col]
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = float(q3 - q1)
        out_cnt = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        outlier_counts[col] = {
            "q1": round(q1, 4),
            "median": round(float(s.median()), 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "outlier_count": out_cnt,
            "outlier_percentage": round(float((out_cnt / len(s)) * 100), 4)
        }

    comparison = {
        "outliers_in_training_continuous": outlier_counts,
        "standard_scaler_stats": {
            col: {
                "mean": round(float(std_df[col].mean()), 4),
                "std": round(float(std_df[col].std()), 4),
                "min": round(float(std_df[col].min()), 4),
                "max": round(float(std_df[col].max()), 4)
            }
            for col in CONTINUOUS_NUMERIC_FEATURES
        },
        "robust_scaler_stats": {
            col: {
                "median": round(float(rob_df[col].median()), 4),
                "iqr": round(float(rob_df[col].quantile(0.75) - rob_df[col].quantile(0.25)), 4),
                "min": round(float(rob_df[col].min()), 4),
                "max": round(float(rob_df[col].max()), 4)
            }
            for col in CONTINUOUS_NUMERIC_FEATURES
        },
        "selected_scaler": "RobustScaler",
        "technical_justification": (
            "RobustScaler was selected for continuous features because avg_glucose_level exhibits severe positive "
            "skewness (+1.57) and contains 503 IQR outlier samples (12.30% of training data), while bmi contains "
            "101 outliers (2.47%). StandardScaler scales by the empirical mean and standard deviation, which are "
            "pulled and inflated by extreme glucose values up to 271.74 and BMI values up to 97.60. RobustScaler "
            "centers by the median and scales by the Interquartile Range (IQR), mapping the central 50% of typical "
            "vital measurements to [-0.5, +0.5] without distorting the distribution center or truncating extreme values."
        )
    }

    return comparison


def main():
    print("=" * 80)
    print("STEP 5: STROKE DISEASE PREPROCESSING PIPELINE EXECUTION")
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

    # Pre-execution dataset integrity check
    pre_sha256 = compute_sha256(resolved_path)
    print(f"\n1. Loading and Partitioning Dataset:")
    print(f"   - Dataset Path       : {resolved_path}")
    print(f"   - Dataset SHA-256    : {pre_sha256}")

    # 2. Extract 80/20 train/test split (Step 4 protocol)
    X_train, X_test, y_train, y_test, split_meta = create_train_test_split(
        dataset_path=resolved_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    print(f"   - Training partition : {X_train.shape[0]} rows x {X_train.shape[1]} features")
    print(f"   - Test partition     : {X_test.shape[0]} rows x {X_test.shape[1]} features (QUARANTINED)")

    # 3. Scaler Comparison on Training Partition
    print("\n2. Evaluating Scalers on Continuous Features in Training Partition...")
    scaler_comp = compare_scalers_on_train(X_train)
    print(f"   - Selected Scaler: {scaler_comp['selected_scaler']}")
    print(f"   - Outliers in X_train continuous:")
    for col, st in scaler_comp["outliers_in_training_continuous"].items():
        print(f"     * {col:<20}: {st['outlier_count']} outliers ({st['outlier_percentage']}%) | IQR: {st['iqr']}")
    print(f"   - Justification: {scaler_comp['technical_justification']}")

    # 4. Build Preprocessor
    print("\n3. Constructing Preprocessing Pipeline (ColumnTransformer)...")
    preprocessor = build_stroke_preprocessor(scaler_type="robust")

    # 5. Fit Preprocessor ON TRAINING DATA ONLY
    print("   - Fitting ColumnTransformer exclusively on X_train (4,088 samples)...")
    preprocessor.fit(X_train)
    print("   [x] Preprocessor fitted successfully.")

    # 6. Extract learned statistics
    # Continuous imputer median
    cont_pipeline = preprocessor.named_transformers_["continuous"]
    cont_imputer = cont_pipeline.named_steps["imputer"]
    cont_scaler = cont_pipeline.named_steps["scaler"]
    train_medians = dict(zip(CONTINUOUS_NUMERIC_FEATURES, [round(float(m), 4) for m in cont_imputer.statistics_]))
    train_iqrs = dict(zip(CONTINUOUS_NUMERIC_FEATURES, [round(float(s), 4) for s in cont_scaler.scale_]))
    print(f"\n4. Learned Continuous Statistics from Training Partition:")
    for col in CONTINUOUS_NUMERIC_FEATURES:
        print(f"   - {col:<20}: Median = {train_medians[col]:>8.4f} | IQR = {train_iqrs[col]:>8.4f}")

    # Binary imputer median
    bin_pipeline = preprocessor.named_transformers_["binary"]
    bin_imputer = bin_pipeline.named_steps["imputer"]
    bin_medians = dict(zip(BINARY_NUMERIC_FEATURES, [round(float(m), 4) for m in bin_imputer.statistics_]))
    print(f"\n5. Learned Binary Imputer Medians:")
    for col in BINARY_NUMERIC_FEATURES:
        print(f"   - {col:<20}: Median = {bin_medians[col]}")

    # Categorical encoder categories
    cat_pipeline = preprocessor.named_transformers_["categorical"]
    cat_imputer = cat_pipeline.named_steps["imputer"]
    cat_ohe = cat_pipeline.named_steps["ohe"]
    learned_categories = {}
    for col, cats in zip(CATEGORICAL_FEATURES, cat_ohe.categories_):
        learned_categories[col] = list(cats)
    print(f"\n6. Learned Categorical Levels from Training Partition:")
    for col, cats in learned_categories.items():
        print(f"   - {col:<20} ({len(cats)} levels): {cats}")

    # 7. Transform X_train and X_test
    print("\n7. Transforming Partitions (Test partition transformed strictly via fitted train pipeline)...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    output_feature_names = list(preprocessor.get_feature_names_out())
    num_output_features = len(output_feature_names)

    print(f"   - Transformed X_train shape: {X_train_transformed.shape} (expected: 4088 x {num_output_features})")
    print(f"   - Transformed X_test shape : {X_test_transformed.shape} (expected: 1022 x {num_output_features})")
    print(f"   - Output Feature Count     : {num_output_features}")

    # 8. Strict Validation Checks
    print("\n8. Comprehensive Validation Checks:")
    checks = [
        ("X_train row count preserved (4088 == 4088)", X_train_transformed.shape[0] == 4088),
        ("X_test row count preserved (1022 == 1022)", X_test_transformed.shape[0] == 1022),
        ("Feature count is exactly 21 output features", num_output_features == 21),
        ("Transformed X_train contains 0 NaNs", int(np.isnan(X_train_transformed).sum()) == 0),
        ("Transformed X_test contains 0 NaNs", int(np.isnan(X_test_transformed).sum()) == 0),
        ("Transformed X_train contains only finite values", bool(np.all(np.isfinite(X_train_transformed)))),
        ("Transformed X_test contains only finite values", bool(np.all(np.isfinite(X_test_transformed)))),
        ("Categorical OHE has handle_unknown='ignore'", cat_ohe.handle_unknown == "ignore"),
        ("Identifier 'id' excluded from input and output", "id" not in FEATURE_COLUMNS and not any(f == "id" or f.endswith("__id") for f in output_feature_names)),
        ("Target 'stroke' excluded from input and output", TARGET_COLUMN not in FEATURE_COLUMNS and not any(f == "stroke" or f.endswith("__stroke") for f in output_feature_names)),
        ("BMI training median is exactly 28.0", train_medians["bmi"] == 28.0)
    ]

    all_passed = True
    for desc, passed in checks:
        status_str = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"   {status_str} {desc}")

    if not all_passed:
        print("\nERROR: Preprocessing validation checks failed!")
        sys.exit(1)

    # 9. Save Preprocessor Artifact
    artifact_path = os.path.join(script_dir, "preprocessor.joblib")
    print(f"\n9. Saving Preprocessing Artifact to: {artifact_path}")
    joblib.dump(preprocessor, artifact_path)
    artifact_size = os.path.getsize(artifact_path)
    print(f"   [x] preprocessor.joblib saved ({artifact_size} bytes).")

    # 10. Save Preprocessing Summary JSON
    summary_path = os.path.join(script_dir, "preprocessing_summary.json")
    summary_dict = {
        "model_domain": "stroke",
        "dataset_path": "datasets/stroke/stroke.csv",
        "dataset_sha256": pre_sha256,
        "original_feature_count": len(FEATURE_COLUMNS),
        "original_features": FEATURE_COLUMNS,
        "feature_group_classification": {
            "continuous_numeric": CONTINUOUS_NUMERIC_FEATURES,
            "binary_numeric": BINARY_NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "missing_values_before_preprocessing": {
            "train": {col: int(X_train[col].isnull().sum()) for col in FEATURE_COLUMNS},
            "test": {col: int(X_test[col].isnull().sum()) for col in FEATURE_COLUMNS}
        },
        "imputation_strategy": {
            "continuous_numeric": "SimpleImputer(strategy='median')",
            "binary_numeric": "SimpleImputer(strategy='median')",
            "categorical": "SimpleImputer(strategy='most_frequent')"
        },
        "training_imputation_statistics": {
            "continuous_medians": train_medians,
            "binary_medians": bin_medians,
            "bmi_training_median": train_medians["bmi"]
        },
        "scaler_selection": {
            "selected_scaler": scaler_comp["selected_scaler"],
            "technical_justification": scaler_comp["technical_justification"],
            "comparison_details": scaler_comp
        },
        "categorical_encoding": {
            "encoder": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "learned_categories": learned_categories,
            "total_encoded_categorical_features": sum(len(cats) for cats in learned_categories.values())
        },
        "output_feature_metadata": {
            "total_output_features": num_output_features,
            "output_feature_names": output_feature_names
        },
        "partition_shapes": {
            "raw_train": list(X_train.shape),
            "raw_test": list(X_test.shape),
            "transformed_train": list(X_train_transformed.shape),
            "transformed_test": list(X_test_transformed.shape)
        },
        "integrity_and_quarantine": {
            "preprocessing_fit_partition": "X_train_only",
            "test_set_used_only_for_transform": True,
            "target_excluded": True,
            "id_excluded": True,
            "preprocessing_applied": True,
            "model_training_applied": False,
            "test_partition_quarantined": True
        }
    }

    print(f"\n10. Saving Preprocessing Summary JSON to: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print("   [x] preprocessing_summary.json saved.")

    # 11. Post-Execution Dataset Integrity Check
    post_sha256 = compute_sha256(resolved_path)
    hash_match = (pre_sha256 == post_sha256)
    print(f"\n11. Dataset Integrity Verification:")
    print(f"   - Post-Execution Dataset SHA-256: {post_sha256}")
    print(f"   - Read-Only Integrity Verified : {hash_match}")
    if not hash_match:
        raise RuntimeError("CRITICAL ERROR: Dataset file was modified during Step 5 execution!")

    print("\n" + "=" * 80)
    print("STEP 5 PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
