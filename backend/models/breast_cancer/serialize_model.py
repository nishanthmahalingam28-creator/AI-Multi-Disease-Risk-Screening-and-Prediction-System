"""Model and Preprocessor Artifact Serialization Module.
Breast Cancer Model - Member 1
AI Multi-Disease Risk Screening and Prediction System

Fits the final Pipeline (ColumnTransformer + LogisticRegression) strictly
on the full training partition (N_train = 455) and serializes:
1. backend/models/breast_cancer/model.joblib (full Pipeline)
2. backend/models/breast_cancer/preprocessor.joblib (ColumnTransformer)
3. backend/models/breast_cancer/feature_schema.json (validated schema metadata)
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import (
    FEATURE_COLUMNS,
    MEAN_FEATURES,
    SE_FEATURES,
    WORST_FEATURES,
    TARGET_COLUMN,
    TARGET_ENCODING,
    EXCLUDED_COLUMNS,
    build_preprocessor,
    prepare_data,
    load_dataset,
)


def serialize_final_artifacts(
    csv_path: str,
    output_model_path: str,
    output_preprocessor_path: str,
    output_schema_path: str,
):
    print("=" * 80)
    print("STEP 8: FINAL MODEL & PREPROCESSOR ARTIFACT SERIALIZATION")
    print("=" * 80)

    # 1. Load data and split
    print(f"Loading dataset from: {os.path.abspath(csv_path)}")
    X_train, X_test, y_train, y_test, _ = prepare_data(
        csv_path, test_size=0.20, random_state=42, stratify=True
    )
    print(f"Full training partition size: {len(X_train)} samples (X_test quarantined: {len(X_test)} samples)")

    # 2. Build Pipeline: Preprocessor + LogisticRegression
    print("\nConstructing scikit-learn Pipeline with ColumnTransformer and LogisticRegression...")
    preprocessor = build_preprocessor()
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # 3. Fit Pipeline STRICTLY on X_train only
    print(f"Fitting Pipeline strictly on X_train (N = {len(X_train)} samples)...")
    pipeline.fit(X_train, y_train)
    print("Pipeline fitted successfully.")

    # 4. Serialize model.joblib (complete Pipeline)
    joblib.dump(pipeline, output_model_path)
    model_size = os.path.getsize(output_model_path)
    print(f"  -> Saved model artifact to: {output_model_path} ({model_size} bytes)")

    # 5. Serialize preprocessor.joblib (fitted ColumnTransformer)
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    joblib.dump(fitted_preprocessor, output_preprocessor_path)
    prep_size = os.path.getsize(output_preprocessor_path)
    print(f"  -> Saved preprocessor artifact to: {output_preprocessor_path} ({prep_size} bytes)")

    # 6. Generate and save validated feature_schema.json
    print(f"\nGenerating and saving feature schema metadata to: {output_schema_path}...")
    raw_df = pd.read_csv(csv_path)

    features_meta = []
    for col in FEATURE_COLUMNS:
        features_meta.append(
            {
                "name": col,
                "type": "numerical",
                "min": float(round(raw_df[col].min(), 4)),
                "max": float(round(raw_df[col].max(), 4)),
                "scaling": "StandardScaler",
                "description": f"Fine Needle Aspirate (FNA) digitized cell nucleus metric: {col}",
            }
        )

    schema = {
        "disease": "breast_cancer",
        "model_name": "breast_cancer_risk_screening",
        "version": "1.0.0",
        "task": "binary_classification",
        "feature_count": len(FEATURE_COLUMNS),
        "expected_numeric_input_type": "float64",
        "feature_names": FEATURE_COLUMNS,
        "feature_order": FEATURE_COLUMNS,
        "feature_groups": {
            "mean_features": MEAN_FEATURES,
            "se_features": SE_FEATURES,
            "worst_features": WORST_FEATURES,
        },
        "target": {
            "name": TARGET_COLUMN,
            "positive_class": {
                "value": 1,
                "label": "Malignant (M)",
                "description": "High risk screening estimate (malignant breast lesion)",
            },
            "negative_class": {
                "value": 0,
                "label": "Benign (B)",
                "description": "Low risk screening estimate (benign breast lesion)",
            },
        },
        "target_mapping": TARGET_ENCODING,
        "target_classes": {
            "0": "Benign (B)",
            "1": "Malignant (M)",
        },
        "excluded_columns": [
            {
                "name": "id",
                "reason": "Non-clinical arbitrary database identifier with zero physiological value",
            },
            {
                "name": "Unnamed: 32",
                "reason": "CSV parser artifact from trailing comma; contains 100% missing values",
            },
        ],
        "model_input_requirements": {
            "input_format": "Tabular pandas DataFrame or 2D array-like with 30 columns",
            "required_column_count": 30,
            "missing_values_allowed": False,
            "ordering_enforced": True,
            "feature_dtype": "float64",
        },
        "features": features_meta,
        "disclaimer": "This model provides an AI screening estimate for risk evaluation and does not replace professional histological diagnosis.",
    }

    with open(output_schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print("Feature schema written successfully.")

    print("\n" + "=" * 80)
    print("SERIALIZATION PIPELINE COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    default_dataset = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "datasets", "breast_cancer", "breast.csv"
    )
    if not os.path.exists(default_dataset):
        default_dataset = os.path.join("datasets", "breast_cancer", "breast.csv")

    model_file = os.path.join(current_dir, "model.joblib")
    prep_file = os.path.join(current_dir, "preprocessor.joblib")
    schema_file = os.path.join(current_dir, "feature_schema.json")

    serialize_final_artifacts(default_dataset, model_file, prep_file, schema_file)
