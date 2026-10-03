"""
Final Model Selection and Test Evaluation Script
Diabetes Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System

Tasks:
1. Recreates the exact 80/20 train/test split (614 train, 154 test, random_state=42, stratify=y).
2. Documents the technical selection of the final candidate model based strictly on Step 6 CV evidence.
3. Constructs a unified, production-ready scikit-learn Pipeline (ZeroToNan -> SimpleImputer(median) -> RobustScaler -> LogisticRegression).
4. Fits the complete pipeline exclusively on the 614 training samples.
5. Evaluates exactly once on the quarantined 154 test samples.
6. Serializes the complete trained pipeline as 'model.joblib'.
7. Exports comprehensive evaluation metrics to 'metrics.json'.
8. Exports the technical model selection report to 'model_selection.json'.
9. Verifies the saved joblib artifact can be loaded and perform valid inference on test data.
"""

import os
import sys
import json
import warnings
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure local imports work in both standalone and module mode
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from split_data import create_train_test_split
    from preprocessing import convert_zeros_to_nan, FEATURE_COLUMNS
except ImportError:
    from backend.models.diabetes.define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from backend.models.diabetes.split_data import create_train_test_split
    from backend.models.diabetes.preprocessing import convert_zeros_to_nan, FEATURE_COLUMNS


def build_final_pipeline() -> Pipeline:
    """
    Constructs the unified production Pipeline combining preprocessing and the selected classifier.
    
    Architecture:
    1. 'zero_to_nan': FunctionTransformer converting 0 to NaN for Glucose, BloodPressure, SkinThickness, Insulin, BMI.
    2. 'imputer': SimpleImputer(strategy='median') for robust median imputation.
    3. 'scaler': RobustScaler() for outlier-resistant centering and IQR scaling.
    4. 'classifier': LogisticRegression(max_iter=1000, random_state=42) as selected from Step 6.
    """
    pipeline = Pipeline([
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
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])
    return pipeline


def run_final_evaluation(dataset_path: str, output_dir: str):
    print("=" * 80)
    print("STEP 7: FINAL MODEL SELECTION & TEST EVALUATION - DIABETES MODEL")
    print("=" * 80)

    abs_dataset_path = os.path.abspath(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dataset location: {abs_dataset_path}")

    # 1. Exact 80/20 Stratified Split
    X_train, X_test, y_train, y_test, split_metrics = create_train_test_split(
        dataset_path=abs_dataset_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )

    n_train = len(X_train)
    n_test = len(X_test)
    print(f"\n1. Data Partitioning:")
    print(f"   - Training partition (X_train): {n_train} samples (80.0%)")
    print(f"   - Test partition (X_test):       {n_test} samples (20.0%) [STRICTLY QUARANTINED]")
    print(f"   - Train target counts: Class 0 = {(y_train == 0).sum()}, Class 1 = {(y_train == 1).sum()}")
    print(f"   - Test target counts:  Class 0 = {(y_test == 0).sum()}, Class 1 = {(y_test == 1).sum()}")

    # 2. Document Model Selection Decision from Step 6
    print("\n2. Model Selection Decision (from Step 6 Cross-Validation Evidence):")
    selection_rationale = (
        "Logistic Regression was selected based strictly on Step 6 5-fold cross-validation evidence on "
        "the 614-row training partition. It achieved the highest mean Accuracy (0.7899 +/- 0.0194), highest "
        "Precision (0.7652 +/- 0.0605), highest F1-Score (0.6583 +/- 0.0231), and highest ROC-AUC "
        "(0.8434 +/- 0.0188) among all candidates. It demonstrated the lowest standard deviation across "
        "validation folds (e.g. recall std of 0.0152). While tree-based ensembles (Gradient Boosting and Random "
        "Forest) yielded marginally higher recall (0.5981 and 0.5935 vs 0.5794, representing 4 and 3 fewer "
        "false negatives across 614 validation samples), they incurred substantially more false positives "
        "(64 and 57 vs 39 for Logistic Regression) and lower overall discrimination (ROC-AUC ~0.82 vs 0.8434). "
        "Logistic Regression also provides superior parsimony, resistance to overfitting on small tabular samples, "
        "linear interpretability, and well-calibrated probabilistic estimates."
    )
    print(f"   - Selected Model: Logistic Regression (max_iter=1000, random_state=42)")
    print(f"   - Technical Rationale: {selection_rationale}")

    # 3. Build & Train Pipeline on Training Data Only
    print("\n3. Training Final Model Pipeline strictly on X_train (614 rows):")
    final_pipeline = build_final_pipeline()
    final_pipeline.fit(X_train, y_train)
    print("   - Unified pipeline fitted successfully.")
    print("   - Pipeline steps: ['zero_to_nan', 'imputer', 'scaler', 'classifier']")

    # 4. Single Final Evaluation on Untouched Test Set
    print("\n4. Final Test Set Evaluation (154 rows):")
    y_test_pred = final_pipeline.predict(X_test)
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Metrics computation
    acc = float(round(accuracy_score(y_test, y_test_pred), 4))
    prec = float(round(precision_score(y_test, y_test_pred, zero_division=0), 4))
    rec = float(round(recall_score(y_test, y_test_pred, zero_division=0), 4))
    f1 = float(round(f1_score(y_test, y_test_pred, zero_division=0), 4))
    roc_auc = float(round(roc_auc_score(y_test, y_test_proba), 4))
    pr_auc = float(round(average_precision_score(y_test, y_test_proba), 4))

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    n_actual_0 = int((y_test == 0).sum())
    n_actual_1 = int((y_test == 1).sum())
    n_pred_0 = int((y_test_pred == 0).sum())
    n_pred_1 = int((y_test_pred == 1).sum())

    fnr = float(round(fn / n_actual_1, 4)) if n_actual_1 > 0 else 0.0
    fpr = float(round(fp / n_actual_0, 4)) if n_actual_0 > 0 else 0.0

    print(f"   - Test Accuracy:                {acc:.4f} ({acc*100:.2f}%)")
    print(f"   - Test Precision:               {prec:.4f}")
    print(f"   - Test Recall / Sensitivity:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"   - Test F1-Score:                {f1:.4f}")
    print(f"   - Test ROC-AUC:                 {roc_auc:.4f}")
    print(f"   - Test PR-AUC:                  {pr_auc:.4f}")
    print(f"\n   Confusion Matrix (Total N={n_test}):")
    print(f"     * True Negatives  (TN): {tn:<4} (Actual Class 0 correctly predicted Class 0)")
    print(f"     * False Positives (FP): {fp:<4} (Actual Class 0 incorrectly predicted Class 1)")
    print(f"     * False Negatives (FN): {fn:<4} (Actual Class 1 incorrectly predicted Class 0)")
    print(f"     * True Positives  (TP): {tp:<4} (Actual Class 1 correctly predicted Class 1)")
    print(f"\n   Error Rate Audit:")
    print(f"     * Test False Negative Rate (FNR): {fnr:.4f} ({fn}/{n_actual_1})")
    print(f"     * Test False Positive Rate (FPR): {fpr:.4f} ({fp}/{n_actual_0})")
    print(f"     * Actual Class Distribution:    Class 0 = {n_actual_0}, Class 1 = {n_actual_1}")
    print(f"     * Predicted Class Distribution: Class 0 = {n_pred_0}, Class 1 = {n_pred_1}")

    # 5. Save Model Artifact
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(final_pipeline, model_path)
    print(f"\n5. Saved Final Model Artifact to: {os.path.abspath(model_path)}")
    print(f"   - Artifact file size: {os.path.getsize(model_path)} bytes")

    # 6. Save model_selection.json
    selection_file = os.path.join(output_dir, "model_selection.json")
    selection_data = {
        "model_domain": "diabetes",
        "dataset_path": "datasets/diabetes/diabetes.csv",
        "evaluation_protocol": "5-fold StratifiedKFold on X_train (614 rows); final test set (154 rows) strictly quarantined",
        "candidate_models_evaluated": [
            "Logistic_Regression",
            "Random_Forest",
            "Support_Vector_Classifier",
            "Gradient_Boosting",
            "XGBoost"
        ],
        "step_6_cross_validation_metrics": {
            "Logistic_Regression": {
                "accuracy": "0.7899 +/- 0.0194",
                "precision": "0.7652 +/- 0.0605",
                "recall_sensitivity": "0.5794 +/- 0.0152",
                "f1_score": "0.6583 +/- 0.0231",
                "roc_auc": "0.8434 +/- 0.0188",
                "false_negatives": 90,
                "false_positives": 39
            },
            "Random_Forest": {
                "accuracy": "0.7655 +/- 0.0254",
                "precision": "0.6965 +/- 0.0629",
                "recall_sensitivity": "0.5935 +/- 0.0180",
                "f1_score": "0.6391 +/- 0.0244",
                "roc_auc": "0.8209 +/- 0.0211",
                "false_negatives": 87,
                "false_positives": 57
            },
            "Support_Vector_Classifier": {
                "accuracy": "0.7279 +/- 0.0542",
                "precision": "0.6794 +/- 0.1146",
                "recall_sensitivity": "0.3822 +/- 0.1784",
                "f1_score": "0.4714 +/- 0.1771",
                "roc_auc": "0.8145 +/- 0.0367",
                "false_negatives": 132,
                "false_positives": 35
            },
            "Gradient_Boosting": {
                "accuracy": "0.7557 +/- 0.0180",
                "precision": "0.6721 +/- 0.0508",
                "recall_sensitivity": "0.5981 +/- 0.0308",
                "f1_score": "0.6308 +/- 0.0134",
                "roc_auc": "0.8195 +/- 0.0165",
                "false_negatives": 86,
                "false_positives": 64
            },
            "XGBoost": {
                "accuracy": "0.7589 +/- 0.0344",
                "precision": "0.6877 +/- 0.0772",
                "recall_sensitivity": "0.5842 +/- 0.0513",
                "f1_score": "0.6285 +/- 0.0457",
                "roc_auc": "0.8205 +/- 0.0249",
                "false_negatives": 89,
                "false_positives": 59
            }
        },
        "selected_model": "Logistic_Regression",
        "selected_model_parameters": {
            "max_iter": 1000,
            "random_state": 42
        },
        "technical_selection_rationale": selection_rationale,
        "trade_off_analysis": {
            "recall_vs_precision": (
                "Gradient Boosting and Random Forest achieved 4 and 3 fewer false negatives respectively across "
                "the 614 training validation folds (86 and 87 vs 90 FNs for Logistic Regression). However, this "
                "marginal recall gain (+1.87% and +1.41%) required a substantial increase in false positives "
                "(64 and 57 FPs vs 39 for Logistic Regression, representing +64.1% and +46.2% more false alarms). "
                "Logistic Regression maintained superior overall screening discrimination (ROC-AUC 0.8434 vs 0.8195 "
                "and 0.8209) and the highest F1-score (0.6583)."
            ),
            "variance_stability": (
                "Logistic Regression demonstrated the lowest recall variance across folds (std 0.0152) compared to "
                "Gradient Boosting (std 0.0308) and XGBoost (std 0.0513), confirming consistent behavior across partitions."
            ),
            "simplicity_and_calibration": (
                "As an L2-regularized linear model, Logistic Regression provides smooth native probability calibration, "
                "low computational inference latency, and high robustness against overfitting on small tabular samples."
            )
        },
        "quarantine_confirmation": {
            "test_set_used_during_selection": False,
            "test_rows_quarantined": 154
        }
    }

    with open(selection_file, "w", encoding="utf-8") as f:
        json.dump(selection_data, f, indent=2)
    print(f"6. Saved Model Selection Report to: {os.path.abspath(selection_file)}")

    # 7. Save metrics.json
    metrics_file = os.path.join(output_dir, "metrics.json")
    metrics_data = {
        "model_domain": "diabetes",
        "dataset_path": "datasets/diabetes/diabetes.csv",
        "random_state": 42,
        "test_size": 0.20,
        "train_samples": n_train,
        "test_samples": n_test,
        "target_variable": TARGET_COLUMN,
        "target_classes": {
            "0": "Class 0 (Negative)",
            "1": "Class 1 (Positive)"
        },
        "selected_model": "Logistic_Regression",
        "selected_model_parameters": {
            "max_iter": 1000,
            "random_state": 42
        },
        "cross_validation_metrics_step_6": {
            "accuracy": {"mean": 0.7899, "std": 0.0194},
            "precision": {"mean": 0.7652, "std": 0.0605},
            "recall_sensitivity": {"mean": 0.5794, "std": 0.0152},
            "f1_score": {"mean": 0.6583, "std": 0.0231},
            "roc_auc": {"mean": 0.8434, "std": 0.0188},
            "screening_error_counts": {
                "total_false_negatives": 90,
                "total_false_positives": 39,
                "total_positive_cases": 214,
                "total_negative_cases": 400
            }
        },
        "test_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall_sensitivity": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp
            },
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "actual_class_distribution": {
                "class_0": n_actual_0,
                "class_1": n_actual_1
            },
            "predicted_class_distribution": {
                "class_0": n_pred_0,
                "class_1": n_pred_1
            }
        },
        "disclaimer": "This model provides an AI-based risk screening estimate and does not constitute a definitive medical diagnosis."
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"7. Saved Final Evaluation Metrics to: {os.path.abspath(metrics_file)}")

    # 8. Post-Save Verification: Load model.joblib and test inference
    print("\n8. Model Loading and Inference Verification:")
    loaded_pipeline = joblib.load(model_path)
    loaded_pred = loaded_pipeline.predict(X_test)
    loaded_proba = loaded_pipeline.predict_proba(X_test)[:, 1]

    # Verify predictions
    assert len(loaded_pred) == 154, f"Loaded prediction count {len(loaded_pred)} != 154"
    assert len(loaded_proba) == 154, f"Loaded probability count {len(loaded_proba)} != 154"
    assert not np.isnan(loaded_pred).any(), "NaN found in loaded model predictions!"
    assert not np.isinf(loaded_pred).any(), "Inf found in loaded model predictions!"
    assert not np.isnan(loaded_proba).any(), "NaN found in loaded model probabilities!"
    assert not np.isinf(loaded_proba).any(), "Inf found in loaded model probabilities!"
    assert (loaded_proba >= 0.0).all() and (loaded_proba <= 1.0).all(), "Loaded probabilities outside [0, 1]!"
    assert np.array_equal(loaded_pred, y_test_pred), "Loaded predictions do not match in-memory predictions!"
    assert np.allclose(loaded_proba, y_test_proba), "Loaded probabilities do not match in-memory probabilities!"

    loaded_cm = confusion_matrix(y_test, loaded_pred)
    assert loaded_cm.sum() == 154, f"Confusion matrix total {loaded_cm.sum()} != 154"

    print("   - Loaded model artifact successfully via joblib.load.")
    print("   - Loaded model predictions count: 154 (Matches test set).")
    print("   - Probability values verified finite and strictly bounded within [0.0, 1.0].")
    print("   - Loaded model outputs match in-memory outputs with 100% identity.")
    print("   - Confusion matrix sum: 154 (100% accounted for).")

    print("\n" + "=" * 80)
    print("STEP 7 COMPLETE")
    print("=" * 80)


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

    run_final_evaluation(selected_dataset, current_dir)
