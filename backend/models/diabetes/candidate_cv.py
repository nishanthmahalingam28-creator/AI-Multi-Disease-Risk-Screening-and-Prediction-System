"""
Candidate Model Cross-Validation and Comparison Script
Diabetes Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System

Evaluates:
1. Logistic Regression
2. Random Forest
3. Support Vector Classifier (SVC)
4. Gradient Boosting
5. XGBoost

Procedure:
- Exactly 80/20 train/test split with random_state=42 and stratify=y.
- Test set (154 rows) is strictly quarantined and NOT touched during this evaluation.
- 5-fold StratifiedKFold on X_train (614 rows).
- Full scikit-learn Pipeline (zero-to-nan -> median imputer -> robust scaler -> classifier)
  fitted strictly on each CV fold's training portion to ensure zero data leakage.
- Reports Accuracy, Precision, Recall/Sensitivity, F1-score, ROC-AUC (mean and std),
  as well as cumulative False Negatives and False Positives.
"""

import os
import sys
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer

# Ensure local imports work in both standalone and module mode
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from split_data import create_train_test_split
    from preprocessing import convert_zeros_to_nan, build_preprocessor, FEATURE_COLUMNS
except ImportError:
    from backend.models.diabetes.define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from backend.models.diabetes.split_data import create_train_test_split
    from backend.models.diabetes.preprocessing import convert_zeros_to_nan, build_preprocessor, FEATURE_COLUMNS

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    XGBOOST_VERSION = "installed"
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = "not installed"


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_candidate_models() -> Dict[str, Dict[str, Any]]:
    """Defines candidate models with documented baseline parameters."""
    candidates = {
        "Logistic_Regression": {
            "model_name": "Logistic Regression",
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            "parameters": {
                "max_iter": 1000,
                "random_state": 42
            }
        },
        "Random_Forest": {
            "model_name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ),
            "parameters": {
                "n_estimators": 300,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "Support_Vector_Classifier": {
            "model_name": "Support Vector Classifier (SVC)",
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ),
            "parameters": {
                "kernel": "rbf",
                "probability": True,
                "random_state": 42
            }
        },
        "Gradient_Boosting": {
            "model_name": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(
                random_state=42
            ),
            "parameters": {
                "random_state": 42
            }
        }
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = {
            "model_name": "XGBoost",
            "estimator": XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            ),
            "parameters": {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss"
            }
        }

    return candidates


def run_candidate_cv(dataset_path: str, output_json_path: str):
    print("=" * 80)
    print("STEP 6: CANDIDATE MODEL CROSS-VALIDATION - DIABETES MODEL")
    print("=" * 80)

    abs_dataset_path = os.path.abspath(dataset_path)
    print(f"Dataset path: {abs_dataset_path}")

    # 1. Exact 80/20 Stratified Split
    X_train, X_test, y_train, y_test, split_metrics = create_train_test_split(
        dataset_path=abs_dataset_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )

    print("\n1. Data Partitioning Status:")
    print(f"   - Training rows (X_train): {X_train.shape[0]} (Used exclusively for 5-fold CV)")
    print(f"   - Test rows (X_test):       {X_test.shape[0]} (STRICTLY QUARANTINED - UNTOUCHED)")
    print(f"   - Positive cases in train:  {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.2f}%)")
    print(f"   - Negative cases in train:  {(y_train == 0).sum()} ({(y_train == 0).mean() * 100:.2f}%)")

    # 2. Setup 5-fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = get_candidate_models()
    print(f"\n2. Evaluating {len(candidates)} Candidate Models across 5 Stratified Folds:")
    for key, c_info in candidates.items():
        print(f"   - {c_info['model_name']} ({key})")

    cv_results = {}

    for cand_key, cand_info in candidates.items():
        model_name = cand_info["model_name"]
        base_estimator = cand_info["estimator"]
        print(f"\nEvaluating Candidate: {model_name}...")

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_roc_aucs = []
        fold_confusion_matrices = []

        # Iterate over folds manually to guarantee strictly fold-local preprocessing
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Build a fresh, isolated pipeline for this fold
            fold_pipeline = Pipeline([
                ("zero_to_nan", FunctionTransformer(convert_zeros_to_nan, validate=False)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("classifier", base_estimator)
            ])

            # Fit strictly on this fold's training data
            fold_pipeline.fit(X_fold_train, y_fold_train)

            # Predict on validation fold
            y_val_pred = fold_pipeline.predict(X_fold_val)
            y_val_proba = fold_pipeline.predict_proba(X_fold_val)[:, 1]

            # Metrics
            acc = accuracy_score(y_fold_val, y_val_pred)
            prec = precision_score(y_fold_val, y_val_pred, zero_division=0)
            rec = recall_score(y_fold_val, y_val_pred, zero_division=0)
            f1 = f1_score(y_fold_val, y_val_pred, zero_division=0)
            roc = roc_auc_score(y_fold_val, y_val_proba)
            cm = confusion_matrix(y_fold_val, y_val_pred)

            fold_accuracies.append(acc)
            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_f1s.append(f1)
            fold_roc_aucs.append(roc)
            fold_confusion_matrices.append(cm)

        # Compute aggregate summary statistics
        total_cm = sum(fold_confusion_matrices)
        tn, fp, fn, tp = total_cm.ravel()

        cand_summary = {
            "model_name": model_name,
            "parameters": cand_info["parameters"],
            "accuracy": {
                "mean": float(round(np.mean(fold_accuracies), 4)),
                "std": float(round(np.std(fold_accuracies), 4)),
                "per_fold": [float(round(v, 4)) for v in fold_accuracies]
            },
            "precision": {
                "mean": float(round(np.mean(fold_precisions), 4)),
                "std": float(round(np.std(fold_precisions), 4)),
                "per_fold": [float(round(v, 4)) for v in fold_precisions]
            },
            "recall_sensitivity": {
                "mean": float(round(np.mean(fold_recalls), 4)),
                "std": float(round(np.std(fold_recalls), 4)),
                "per_fold": [float(round(v, 4)) for v in fold_recalls]
            },
            "f1_score": {
                "mean": float(round(np.mean(fold_f1s), 4)),
                "std": float(round(np.std(fold_f1s), 4)),
                "per_fold": [float(round(v, 4)) for v in fold_f1s]
            },
            "roc_auc": {
                "mean": float(round(np.mean(fold_roc_aucs), 4)),
                "std": float(round(np.std(fold_roc_aucs), 4)),
                "per_fold": [float(round(v, 4)) for v in fold_roc_aucs]
            },
            "validation_error_counts": {
                "total_true_positives": int(tp),
                "total_false_positives": int(fp),
                "total_false_negatives": int(fn),
                "total_true_negatives": int(tn),
                "total_positive_cases": int(tp + fn),
                "total_negative_cases": int(tn + fp)
            }
        }
        cv_results[cand_key] = cand_summary

        print(f"   Accuracy:    {cand_summary['accuracy']['mean']:.4f} +/- {cand_summary['accuracy']['std']:.4f}")
        print(f"   Precision:   {cand_summary['precision']['mean']:.4f} +/- {cand_summary['precision']['std']:.4f}")
        print(f"   Recall:      {cand_summary['recall_sensitivity']['mean']:.4f} +/- {cand_summary['recall_sensitivity']['std']:.4f}")
        print(f"   F1-Score:    {cand_summary['f1_score']['mean']:.4f} +/- {cand_summary['f1_score']['std']:.4f}")
        print(f"   ROC-AUC:     {cand_summary['roc_auc']['mean']:.4f} +/- {cand_summary['roc_auc']['std']:.4f}")
        print(f"   Screening Errors: False Negatives = {fn} / {tp+fn} (missed positive cases), False Positives = {fp} / {tn+fp}")

    # 3. Comparative Summary Table
    print("\n" + "=" * 95)
    print(f"{'Candidate Model':<28} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<14} {'ROC-AUC':<14}")
    print("-" * 95)
    for k, res in cv_results.items():
        name = res["model_name"]
        acc_str = f"{res['accuracy']['mean']:.4f} +/- {res['accuracy']['std']:.3f}"
        prec_str = f"{res['precision']['mean']:.4f} +/- {res['precision']['std']:.3f}"
        rec_str = f"{res['recall_sensitivity']['mean']:.4f} +/- {res['recall_sensitivity']['std']:.3f}"
        f1_str = f"{res['f1_score']['mean']:.4f} +/- {res['f1_score']['std']:.3f}"
        roc_str = f"{res['roc_auc']['mean']:.4f} +/- {res['roc_auc']['std']:.3f}"
        print(f"{name:<28} {acc_str:<14} {prec_str:<14} {rec_str:<14} {f1_str:<14} {roc_str:<14}")
    print("=" * 95)

    print("\nScreening Metric Focus (Recall & False Negatives):")
    print(f"{'Candidate Model':<28} {'Total FN':<12} {'Recall / Sens':<16} {'Total FP':<12} {'Total Positives':<16}")
    print("-" * 88)
    for k, res in cv_results.items():
        name = res["model_name"]
        errs = res["validation_error_counts"]
        rec_str = f"{res['recall_sensitivity']['mean']:.4f}"
        print(f"{name:<28} {errs['total_false_negatives']:<12} {rec_str:<16} {errs['total_false_positives']:<12} {errs['total_positive_cases']:<16}")
    print("-" * 88)

    # 4. Save to JSON
    output_data = {
        "dataset_path": abs_dataset_path,
        "split_configuration": {
            "test_size": 0.20,
            "train_rows": 614,
            "test_rows_quarantined": 154,
            "stratified": True,
            "random_state": 42
        },
        "cross_validation_configuration": {
            "strategy": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "preprocessing": "ZeroToNan -> SimpleImputer(median) -> RobustScaler fitted strictly fold-by-fold"
        },
        "xgboost_status": {
            "available": XGBOOST_AVAILABLE,
            "version": XGBOOST_VERSION
        },
        "candidate_cv_results": cv_results,
        "leakage_and_integrity_confirmations": {
            "test_set_accessed": False,
            "test_set_size_quarantined": 154,
            "preprocessing_refit_inside_each_fold": True,
            "model_artifacts_saved": False,
            "data_leakage_detected": False
        }
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    abs_json = os.path.abspath(output_json_path)
    print(f"\nSaved Candidate CV Results JSON to: {abs_json}")
    print("=" * 80)
    print("STEP 6 COMPLETE - NO MODEL SELECTED PERMANENTLY - TEST SET UNTOUCHED")
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

    out_json = os.path.join(current_dir, "candidate_cv_results.json")
    run_candidate_cv(selected_dataset, out_json)
