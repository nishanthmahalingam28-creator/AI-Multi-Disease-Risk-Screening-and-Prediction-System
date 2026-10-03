"""
Candidate Model Cross-Validation and Comparison Script
Heart Disease Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 6 — Candidate Model Comparison
Evaluates 5 candidate classifiers using 5-fold StratifiedKFold cross-validation
exclusively on X_train (216 samples). The 54-sample test set is strictly quarantined.
Preprocessing (Approach B) is fitted fold-by-fold strictly inside each fold to guarantee
zero data leakage.
"""

import os
import sys
import json
import warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure local imports work in both standalone and module mode
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from split_data import create_train_test_split
    from preprocessing import build_preprocessing_pipeline, FEATURE_COLUMNS
except ImportError:
    from backend.models.heart_disease.define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from backend.models.heart_disease.split_data import create_train_test_split
    from backend.models.heart_disease.preprocessing import build_preprocessing_pipeline, FEATURE_COLUMNS

# Check XGBoost availability
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    import xgboost
    XGBOOST_VERSION = xgboost.__version__
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = "not installed"


def get_candidate_models() -> Dict[str, Any]:
    """
    Initializes candidate classifiers with standard, reproducible hyperparameters.
    """
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "Support_Vector_Classifier": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        )

    return models


def evaluate_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes 5-fold StratifiedKFold CV on X_train.
    CRITICAL: Preprocessor is constructed and fitted strictly on each fold's training portion.
    """
    models = get_candidate_models()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: Dict[str, Any] = {}

    print(f"Beginning 5-fold StratifiedKFold Cross-Validation on {len(X_train)} training samples...")
    print(f"Candidate models to evaluate: {list(models.keys())}\n")

    for model_key, model_instance in models.items():
        print(f"Evaluating {model_key}...")

        fold_accuracies: List[float] = []
        fold_precisions: List[float] = []
        fold_recalls: List[float] = []
        fold_f1s: List[float] = []
        fold_roc_aucs: List[float] = []
        fold_pr_aucs: List[float] = []

        total_fn = 0
        total_fp = 0
        total_tp = 0
        total_tn = 0

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_f_train = X_train.iloc[train_idx]
            y_f_train = y_train.iloc[train_idx]
            X_f_val = X_train.iloc[val_idx]
            y_f_val = y_train.iloc[val_idx]

            # Build a fresh, unfitted pipeline for this specific fold
            # Guarantees ZERO leakage between fold training and validation
            preprocessor = build_preprocessing_pipeline("approach_b")
            fold_model = clone(model_instance)

            fold_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", fold_model)
            ])

            # Fit strictly on fold training data
            fold_pipeline.fit(X_f_train, y_f_train)

            # Predict on fold validation data
            y_pred = fold_pipeline.predict(X_f_val)
            y_proba = fold_pipeline.predict_proba(X_f_val)[:, 1]

            # Calculate fold metrics
            acc = accuracy_score(y_f_val, y_pred)
            prec = precision_score(y_f_val, y_pred, zero_division=0)
            rec = recall_score(y_f_val, y_pred, zero_division=0)
            f1 = f1_score(y_f_val, y_pred, zero_division=0)
            roc = roc_auc_score(y_f_val, y_proba)
            pr = average_precision_score(y_f_val, y_proba)

            fold_accuracies.append(acc)
            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_f1s.append(f1)
            fold_roc_aucs.append(roc)
            fold_pr_aucs.append(pr)

            cm = confusion_matrix(y_f_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total_tn += int(tn)
            total_fp += int(fp)
            total_fn += int(fn)
            total_tp += int(tp)

        # Store aggregate results
        results[model_key] = {
            "model_name": model_key.replace("_", " "),
            "parameters": {k: str(v) for k, v in model_instance.get_params().items() if k in ["max_iter", "n_estimators", "max_depth", "learning_rate", "kernel", "probability", "random_state"]},
            "accuracy": {
                "mean": round(float(np.mean(fold_accuracies)), 4),
                "std": round(float(np.std(fold_accuracies)), 4),
                "per_fold": [round(float(x), 4) for x in fold_accuracies]
            },
            "precision": {
                "mean": round(float(np.mean(fold_precisions)), 4),
                "std": round(float(np.std(fold_precisions)), 4),
                "per_fold": [round(float(x), 4) for x in fold_precisions]
            },
            "recall_sensitivity": {
                "mean": round(float(np.mean(fold_recalls)), 4),
                "std": round(float(np.std(fold_recalls)), 4),
                "per_fold": [round(float(x), 4) for x in fold_recalls]
            },
            "f1_score": {
                "mean": round(float(np.mean(fold_f1s)), 4),
                "std": round(float(np.std(fold_f1s)), 4),
                "per_fold": [round(float(x), 4) for x in fold_f1s]
            },
            "roc_auc": {
                "mean": round(float(np.mean(fold_roc_aucs)), 4),
                "std": round(float(np.std(fold_roc_aucs)), 4),
                "per_fold": [round(float(x), 4) for x in fold_roc_aucs]
            },
            "pr_auc": {
                "mean": round(float(np.mean(fold_pr_aucs)), 4),
                "std": round(float(np.std(fold_pr_aucs)), 4),
                "per_fold": [round(float(x), 4) for x in fold_pr_aucs]
            },
            "screening_error_counts": {
                "total_false_negatives": total_fn,
                "total_false_positives": total_fp,
                "total_true_positives": total_tp,
                "total_true_negatives": total_tn,
                "total_positive_cases": total_fn + total_tp,
                "total_negative_cases": total_fp + total_tn
            },
            "completed_successfully": True
        }

        r = results[model_key]
        print(f"   Accuracy : {r['accuracy']['mean']:.4f} +/- {r['accuracy']['std']:.4f}")
        print(f"   Precision: {r['precision']['mean']:.4f} +/- {r['precision']['std']:.4f}")
        print(f"   Recall   : {r['recall_sensitivity']['mean']:.4f} +/- {r['recall_sensitivity']['std']:.4f}")
        print(f"   F1-Score : {r['f1_score']['mean']:.4f} +/- {r['f1_score']['std']:.4f}")
        print(f"   ROC-AUC  : {r['roc_auc']['mean']:.4f} +/- {r['roc_auc']['std']:.4f}")
        print(f"   PR-AUC   : {r['pr_auc']['mean']:.4f} +/- {r['pr_auc']['std']:.4f}")
        print(f"   Errors   : FN = {total_fn:2d}, FP = {total_fp:2d} (out of {len(X_train)} validation samples)\n")

    return results


def main():
    print("=" * 80)
    print("STEP 6: HEART DISEASE CANDIDATE MODEL CROSS-VALIDATION")
    print("=" * 80)

    # 1. Resolve dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    candidate_paths = [
        os.path.join(workspace_root, "datasets", "heart_disease", "heart.csv"),
        os.path.join("datasets", "heart_disease", "heart.csv"),
        os.path.abspath("datasets/heart_disease/heart.csv")
    ]
    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        resolved_path = candidate_paths[0]

    # 2. Reproduce exact Step 4 stratified 80/20 train/test split
    print(f"\n1. Loading and Partitioning Dataset (Step 4 protocol):")
    X_train, X_test, y_train, y_test, split_meta = create_train_test_split(
        dataset_path=resolved_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    print(f"   - Training Partition (X_train): {X_train.shape[0]} rows (USED FOR 5-FOLD CV)")
    print(f"   - Test Partition (X_test)     : {X_test.shape[0]} rows (STRICTLY QUARANTINED - UNTOUCHED)")

    # 3. Execute Candidate Model Evaluation
    cv_results = evaluate_candidate_models(
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        random_state=42
    )

    # 4. Save CV Results JSON
    output_json_path = os.path.join(script_dir, "candidate_cv_results.json")
    export_payload = {
        "dataset_path": "datasets/heart_disease/heart.csv",
        "split_configuration": {
            "test_size": 0.20,
            "train_rows": len(X_train),
            "test_rows_quarantined": len(X_test),
            "stratified": True,
            "random_state": 42
        },
        "cross_validation_configuration": {
            "strategy": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "preprocessing": "Approach B (ColumnTransformer: RobustScaler on continuous vitals, passthrough on binary/discrete) fitted strictly fold-by-fold"
        },
        "xgboost_status": {
            "available": XGBOOST_AVAILABLE,
            "version": XGBOOST_VERSION
        },
        "candidate_cv_results": cv_results,
        "quarantine_confirmation": {
            "test_set_used_during_cv": False,
            "test_rows_quarantined": 54
        }
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"Candidate CV results successfully exported to: {output_json_path}")

    # 5. Print Candidate Comparison Table
    print("\n" + "=" * 105)
    print(f"{'Model':<26} | {'Accuracy':<17} | {'Precision':<17} | {'Recall':<17} | {'F1-Score':<17} | {'ROC-AUC':<17} | {'PR-AUC':<17} | {'FN':<4} | {'FP':<4}")
    print("-" * 105)
    for model_key, r in cv_results.items():
        acc_str = f"{r['accuracy']['mean']:.4f} +/- {r['accuracy']['std']:.4f}"
        prec_str = f"{r['precision']['mean']:.4f} +/- {r['precision']['std']:.4f}"
        rec_str = f"{r['recall_sensitivity']['mean']:.4f} +/- {r['recall_sensitivity']['std']:.4f}"
        f1_str = f"{r['f1_score']['mean']:.4f} +/- {r['f1_score']['std']:.4f}"
        roc_str = f"{r['roc_auc']['mean']:.4f} +/- {r['roc_auc']['std']:.4f}"
        pr_str = f"{r['pr_auc']['mean']:.4f} +/- {r['pr_auc']['std']:.4f}"
        fn_val = r["screening_error_counts"]["total_false_negatives"]
        fp_val = r["screening_error_counts"]["total_false_positives"]
        print(f"{model_key:<26} | {acc_str:<17} | {prec_str:<17} | {rec_str:<17} | {f1_str:<17} | {roc_str:<17} | {pr_str:<17} | {fn_val:<4} | {fp_val:<4}")
    print("=" * 105)

    print("\n" + "=" * 80)
    print("STEP 6 CANDIDATE CV COMPLETED SUCCESSFULLY — NO WINNER SELECTED")
    print("=" * 80)


if __name__ == "__main__":
    main()
