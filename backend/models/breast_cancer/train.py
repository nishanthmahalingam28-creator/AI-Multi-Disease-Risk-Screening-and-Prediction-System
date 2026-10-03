"""Candidate Model Definition & Cross-Validation Exploration Script.
Breast Cancer Model - Member 1
AI Multi-Disease Risk Screening and Prediction System

Evaluates candidate binary classification architectures via 5-Fold Stratified
Cross-Validation strictly on the training partition (N_train = 455).
The test partition (N_test = 114) is completely quarantined for final evaluation.
"""

import json
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_ENCODING,
    build_preprocessor,
    load_dataset,
    prepare_data,
)


def get_candidate_models() -> Dict[str, Any]:
    """Defines and returns candidate binary classification models with baseline configurations."""
    models: Dict[str, Any] = {
        "Logistic_Regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
        ),
        "Support_Vector_Classifier": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=42,
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            eval_metric="logloss",
            random_state=42,
        ),
    }
    return models


def evaluate_candidates_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Executes 5-fold Stratified Cross-Validation on X_train only.

    Guarantees:
    - Preprocessing is fitted strictly on X_train fold subsets (zero fold leakage).
    - Computes mean and standard deviation for Accuracy, Precision, Recall, F1, ROC-AUC.
    - Tracks False Negative counts across folds for clinical screening risk transparency.
    """
    candidates = get_candidate_models()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results: Dict[str, Any] = {}

    total_malignant_in_train = int((y_train == 1).sum())

    for model_name, model in candidates.items():
        fold_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
            "false_negatives": [],
            "false_positives": [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit preprocessor strictly on the training fold subset
            preprocessor = build_preprocessor()
            X_tr_proc = preprocessor.fit_transform(X_tr)
            X_val_proc = preprocessor.transform(X_val)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_proc, y_tr)

            y_pred = model.predict(X_val_proc)
            y_prob = model.predict_proba(X_val_proc)[:, 1]

            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
            fold_metrics["false_negatives"].append(int(fn))
            fold_metrics["false_positives"].append(int(fp))

        cv_results[model_name] = {
            "accuracy": {
                "mean": round(float(np.mean(fold_metrics["accuracy"])), 4),
                "std": round(float(np.std(fold_metrics["accuracy"])), 4),
            },
            "precision": {
                "mean": round(float(np.mean(fold_metrics["precision"])), 4),
                "std": round(float(np.std(fold_metrics["precision"])), 4),
            },
            "recall_sensitivity": {
                "mean": round(float(np.mean(fold_metrics["recall"])), 4),
                "std": round(float(np.std(fold_metrics["recall"])), 4),
            },
            "f1_score": {
                "mean": round(float(np.mean(fold_metrics["f1"])), 4),
                "std": round(float(np.std(fold_metrics["f1"])), 4),
            },
            "roc_auc": {
                "mean": round(float(np.mean(fold_metrics["roc_auc"])), 4),
                "std": round(float(np.std(fold_metrics["roc_auc"])), 4),
            },
            "screening_error_counts": {
                "total_false_negatives": int(sum(fold_metrics["false_negatives"])),
                "total_false_positives": int(sum(fold_metrics["false_positives"])),
                "total_malignant_cases": total_malignant_in_train,
            },
        }

    return cv_results


def run_candidate_exploration(csv_path: str, output_json_path: str):
    print("=" * 80)
    print("STEP 6: CANDIDATE MODEL EXPLORATION & STRATIFIED CROSS-VALIDATION")
    print("=" * 80)

    # 1. Load data and create split
    print(f"Loading dataset from: {os.path.abspath(csv_path)}")
    X_train, X_test, y_train, y_test, _ = prepare_data(
        csv_path, test_size=0.20, random_state=42, stratify=True
    )

    print(f"Training partition (used for CV): {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test partition (quarantined/unseen): {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Target distribution in training: Benign (0) = {(y_train == 0).sum()}, Malignant (1) = {(y_train == 1).sum()}")

    # 2. Run 5-fold Stratified Cross-Validation on training data
    print("\nExecuting 5-Fold Stratified Cross-Validation on X_train...")
    cv_results = evaluate_candidates_cv(X_train, y_train, n_splits=5, random_state=42)

    # 3. Print formatted comparison table
    print("\n" + "-" * 80)
    print(f"{'Model Name':<28} | {'Accuracy':<15} | {'Recall (Sens)':<15} | {'ROC-AUC':<15} | {'FN Count':<8}")
    print("-" * 80)
    for model_name, metrics in cv_results.items():
        acc = f"{metrics['accuracy']['mean']:.4f} +/- {metrics['accuracy']['std']:.4f}"
        rec = f"{metrics['recall_sensitivity']['mean']:.4f} +/- {metrics['recall_sensitivity']['std']:.4f}"
        roc = f"{metrics['roc_auc']['mean']:.4f} +/- {metrics['roc_auc']['std']:.4f}"
        fn = f"{metrics['screening_error_counts']['total_false_negatives']}/{metrics['screening_error_counts']['total_malignant_cases']}"
        print(f"{model_name:<28} | {acc:<15} | {rec:<15} | {roc:<15} | {fn:<8}")
    print("-" * 80)

    # 4. Save CV comparison results for auditability
    output_data = {
        "dataset": os.path.abspath(csv_path),
        "split_configuration": {
            "test_size": 0.20,
            "train_samples": len(X_train),
            "test_samples_quarantined": len(X_test),
            "stratified": True,
            "random_state": 42,
        },
        "cross_validation_configuration": {
            "strategy": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "preprocessing": "StandardScaler fitted fold-by-fold strictly on training subsets",
        },
        "candidate_cv_results": cv_results,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved cross-validation audit results to: {output_json_path}")

    print("\n" + "=" * 80)
    print("BOUNDARY CONFIRMATION:")
    print("  * X_test WAS NOT ACCESSED OR EVALUATED (COMPLETELY QUARANTINED).")
    print("  * NO FINAL MODEL SELECTED.")
    print("  * NO FINAL MODEL ARTIFACT (model.joblib) SAVED.")
    print("=" * 80)


if __name__ == "__main__":
    default_dataset = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "datasets", "breast_cancer", "breast.csv"
    )
    if not os.path.exists(default_dataset):
        default_dataset = os.path.join("datasets", "breast_cancer", "breast.csv")

    output_results_file = os.path.join(current_dir, "candidate_cv_results.json")
    run_candidate_exploration(default_dataset, output_results_file)
