"""
Candidate Model Cross-Validation and Comparison Script
Stroke Prediction Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 6 — Candidate Model Comparison
Evaluates 5 candidate classifiers using 5-fold StratifiedKFold cross-validation
exclusively on X_train (4,088 samples). The 1,022-sample test set is strictly quarantined.
Preprocessing (RobustScaler continuous + passthrough binary + OneHotEncoder categorical)
is constructed and fitted strictly inside each fold to guarantee zero data leakage.
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

warnings.filterwarnings("ignore")

# Ensure local directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from split_data import create_train_test_split, compute_sha256
    from preprocessing import build_stroke_preprocessor, FEATURE_COLUMNS
except ImportError:
    from backend.models.stroke.define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from backend.models.stroke.split_data import create_train_test_split, compute_sha256
    from backend.models.stroke.preprocessing import build_stroke_preprocessor, FEATURE_COLUMNS

# Check XGBoost availability
try:
    from xgboost import XGBClassifier
    import xgboost
    XGBOOST_AVAILABLE = True
    XGBOOST_VERSION = xgboost.__version__
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = "not installed"

EXPECTED_DATASET_HASH = "aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815"


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
    No test data is used.
    """
    models = get_candidate_models()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: Dict[str, Any] = {}

    print(f"Beginning 5-fold StratifiedKFold Cross-Validation on {len(X_train)} training samples...")
    print(f"Candidate models to evaluate: {list(models.keys())}\n")

    # Track overall validation row coverage across folds
    visited_val_indices = set()

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
            # Disjointness check within fold
            assert len(set(train_idx).intersection(set(val_idx))) == 0, f"Fold {fold_idx} train/val overlap!"
            if model_key == list(models.keys())[0]:
                visited_val_indices.update(val_idx)

            X_f_train = X_train.iloc[train_idx]
            y_f_train = y_train.iloc[train_idx]
            X_f_val = X_train.iloc[val_idx]
            y_f_val = y_train.iloc[val_idx]

            # Build a fresh, unfitted pipeline for this specific fold
            # Guarantees ZERO preprocessing leakage between fold training and validation
            preprocessor = build_stroke_preprocessor(scaler_type="robust")
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

            # Verify probabilities are valid and finite
            assert bool(np.all(np.isfinite(y_proba))), f"Non-finite probabilities in {model_key} fold {fold_idx}!"
            assert bool(np.all((y_proba >= 0.0) & (y_proba <= 1.0))), f"Probabilities outside [0, 1] in {model_key} fold {fold_idx}!"

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
            "parameters": {k: str(v) for k, v in model_instance.get_params().items() if k in [
                "max_iter", "n_estimators", "max_depth", "learning_rate", "kernel", "probability", "random_state", "eval_metric", "n_jobs"
            ]},
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
        print(f"   OOF Errors: FN = {total_fn:3d}, FP = {total_fp:3d} (out of {len(X_train)} validation samples)\n")

    # Verify every training index was visited as validation exactly once
    assert len(visited_val_indices) == len(X_train), "Not all training indices were used in validation!"

    return results


def main():
    print("=" * 80)
    print("STEP 6: STROKE DISEASE CANDIDATE MODEL CROSS-VALIDATION")
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
    print(f"\n1. Split Configuration & Quarantine Check:")
    print(f"   - Dataset Path       : {resolved_path}")
    print(f"   - Dataset SHA-256    : {pre_sha256}")

    # 2. Extract train/test split (Step 4 protocol)
    X_train, X_test, y_train, y_test, split_meta = create_train_test_split(
        dataset_path=resolved_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    assert len(X_train) == 4088, f"Expected 4088 train samples, got {len(X_train)}"
    assert len(X_test) == 1022, f"Expected 1022 test samples, got {len(X_test)}"
    print(f"   - Training partition : {X_train.shape[0]} rows x {X_train.shape[1]} features (USED FOR CV)")
    print(f"   - Test partition     : {X_test.shape[0]} rows x {X_test.shape[1]} features (STRICTLY QUARANTINED)")
    print("   - Test data quarantine confirmation: ACTIVE. Zero test records used for CV or model tuning.")

    # 3. Class Imbalance Analysis of Training Set
    train_counts = y_train.value_counts().to_dict()
    c0 = train_counts.get(0, 0)
    c1 = train_counts.get(1, 0)
    imbalance_ratio = c0 / c1 if c1 > 0 else 0
    print(f"\n2. Class Imbalance Analysis (X_train):")
    print(f"   - Class 0 (No Stroke): {c0:5d} ({c0/len(y_train)*100:.2f}%)")
    print(f"   - Class 1 (Stroke)   : {c1:5d} ({c1/len(y_train)*100:.2f}%)")
    print(f"   - Imbalance Ratio    : {imbalance_ratio:.2f}:1")
    print("   - Technical Implication: Accuracy alone is an inadequate evaluation metric.")
    print("     A trivial majority classifier predicting 0 for all instances achieves 95.13% accuracy")
    print("     with zero recall. ROC-AUC and PR-AUC are the primary rank-discrimination indicators.")

    # 4. Execute 5-Fold Stratified Cross-Validation
    print("\n3. Executing 5-Fold Stratified Cross-Validation on X_train...")
    cv_results = evaluate_candidate_models(X_train, y_train, n_splits=5, random_state=42)

    # 5. Export Results to candidate_cv_results.json
    output_json_path = os.path.join(script_dir, "candidate_cv_results.json")
    print(f"\n4. Exporting Cross-Validation Results to: {output_json_path}")

    full_results_payload = {
        "model_domain": "stroke",
        "dataset_path": "datasets/stroke/stroke.csv",
        "dataset_sha256": pre_sha256,
        "split_configuration": {
            "test_size": 0.20,
            "random_state": 42,
            "stratified": True,
            "train_rows": len(X_train),
            "quarantined_test_rows": len(X_test),
            "train_class_distribution": {str(k): int(v) for k, v in train_counts.items()},
            "imbalance_ratio": round(imbalance_ratio, 4)
        },
        "cross_validation_configuration": {
            "n_splits": 5,
            "strategy": "StratifiedKFold",
            "shuffle": True,
            "random_state": 42,
            "preprocessing_fit_scope": "fold_local_training_only",
            "leakage_prevention": "ColumnTransformer pipeline reconstructed and fitted independently inside every CV fold"
        },
        "xgboost_availability": {
            "installed": XGBOOST_AVAILABLE,
            "version": XGBOOST_VERSION
        },
        "candidate_cv_results": cv_results,
        "class_imbalance_analysis": {
            "majority_class_percentage": round(float((c0 / len(y_train)) * 100), 4),
            "minority_class_percentage": round(float((c1 / len(y_train)) * 100), 4),
            "accuracy_insufficiency_statement": (
                "Due to the 19.54:1 class imbalance in the training partition, accuracy of ~95% reflects "
                "the majority class baseline rather than discriminative capability. At the default 0.5 decision "
                "threshold, unweighted classifiers detect few positive cases (low recall/F1). Therefore, ROC-AUC "
                "and PR-AUC provide the primary threshold-independent assessment of risk ranking."
            )
        },
        "quarantine_confirmation": {
            "test_set_quarantined": True,
            "test_rows_quarantined": len(X_test),
            "test_set_used_in_step_6": False,
            "final_model_selected": False,
            "notes": "No final model selection performed in Step 6. Test set remains untouched for Step 7."
        }
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results_payload, f, indent=2)
    print("   [x] candidate_cv_results.json successfully saved.")

    # 6. Post-execution dataset integrity check
    post_sha256 = compute_sha256(resolved_path)
    hash_match = (pre_sha256 == post_sha256 == EXPECTED_DATASET_HASH)
    print(f"\n5. Dataset Integrity Verification:")
    print(f"   - Post-Execution Dataset SHA-256: {post_sha256}")
    print(f"   - Read-Only Integrity Verified : {hash_match}")
    if not hash_match:
        raise RuntimeError("CRITICAL ERROR: Dataset file was modified during Step 6 execution!")

    print("\n" + "=" * 80)
    print("STEP 6 CANDIDATE MODEL CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
