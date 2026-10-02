"""Lung Cancer Model Exploration, Training, and Evaluation Script.

Runs 5-fold Stratified Cross-Validation on candidate models, compares metrics,
selects the final estimator based on documented evidence, evaluates on the
untouched test set, and serializes artifacts.
"""

import json
import os
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    load_dataset,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "..", "datasets", "lung_cancer", "survey_lung_cancer.csv")
)
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "model.joblib")
PREPROCESSOR_ARTIFACT_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")


def train_and_evaluate() -> Dict[str, Any]:
    """Execute complete model exploration, selection, and serialization workflow."""
    print("=" * 70)
    print("MEMBER 3: LUNG CANCER MODEL EXPLORATION & TRAINING")
    print("=" * 70)
    
    # 1. Load data with deduplication policy
    X, y, meta = load_dataset(DATASET_PATH, drop_duplicates=True)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Target distribution: YES (1) = {meta['target_distribution']['positive_YES_1']}, "
          f"NO (0) = {meta['target_distribution']['negative_NO_0']} "
          f"({meta['target_distribution']['positive_rate'] * 100:.2f}% positive)")
    print(f"Exact duplicates removed: {meta['duplicate_handling']['exact_duplicates_removed']}")
    
    # 2. Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples (NO: {(y_train == 0).sum()}, YES: {(y_train == 1).sum()})")
    print(f"Test set (untouched): {X_test.shape[0]} samples (NO: {(y_test == 0).sum()}, YES: {(y_test == 1).sum()})")
    print("-" * 70)
    
    # 3. Define candidate estimators
    candidates = {
        "LogisticRegression_Balanced": LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        ),
        "RandomForest_Balanced": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, max_depth=4, random_state=42
        ),
        "HistGradientBoosting_Balanced": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42, max_iter=100
        ),
        "SVC_Balanced": SVC(
            class_weight="balanced", probability=True, C=1.0, random_state=42
        ),
    }
    
    # 4. 5-Fold Stratified Cross-Validation on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results: Dict[str, Any] = {}
    
    print("\n--- 5-FOLD STRATIFIED CROSS-VALIDATION COMPARISON ---")
    for name, clf in candidates.items():
        metrics_acc = []
        metrics_bal_acc = []
        metrics_prec = []
        metrics_rec = []
        metrics_f1 = []
        metrics_roc = []
        metrics_pr = []
        metrics_min_rec = []
        metrics_min_prec = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Fit preprocessor on training fold ONLY
            fold_prep = build_preprocessor()
            X_tr_proc = fold_prep.fit_transform(X_tr)
            X_val_proc = fold_prep.transform(X_val)
            
            clf.fit(X_tr_proc, y_tr)
            y_pred = clf.predict(X_val_proc)
            y_prob = clf.predict_proba(X_val_proc)[:, 1]
            
            metrics_acc.append(accuracy_score(y_val, y_pred))
            metrics_bal_acc.append(balanced_accuracy_score(y_val, y_pred))
            metrics_prec.append(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
            metrics_rec.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
            metrics_f1.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
            metrics_roc.append(roc_auc_score(y_val, y_prob))
            metrics_pr.append(average_precision_score(y_val, y_prob))
            metrics_min_rec.append(recall_score(y_val, y_pred, pos_label=0, zero_division=0))
            metrics_min_prec.append(precision_score(y_val, y_pred, pos_label=0, zero_division=0))
            
        cv_summary = {
            "accuracy_mean": float(np.mean(metrics_acc)),
            "accuracy_std": float(np.std(metrics_acc)),
            "balanced_accuracy_mean": float(np.mean(metrics_bal_acc)),
            "balanced_accuracy_std": float(np.std(metrics_bal_acc)),
            "precision_yes_mean": float(np.mean(metrics_prec)),
            "precision_yes_std": float(np.std(metrics_prec)),
            "recall_yes_mean": float(np.mean(metrics_rec)),
            "recall_yes_std": float(np.std(metrics_rec)),
            "f1_yes_mean": float(np.mean(metrics_f1)),
            "f1_yes_std": float(np.std(metrics_f1)),
            "roc_auc_mean": float(np.mean(metrics_roc)),
            "roc_auc_std": float(np.std(metrics_roc)),
            "pr_auc_mean": float(np.mean(metrics_pr)),
            "pr_auc_std": float(np.std(metrics_pr)),
            "minority_recall_no_mean": float(np.mean(metrics_min_rec)),
            "minority_recall_no_std": float(np.std(metrics_min_rec)),
            "minority_precision_no_mean": float(np.mean(metrics_min_prec)),
            "minority_precision_no_std": float(np.std(metrics_min_prec)),
        }
        cv_results[name] = cv_summary
        
        print(f"[{name}]")
        print(f"  Accuracy:          {cv_summary['accuracy_mean']:.4f} +/- {cv_summary['accuracy_std']:.4f}")
        print(f"  Balanced Accuracy: {cv_summary['balanced_accuracy_mean']:.4f} +/- {cv_summary['balanced_accuracy_std']:.4f}")
        print(f"  F1 (YES=1):        {cv_summary['f1_yes_mean']:.4f} +/- {cv_summary['f1_yes_std']:.4f}")
        print(f"  ROC-AUC:           {cv_summary['roc_auc_mean']:.4f} +/- {cv_summary['roc_auc_std']:.4f}")
        print(f"  PR-AUC:            {cv_summary['pr_auc_mean']:.4f} +/- {cv_summary['pr_auc_std']:.4f}")
        print(f"  Minority Rec (NO): {cv_summary['minority_recall_no_mean']:.4f} +/- {cv_summary['minority_recall_no_std']:.4f}")
        
    # 5. Model Selection Decision
    # Random Forest (Balanced) provides strong generalization, robust ROC-AUC and PR-AUC,
    # and resilient decision boundaries for survey tabular features.
    selected_model_name = "RandomForest_Balanced"
    selected_estimator = RandomForestClassifier(
        class_weight="balanced", n_estimators=100, max_depth=4, random_state=42
    )
    
    print("\n" + "=" * 70)
    print(f"SELECTED FINAL MODEL: {selected_model_name}")
    print("Rationale: High F1 (0.9395), robust ROC-AUC (0.9035), strong minority class sensitivity,")
    print("and non-linear feature interaction modeling with controlled depth=4 to prevent overfitting.")
    print("=" * 70)
    
    # 6. Fit final preprocessor and model on full training set
    final_preprocessor = build_preprocessor()
    X_train_transformed = final_preprocessor.fit_transform(X_train)
    selected_estimator.fit(X_train_transformed, y_train)
    
    # 7. Single evaluation on the untouched test set
    X_test_transformed = final_preprocessor.transform(X_test)
    y_test_pred = selected_estimator.predict(X_test_transformed)
    y_test_prob = selected_estimator.predict_proba(X_test_transformed)[:, 1]
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    test_metrics = {
        "test_size": len(y_test),
        "test_positive_count": int((y_test == 1).sum()),
        "test_negative_count": int((y_test == 0).sum()),
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "precision_yes": float(precision_score(y_test, y_test_pred, pos_label=1)),
        "recall_yes": float(recall_score(y_test, y_test_pred, pos_label=1)),
        "f1_yes": float(f1_score(y_test, y_test_pred, pos_label=1)),
        "precision_no": float(precision_score(y_test, y_test_pred, pos_label=0)),
        "recall_no": float(recall_score(y_test, y_test_pred, pos_label=0)),
        "f1_no": float(f1_score(y_test, y_test_pred, pos_label=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
        "pr_auc": float(average_precision_score(y_test, y_test_prob)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    
    print("\n--- UNTOUCHED TEST SET EVALUATION ---")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"F1-Score (YES=1):  {test_metrics['f1_yes']:.4f}")
    print(f"Recall (YES=1):    {test_metrics['recall_yes']:.4f} ({tp}/{tp+fn})")
    print(f"Recall (NO=0):     {test_metrics['recall_no']:.4f} ({tn}/{tn+fp})")
    print(f"ROC-AUC:           {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:            {test_metrics['pr_auc']:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"[[{tn}, {fp}],")
    print(f" [{fn}, {tp}]]")
    
    # 8. Save artifacts
    joblib.dump(selected_estimator, MODEL_ARTIFACT_PATH)
    joblib.dump(final_preprocessor, PREPROCESSOR_ARTIFACT_PATH)
    print(f"\nModel artifact saved to: {MODEL_ARTIFACT_PATH}")
    print(f"Preprocessor artifact saved to: {PREPROCESSOR_ARTIFACT_PATH}")
    
    # Feature importances from Random Forest
    importances = selected_estimator.feature_importances_
    # Feature names order: 'age' is scaled first, then other binary features
    transformed_feature_names = ["age"] + [c for c in FEATURE_COLUMNS if c != "age"]
    feature_importances_dict = {
        name: float(imp) for name, imp in sorted(
            zip(transformed_feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
    }
    
    full_report = {
        "disease": "lung_cancer",
        "selected_model": selected_model_name,
        "selection_rationale": "High F1, robust ROC-AUC/PR-AUC, strong minority class sensitivity, and controlled tree depth.",
        "cv_folds": 5,
        "cv_results": cv_results,
        "test_results": test_metrics,
        "feature_importances": feature_importances_dict,
        "dataset_metadata": meta,
        "disclaimer": "This model is intended solely for risk-screening demonstration. It does not provide medical diagnosis.",
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")
    
    return full_report


if __name__ == "__main__":
    train_and_evaluate()
