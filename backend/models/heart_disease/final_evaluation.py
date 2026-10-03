"""
Final Model Selection and Test Evaluation Script
Heart Disease Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 7 — Final Model Selection & Test Evaluation
Tasks:
1. Recreates the exact 80/20 train/test split (216 train, 54 test, random_state=42, stratify=y).
2. Documents the technical selection of Random Forest based strictly on Step 6 CV evidence.
3. Constructs the unified production scikit-learn Pipeline (ColumnTransformer Approach B -> RandomForestClassifier).
4. Fits the complete pipeline exclusively on the 216 training samples.
5. Evaluates exactly once on the quarantined 54 test samples.
6. Serializes the complete trained pipeline as 'model.joblib'.
7. Exports comprehensive evaluation metrics to 'metrics.json'.
8. Exports the technical model selection report to 'model_selection.json'.
9. Verifies the saved joblib artifact can be reloaded and perform valid inference matching the in-memory model.
"""

import os
import sys
import json
import warnings
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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
    from preprocessing import build_preprocessing_pipeline, FEATURE_COLUMNS
except ImportError:
    from backend.models.heart_disease.define_features import EXPECTED_PREDICTOR_COLUMNS, TARGET_COLUMN
    from backend.models.heart_disease.split_data import create_train_test_split
    from backend.models.heart_disease.preprocessing import build_preprocessing_pipeline, FEATURE_COLUMNS


def build_final_pipeline() -> Pipeline:
    """
    Constructs the unified production Pipeline combining preprocessing and the selected classifier.
    
    Architecture:
    1. 'preprocessor': ColumnTransformer (Approach B)
       - Continuous vitals (Age, BP, Cholesterol, Max HR, ST depression): SimpleImputer(median) -> RobustScaler()
       - Binary indicators (Sex, FBS over 120, Exercise angina): passthrough
       - Discrete integer codes (Chest pain type, EKG results, Slope of ST, Number of vessels fluro, Thallium): passthrough
    2. 'classifier': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    """
    pipeline = Pipeline([
        ("preprocessor", build_preprocessing_pipeline("approach_b")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline


def run_final_evaluation(dataset_path: str, output_dir: str):
    print("=" * 80)
    print("STEP 7: FINAL MODEL SELECTION & TEST EVALUATION - HEART DISEASE MODEL")
    print("=" * 80)

    abs_dataset_path = os.path.abspath(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dataset location: {abs_dataset_path}")

    # 1. Load Step 6 Cross-Validation Results
    cv_results_path = os.path.join(output_dir, "candidate_cv_results.json")
    if not os.path.exists(cv_results_path):
        raise FileNotFoundError(
            f"Step 6 CV results not found at: {cv_results_path}. Run candidate_cv.py first."
        )

    with open(cv_results_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)

    # 2. Document Selection Rationale based strictly on Step 6 CV results
    print("\n1. Step 6 Cross-Validation Performance Summary (X_train only, 216 samples):")
    cv_models = cv_data["candidate_cv_results"]
    for m_name, m_metrics in cv_models.items():
        acc = m_metrics["accuracy"]["mean"]
        f1 = m_metrics["f1_score"]["mean"]
        roc = m_metrics["roc_auc"]["mean"]
        rec = m_metrics["recall_sensitivity"]["mean"]
        prec = m_metrics["precision"]["mean"]
        fn = m_metrics["screening_error_counts"]["total_false_negatives"]
        fp = m_metrics["screening_error_counts"]["total_false_positives"]
        print(f"   - {m_name:<26}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc:.4f}, FN={fn:2d}, FP={fp:2d}")

    selected_model_name = "Random_Forest"
    print(f"\n2. Selected Model: {selected_model_name}")
    print("   Selection Basis: Empirical 5-fold cross-validation evidence from Step 6.")
    selection_rationale = (
        "Random Forest was selected based strictly on Step 6 5-fold cross-validation evidence on the 216-row training partition. "
        "It achieved the highest mean Accuracy (0.8332 +/- 0.0478), highest Precision (0.8410 +/- 0.0713), "
        "highest F1-Score (0.8035 +/- 0.0670), highest ROC-AUC (0.9064 +/- 0.0378), and highest PR-AUC (0.9078 +/- 0.0378) "
        "among all evaluated candidate models. It also achieved the lowest cumulative False Positives (15). "
        "While Gradient Boosting yielded slightly higher recall (0.8126 vs 0.7811, representing 18 vs 21 false negatives across "
        "216 validation samples), it incurred nearly double the false positives (29 vs 15) and substantially lower ROC-AUC (0.8816 vs 0.9064) "
        "and F1-score (0.7683 vs 0.8035). Logistic Regression demonstrated competitive metrics (ROC-AUC 0.9007, 20 FNs, 19 FPs) "
        "but lower accuracy and F1 compared to Random Forest."
    )
    print(f"   Rationale: {selection_rationale}")

    # 3. Create Partition Split (Reproduces Step 4 exactly)
    print("\n3. Loading Dataset & Partitioning (Step 4 protocol):")
    X_train, X_test, y_train, y_test, split_meta = create_train_test_split(
        dataset_path=abs_dataset_path,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    print(f"   - X_train shape: {X_train.shape} (216 samples)")
    print(f"   - X_test shape : {X_test.shape} (54 samples - QUARANTINED UNTIL NOW)")

    # 4. Construct and Fit Production Pipeline on Training Set Only
    print("\n4. Fitting Full Production Pipeline on Training Set (216 samples) exclusively...")
    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train)
    print("   - Pipeline successfully trained on X_train.")

    # 5. Evaluate Exactly Once on Quarantined Test Set
    print("\n5. Evaluating Exactly Once on Quarantined Test Set (54 samples)...")
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate Test Metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    print(f"\n--- Final Test Set Evaluation Results (N = 54) ---")
    print(f"   Accuracy            : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"   Precision           : {test_prec:.4f} ({test_prec * 100:.2f}%)")
    print(f"   Recall (Sensitivity): {test_rec:.4f} ({test_rec * 100:.2f}%)")
    print(f"   F1-Score            : {test_f1:.4f} ({test_f1 * 100:.2f}%)")
    print(f"   ROC-AUC             : {test_roc_auc:.4f}")
    print(f"   PR-AUC              : {test_pr_auc:.4f}")
    print(f"\n--- Confusion Matrix (Labels: 0=Absence, 1=Presence) ---")
    print(f"   True Negatives  (TN): {tn:2d} (correctly predicted Absence)")
    print(f"   False Positives (FP): {fp:2d} (actual Absence, predicted Presence)")
    print(f"   False Negatives (FN): {fn:2d} (actual Presence, missed by model)")
    print(f"   True Positives  (TP): {tp:2d} (correctly predicted Presence)")
    print(f"   False Positive Rate : {fpr:.4f} ({fpr * 100:.2f}%)")
    print(f"   False Negative Rate : {fnr:.4f} ({fnr * 100:.2f}%)")

    # 6. Save Complete Pipeline Artifact (model.joblib)
    model_save_path = os.path.join(output_dir, "model.joblib")
    print(f"\n6. Serializing complete pipeline to: {model_save_path}")
    joblib.dump(pipeline, model_save_path)
    model_size_bytes = os.path.getsize(model_save_path)
    print(f"   - model.joblib successfully written ({model_size_bytes} bytes, {model_size_bytes / 1024:.2f} KB).")

    # 7. Reload and Verify Saved Artifact
    print("\n7. Reloading model.joblib to verify serialization integrity...")
    reloaded_pipeline = joblib.load(model_save_path)
    reloaded_pred = reloaded_pipeline.predict(X_test)
    reloaded_proba = reloaded_pipeline.predict_proba(X_test)[:, 1]

    assert np.array_equal(y_test_pred, reloaded_pred), "Reloaded model predictions do not match in-memory model!"
    assert np.allclose(y_test_proba, reloaded_proba), "Reloaded model probabilities do not match in-memory model!"
    assert np.all(np.isfinite(reloaded_proba)), "Non-finite values found in reloaded probabilities!"
    assert np.all((reloaded_proba >= 0.0) & (reloaded_proba <= 1.0)), "Probabilities out of bounds [0, 1]!"
    print("   - Reload verification PASSED: Reloaded model produces identical predictions and valid probabilities.")

    # 8. Export metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    print(f"\n8. Exporting final metrics to: {metrics_path}")

    actual_counts = pd.Series(y_test).value_counts().to_dict()
    pred_counts = pd.Series(y_test_pred).value_counts().to_dict()

    metrics_payload = {
        "model_domain": "heart_disease",
        "dataset_path": "datasets/heart_disease/heart.csv",
        "random_state": 42,
        "test_size": 0.20,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_variable": TARGET_COLUMN,
        "target_classes": {
            "0": "Absence",
            "1": "Presence"
        },
        "selected_model": selected_model_name,
        "selected_model_parameters": {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1
        },
        "cross_validation_metrics_step_6": {
            "accuracy": cv_models[selected_model_name]["accuracy"],
            "precision": cv_models[selected_model_name]["precision"],
            "recall_sensitivity": cv_models[selected_model_name]["recall_sensitivity"],
            "f1_score": cv_models[selected_model_name]["f1_score"],
            "roc_auc": cv_models[selected_model_name]["roc_auc"],
            "pr_auc": cv_models[selected_model_name]["pr_auc"],
            "screening_error_counts": cv_models[selected_model_name]["screening_error_counts"]
        },
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall_sensitivity": round(float(test_rec), 4),
            "f1_score": round(float(test_f1), 4),
            "roc_auc": round(float(test_roc_auc), 4),
            "pr_auc": round(float(test_pr_auc), 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "actual_class_distribution": {
                "class_0": int(actual_counts.get(0, 0)),
                "class_1": int(actual_counts.get(1, 0))
            },
            "predicted_class_distribution": {
                "class_0": int(pred_counts.get(0, 0)),
                "class_1": int(pred_counts.get(1, 0))
            }
        },
        "disclaimer": "This model provides an AI-based risk screening estimate and does not constitute a definitive medical diagnosis."
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print("   - metrics.json successfully written.")

    # 9. Export model_selection.json
    selection_path = os.path.join(output_dir, "model_selection.json")
    print(f"\n9. Exporting model selection report to: {selection_path}")

    # Build concise comparison dictionary for model_selection.json
    candidate_summary = {}
    for k, v in cv_models.items():
        candidate_summary[k] = {
            "accuracy": f"{v['accuracy']['mean']:.4f} +/- {v['accuracy']['std']:.4f}",
            "precision": f"{v['precision']['mean']:.4f} +/- {v['precision']['std']:.4f}",
            "recall_sensitivity": f"{v['recall_sensitivity']['mean']:.4f} +/- {v['recall_sensitivity']['std']:.4f}",
            "f1_score": f"{v['f1_score']['mean']:.4f} +/- {v['f1_score']['std']:.4f}",
            "roc_auc": f"{v['roc_auc']['mean']:.4f} +/- {v['roc_auc']['std']:.4f}",
            "false_negatives": v["screening_error_counts"]["total_false_negatives"],
            "false_positives": v["screening_error_counts"]["total_false_positives"]
        }

    selection_payload = {
        "model_domain": "heart_disease",
        "dataset_path": "datasets/heart_disease/heart.csv",
        "evaluation_protocol": "5-fold StratifiedKFold on X_train (216 rows); final test set (54 rows) strictly quarantined",
        "candidate_models_considered": list(cv_models.keys()),
        "step_6_cross_validation_metrics": candidate_summary,
        "selected_model": selected_model_name,
        "selected_model_parameters": {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1
        },
        "selection_basis": "Empirical 5-fold cross-validation evidence from Step 6",
        "technical_selection_rationale": selection_rationale,
        "quarantine_confirmation": {
            "test_set_used_during_selection": False,
            "test_rows_quarantined": 54,
            "final_training_rows": 216
        }
    }

    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)
    print("   - model_selection.json successfully written.")

    print("\n" + "=" * 80)
    print("STEP 7 FINAL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
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

    run_final_evaluation(resolved_path, script_dir)
