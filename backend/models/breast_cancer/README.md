# Breast Cancer Risk-Screening Model
**Member 1: AI Multi-Disease Risk Screening and Prediction System**

---

## 1. Breast Cancer Model Overview

* **Purpose:** This disease-specific machine learning component provides automated, reproducible risk screening estimates for breast cancer using digitized Fine Needle Aspirate (FNA) cell nucleus characteristics.
* **Intended Use:** The model produces an **AI screening estimate** (`Class 0`: Low Risk / Benign, `Class 1`: High Risk / Malignant) accompanied by continuous class probability scores to assist in triage and prioritization.
* **Important Medical Limitation:** **This model does not provide a confirmed medical diagnosis or diagnostic determination.** It is an automated risk-screening research and software component. Clinical diagnosis of breast lesions requires comprehensive histopathology, biopsy imaging, and direct evaluation by licensed medical professionals.

---

## 2. Dataset Specification

* **Dataset Path:** [`datasets/breast_cancer/breast.csv`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/datasets/breast_cancer/breast.csv)
* **Dataset Nature:** Real-world benchmark (Wisconsin Diagnostic Breast Cancer, WDBC) derived from digitized microscopic images of fine needle aspirates of breast masses.
* **Total Sample Count:** `569` patient rows
* **Total Raw Columns:** `33` columns
* **Target Column:** `diagnosis`
* **Target Encoding:**
  * `0`: Benign (`B`, Negative Screening Cohort) — `357` records (62.74%)
  * `1`: Malignant (`M`, Positive Screening Cohort) — `212` records (37.26%)
* **Number of Model Features:** `30` continuous numerical features
* **Excluded Columns (2):**
  1. `id`: Non-clinical arbitrary database identifier. Quarantined to prevent index memorization and spurious correlation.
  2. `Unnamed: 32`: CSV parser artifact caused by a trailing comma on the header line. Contains 100% missing values (`NaN`).
* **Missing-Value Findings:** Exactly **0** missing values across all 30 model features and the target.
* **Duplicate Findings:** Exactly **0** duplicate sample rows.

---

## 3. Feature Schema (30 Predictors)

All 30 predictor features are continuous real-valued measurements (`float64`) describing cell nucleus morphology. The feature order is strictly maintained in the canonical schema:

### Mean Measurements (10 Features):
1. `radius_mean`: Mean distance from center to points on the contour (range: 6.981 – 28.11)
2. `texture_mean`: Standard deviation of gray-scale values (range: 9.71 – 39.28)
3. `perimeter_mean`: Mean core tumor perimeter (range: 43.79 – 188.5)
4. `area_mean`: Mean nuclear area (range: 143.5 – 2501.0)
5. `smoothness_mean`: Local variation in radius lengths (range: 0.0526 – 0.1634)
6. `compactness_mean`: $(\text{perimeter}^2 / \text{area} - 1.0)$ (range: 0.0194 – 0.3454)
7. `concavity_mean`: Severity of concave contour portions (range: 0.0 – 0.4268)
8. `concave points_mean`: Number of concave contour portions (range: 0.0 – 0.2012)
9. `symmetry_mean`: Nuclear symmetry score (range: 0.1060 – 0.3040)
10. `fractal_dimension_mean`: "Coastline approximation" metric (range: 0.0500 – 0.0974)

### Standard Error (SE) Measurements (10 Features):
11. `radius_se`: Standard error for radius (range: 0.1115 – 2.8730)
12. `texture_se`: Standard error for texture (range: 0.3602 – 4.8850)
13. `perimeter_se`: Standard error for perimeter (range: 0.7570 – 21.98)
14. `area_se`: Standard error for area (range: 6.802 – 542.2)
15. `smoothness_se`: Standard error for smoothness (range: 0.0017 – 0.0311)
16. `compactness_se`: Standard error for compactness (range: 0.0023 – 0.1354)
17. `concavity_se`: Standard error for concavity (range: 0.0 – 0.3960)
18. `concave points_se`: Standard error for concave points (range: 0.0 – 0.0528)
19. `symmetry_se`: Standard error for symmetry (range: 0.0079 – 0.0790)
20. `fractal_dimension_se`: Standard error for fractal dimension (range: 0.0009 – 0.0298)

### Worst / Extreme Measurements (10 Features):
21. `radius_worst`: Mean of the 3 largest values for radius (range: 7.93 – 36.04)
22. `texture_worst`: Mean of the 3 largest values for texture (range: 12.02 – 49.54)
23. `perimeter_worst`: Mean of the 3 largest values for perimeter (range: 50.41 – 251.2)
24. `area_worst`: Mean of the 3 largest values for area (range: 185.2 – 4254.0)
25. `smoothness_worst`: Mean of the 3 largest values for smoothness (range: 0.0712 – 0.2226)
26. `compactness_worst`: Mean of the 3 largest values for compactness (range: 0.0273 – 1.0580)
27. `concavity_worst`: Mean of the 3 largest values for concavity (range: 0.0 – 1.2520)
28. `concave points_worst`: Mean of the 3 largest values for concave points (range: 0.0 – 0.2910)
29. `symmetry_worst`: Mean of the 3 largest values for symmetry (range: 0.1565 – 0.6638)
30. `fractal_dimension_worst`: Mean of the 3 largest values for fractal dimension (range: 0.0550 – 0.2075)

---

## 4. Preprocessing Strategy & Leakage Controls

* **Zero Imputation:** No imputer was applied because all 30 predictor features have 0 missing values.
* **StandardScaler:** Applied across all 30 numerical features using scikit-learn's `ColumnTransformer`. Centers each feature to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
* **Strict Training Isolation:** The scaling transformer was fitted **strictly on the training split** ($X_{\text{train}}$) and applied to validation/test sets via `.transform()` only. Preprocessor parameters ($\mu, \sigma^2$) were never computed on test data.
* **Outlier Preservation:** Statistical outliers identified during EDA (such as high `area_se` values) were **not removed, clipped, or Winsorized**. In FNA cytology, extreme cellular pleomorphism and enlargement reflect authentic biological markers of high-grade malignancy; discarding them would impair screening sensitivity.
* **No PCA / Dimensionality Reduction:** Clinical interpretability of individual cytological parameters is fully preserved.
* **No SMOTE / Resampling:** Natural empirical distribution (62.6% Benign, 37.4% Malignant in training) is preserved; stratified sampling handles class balance without introducing synthetic artifacts.

---

## 5. Train/Test Split Configuration

* **Partition Strategy:** Stratified random sampling on target `diagnosis`
* **Split Ratio:** **80% Training / 20% Held-Out Test** (`test_size = 0.20`)
* **Random Seed:** `random_state = 42` (deterministic reproducibility)
* **Training Partition ($X_{\text{train}}$):** `455` samples
  * Class 0 (Benign): `285` samples (62.64%)
  * Class 1 (Malignant): `170` samples (37.36%)
* **Test Partition ($X_{\text{test}}$):** `114` samples (quarantined during model development)
  * Class 0 (Benign): `72` samples (63.16%)
  * Class 1 (Malignant): `42` samples (36.84%)
* **Mutual Exclusivity:** Zero index overlap between training and test sets.

---

## 6. Candidate Model Exploration (5-Fold Stratified Cross-Validation)

Five candidate binary classification architectures were evaluated strictly on the training partition ($N_{\text{train}} = 455$) using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` with fold-isolated preprocessing:

| Candidate Architecture | CV Accuracy | CV Precision (M) | CV Recall / Sensitivity (M) | CV F1-Score (M) | CV ROC-AUC | CV False Negatives (across 5 folds)* |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **$0.9736 \pm 0.0149$** | **$0.9771 \pm 0.0280$** | **$0.9529 \pm 0.0399$** | **$0.9640 \pm 0.0207$** | **$0.9958 \pm 0.0047$** | **8** / 170 |
| **Random Forest** | $0.9648 \pm 0.0146$ | $0.9641 \pm 0.0107$ | $0.9412 \pm 0.0416$ | $0.9519 \pm 0.0207$ | $0.9889 \pm 0.0067$ | **10** / 170 |
| **Support Vector Classifier (RBF)** | $0.9714 \pm 0.0054$ | $0.9769 \pm 0.0210$ | $0.9471 \pm 0.0343$ | $0.9610 \pm 0.0082$ | $0.9949 \pm 0.0050$ | **9** / 170 |
| **Gradient Boosting** | $0.9670 \pm 0.0139$ | $0.9646 \pm 0.0211$ | $0.9471 \pm 0.0343$ | $0.9553 \pm 0.0196$ | $0.9908 \pm 0.0054$ | **9** / 170 |
| **XGBoost** | $0.9714 \pm 0.0112$ | $0.9704 \pm 0.0181$ | $0.9529 \pm 0.0235$ | $0.9614 \pm 0.0154$ | $0.9940 \pm 0.0036$ | **8** / 170 |

*\* Measured across 170 total malignant validation cases across the 5 cross-validation folds.*

---

## 7. Model Selection Rationale

* **Selected Model:** **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
* **Baseline Hyperparameters:** `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`.
* **Selection Criteria:**
  1. **Screening Sensitivity (Recall):** Minimizing missed cancer cases is the primary screening priority. Logistic Regression tied for highest recall ($0.9529$) with the fewest total False Negatives ($8/170$).
  2. **Discrimination (ROC-AUC):** Achieved the highest mean ROC-AUC ($0.9958 \pm 0.0047$) across all candidate architectures.
  3. **F1 and Precision Balance:** Achieved the highest mean F1-score ($0.9640$) and precision ($0.9771$).
  4. **Parsimony and Calibration:** As an $L_2$-regularized linear model, it is computationally lightweight, robust against overfitting on continuous tabular data, and provides smooth, native probability calibration via the sigmoid link without requiring surrogate Platt scaling.

*(Note: Logistic Regression was selected purely on empirical statistical criteria from the training cross-validation data; no claim of medical superiority is made).*

---

## 8. Final Held-Out Test Evaluation

The selected Logistic Regression pipeline was fitted on the full training set ($N_{\text{train}} = 455$) and evaluated **exactly once** on the quarantined held-out test split ($N_{\text{test}} = 114$ samples; 72 Benign, 42 Malignant):

| Metric | Cross-Validation Mean (Training $X_{\text{train}}$) | Final Test Result ($X_{\text{test}}$) |
|---|---|---|
| **Accuracy** | $0.9736 \pm 0.0149$ | **0.9649** (110 / 114 correct) |
| **Precision (Malignant)** | $0.9771 \pm 0.0280$ | **0.9750** (39 / 40) |
| **Recall / Sensitivity (Malignant)** | $0.9529 \pm 0.0399$ | **0.9286** (39 / 42) |
| **F1-Score (Malignant)** | $0.9640 \pm 0.0207$ | **0.9512** |
| **ROC-AUC** | $0.9958 \pm 0.0047$ | **0.9960** |
| **PR-AUC (Average Precision)** | *(Evaluated on test set)* | **0.9943** |

### Confusion Matrix on Unseen Test Partition ($N = 114$):
$$\begin{pmatrix} \text{True Negatives (TN)} & \text{False Positives (FP)} \\ \text{False Negatives (FN)} & \text{True Positives (TP)} \end{pmatrix} = \begin{pmatrix} 71 & 1 \\ 3 & 39 \end{pmatrix}$$

* **True Negatives (TN):** `71`
* **False Positives (FP):** `1`
* **False Negatives (FN):** `3`
* **True Positives (TP):** `39`
* **Confusion Matrix Plot:** [`backend/models/breast_cancer/plots/11_final_confusion_matrix.png`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/plots/11_final_confusion_matrix.png)

---

## 9. Serialized Artifacts

The following production artifacts are stored in `backend/models/breast_cancer/`:

1. [`model.joblib`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/model.joblib) (4,785 bytes): Fitted scikit-learn `Pipeline` encapsulating the `ColumnTransformer` (`StandardScaler`) and `LogisticRegression` estimator.
2. [`preprocessor.joblib`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/preprocessor.joblib) (3,806 bytes): Fitted `ColumnTransformer` for standalone feature transformation.
3. [`feature_schema.json`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/feature_schema.json) (11,294 bytes): Machine-readable schema specifying feature names, canonical order, min/max bounds, target mapping, and input requirements.
4. [`metrics.json`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/metrics.json) (3,262 bytes): Machine-readable record of training CV metrics, selection rationale, and final test evaluation results.

---

## 10. Prediction Usage & Interface

The inference interface is implemented in [`backend/models/breast_cancer/predict.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/predict.py). It automatically validates inputs, constructs DataFrame representations in canonical order, applies preprocessing, and returns structured probability estimates:

```python
from backend.models.breast_cancer.predict import predict

# Example single patient input dictionary with all 30 features
synthetic_sample = {
    "radius_mean": 14.0, "texture_mean": 19.0, "perimeter_mean": 90.0, "area_mean": 600.0,
    "smoothness_mean": 0.09, "compactness_mean": 0.10, "concavity_mean": 0.08, "concave points_mean": 0.05,
    "symmetry_mean": 0.18, "fractal_dimension_mean": 0.06, "radius_se": 0.40, "texture_se": 1.20,
    "perimeter_se": 2.80, "area_se": 40.0, "smoothness_se": 0.007, "compactness_se": 0.025,
    "concavity_se": 0.03, "concave points_se": 0.012, "symmetry_se": 0.02, "fractal_dimension_se": 0.003,
    "radius_worst": 16.0, "texture_worst": 25.0, "perimeter_worst": 105.0, "area_worst": 800.0,
    "smoothness_worst": 0.13, "compactness_worst": 0.25, "concavity_worst": 0.25, "concave points_worst": 0.11,
    "symmetry_worst": 0.28, "fractal_dimension_worst": 0.08
}

result = predict(synthetic_sample)
print(result)
```

### Returned Structure:
```json
{
  "status": "success",
  "disease": "breast_cancer",
  "predicted_class": 0,
  "predicted_probability": 0.063786,
  "class_probabilities": {
    "0": 0.936214,
    "1": 0.063786
  },
  "screening_estimate": "Class 0",
  "disclaimer": "This is an AI screening estimate for risk evaluation and does not constitute a medical diagnosis."
}
```

---

## 11. Validation and Testing Suites

Three automated test suites verify the complete lifecycle of the model:

| Test Suite File | Test Count | Scope & Verifications |
|---|---|---|
| [`test_predict.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/test_predict.py) | **10 / 10 Passed** | Validates prediction module imports, artifact loading, discrete classes, probability normalization, canonical ordering, and strict rejection of missing, unknown, non-numeric, and non-finite features. |
| [`test_artifacts.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/test_artifacts.py) | **8 / 8 Passed** | Validates disk presence, joblib deserialization, pipeline structure, feature schema alignment, standalone preprocessor execution, and test-set prediction consistency. |
| [`test_breast_cancer.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/test_breast_cancer.py) | **20 / 20 Passed** | End-to-end integration suite verifying dataset loadability, raw dimensions (569 x 33), pipeline components, schema fidelity, metrics immutability, and boundary controls. |

* **Total Tests Executed:** **38 tests** (All 38 passed with zero failures).

---

## 12. Supported Limitations & Operational Boundaries

1. **Dataset-Based Evaluation:** Performance is established on the Wisconsin Diagnostic Breast Cancer (WDBC) cohort. Generalizability to external clinical populations, alternative imaging modalities, or non-FNA biopsy data has not been clinically evaluated.
2. **Held-Out Test Set Scope:** Evaluation reflects a single held-out partition of 114 samples ($20\%$).
3. **No Clinical Validation:** This system has not undergone clinical trials, multi-center retrospective validation, or regulatory clearance (e.g., FDA/CE-IVD).
4. **Screening Estimate Rather than Diagnosis:** The model generates probability scores to highlight elevated risk; it cannot replace histological tissue biopsy, mammography, ultrasound, or oncological evaluation.
5. **No Medical Threshold Inferred:** The default decision threshold ($0.50$) is mathematical, not a clinical risk threshold.

---

## 13. Handover Package for Integration (Member 3 / Platform)

The following components in `backend/models/breast_cancer/` constitute the complete handover package for integration into backend API routing and multi-disease screening services:

1. [`model.joblib`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/model.joblib): Complete preprocessor + classifier pipeline ready for inference.
2. [`preprocessor.joblib`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/preprocessor.joblib): Standalone scaling transformer.
3. [`feature_schema.json`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/feature_schema.json): Canonical API contract for input validation.
4. [`predict.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/predict.py): Self-contained inference entry-point for FastAPI endpoint integration.
5. [`metrics.json`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/metrics.json): Model evaluation benchmark data for documentation and compliance auditing.
6. [`README.md`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI-Multi-Disease-Risk-Screening-and-Prediction-System/backend/models/breast_cancer/README.md): Comprehensive model documentation and technical specifications.
