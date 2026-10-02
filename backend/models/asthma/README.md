# Asthma Risk-Screening Model

## 1. Dataset Overview
- **Dataset File**: `datasets/asthma/synthetic_asthma_dataset.csv`
- **Total Records**: 10,000 patient rows
- **Raw Columns**: 17 columns
- **Predictor Features Used**: 14 features
- **Missing Values**: 0 missing values across all columns (literal string categories `'None'` and `'N/A'` are preserved as valid categorical tokens via NA-safe ingestion).
- **Dataset Nature**: **Synthetic Dataset**. Generated algorithmically for research and software testing. It contains mathematical decision surfaces that do not represent real-world clinical heterogeneity.

---

## 2. Target Variable
- **Target Column**: `Has_Asthma`
- **Class 0 (`No Asthma`)**: 7,567 records (75.67%)
- **Class 1 (`Has Asthma`)**: 2,433 records (24.33%)
- **Imbalance Ratio**: Approximately 3.11 : 1

---

## 3. Excluded Columns & Leakage Prevention

Two columns from the raw CSV are **strictly excluded** from the predictor feature matrix:

1. **`Patient_ID`**:
   - Sequential synthetic identifier (`ASTH100000`–`ASTH109999`). Excluded to prevent arbitrary index memorization.
2. **`Asthma_Control_Level` (Critical Target Leakage)**:
   - In the raw dataset, `Asthma_Control_Level` is populated only for patients diagnosed with asthma:
     - When `Has_Asthma == 0`, `Asthma_Control_Level` is strictly `'N/A'` (100% of rows).
     - When `Has_Asthma == 1`, `Asthma_Control_Level` contains management status (`'Not Controlled'`, `'Poorly Controlled'`, `'Well Controlled'`).
   - Including this column as an input would cause 100% target leakage. It is permanently quarantined and never fed into the preprocessor or model.

---

## 4. Feature Schema (14 Predictors)

### Numerical Features (Scaled with `StandardScaler`):
- `Age`: Patient age (range 1–89 years).
- `BMI`: Body Mass Index (range 15.0–45.0 kg/m²).
- `Medication_Adherence`: Prescription adherence score (range 0.00–0.99).
- `Number_of_ER_Visits`: Emergency room visits count (range 0–6).
- `Peak_Expiratory_Flow`: Spirometry PEF measurement (range 150.0–600.0 L/min).
- `FeNO_Level`: Fractional exhaled Nitric Oxide (range 5.0–63.9 ppb).

### Categorical Features (Encoded with `OneHotEncoder(handle_unknown='ignore')`):
- `Gender`: `Female`, `Male`, `Other`.
- `Smoking_Status`: `Never`, `Former`, `Current`.
- `Allergies`: `None`, `Dust`, `Pollen`, `Pets`, `Multiple`.
- `Air_Pollution_Level`: `Low`, `Moderate`, `High`.
- `Physical_Activity_Level`: `Sedentary`, `Moderate`, `Active`.
- `Occupation_Type`: `Indoor`, `Outdoor`.
- `Comorbidities`: `None`, `Diabetes`, `Hypertension`, `Both`.

### Binary Feature (Passthrough):
- `Family_History`: Binary indicator `{0, 1}` of family history of asthma.

---

## 5. Preprocessing & Partitioning Strategy
- **Ingestion**: Loaded with `keep_default_na=False` to preserve `'None'` as a distinct valid medical category.
- **Partitioning**: 80% Train ($N = 8,000$), 20% Held-Out Test ($N = 2,000$) stratified on `Has_Asthma` using `random_state=42`.
- **Zero Leakage Pipeline**: The `ColumnTransformer` (StandardScaler + OneHotEncoder) is fitted strictly on the training partition inside each cross-validation fold. The test set remained quarantined until final verification.

---

## 6. 5-Fold Stratified Cross-Validation Comparison

Candidate estimators with balanced class weights were evaluated on the training partition ($N = 8,000$):

| Candidate Model | CV Accuracy | CV Balanced Acc | CV Precision (Pos) | CV Recall (Pos) | CV F1 (Pos) | CV ROC-AUC | CV PR-AUC | CV Recall (Neg) |
|---|---|---|---|---|---|---|---|---|
| **HistGradientBoosting (balanced)** *(Selected)* | **0.9999 ± 0.0002** | **0.9999 ± 0.0002** | **0.9995 ± 0.0010** | **1.0000 ± 0.0000** | **0.9997 ± 0.0005** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** | **0.9998 ± 0.0003** |
| **Random Forest (balanced, max_depth=8)** | 0.9956 ± 0.0029 | 0.9921 ± 0.0056 | 0.9969 ± 0.0041 | 0.9851 ± 0.0111 | 0.9909 ± 0.0061 | 0.9999 ± 0.0001 | 0.9996 ± 0.0003 | 0.9990 ± 0.0013 |
| **Logistic Regression (balanced, L2)** | 0.9948 ± 0.0011 | 0.9950 ± 0.0017 | 0.9833 ± 0.0046 | 0.9954 ± 0.0038 | 0.9893 ± 0.0022 | 0.9999 ± 0.0001 | 0.9997 ± 0.0002 | 0.9945 ± 0.0015 |
| **Support Vector Classifier (balanced, RBF)** | 0.9916 ± 0.0025 | 0.9894 ± 0.0048 | 0.9806 ± 0.0066 | 0.9851 ± 0.0099 | 0.9828 ± 0.0051 | 0.9996 ± 0.0002 | 0.9989 ± 0.0007 | 0.9937 ± 0.0022 |

---

## 7. Model Selection Rationale & Trade-offs
- **Selected Model**: `HistGradientBoostingClassifier(class_weight='balanced', max_iter=100, random_state=42)`
- **Selection Basis**:
  - Highest Cross-Validation positive-class recall (**1.0000 ± 0.0000** across all 5 folds), meaning zero false negatives in cross-validation.
  - Highest F1-Score (**0.9997**) and ROC-AUC (**1.0000**).
  - Native handling of tabular feature interactions and non-linear splits.
- **Trade-off & Synthetic Context**:
  - All four models achieve >99% performance because this synthetic dataset was generated with clean mathematical decision boundaries.
  - Logistic Regression remains a highly competitive linear alternative with 0.9954 recall, but HistGradientBoosting is selected for its complete capture of interaction rules.

---

## 8. Final Evaluation on Quarantined Test Set ($N = 2,000$)

Evaluated once on the untouched held-out test split (1,513 negative controls, 487 positive cases):

- **Accuracy**: **99.95%** (1,999 / 2,000 correct)
- **Balanced Accuracy**: **99.97%**
- **ROC-AUC**: **1.0000** (0.999999)
- **PR-AUC (Average Precision)**: **1.0000** (0.999996)
- **Positive Class (`Has Asthma = 1`)**:
  - Precision: **0.9980**
  - Recall: **1.0000** (487 / 487 positive cases detected; 0 false negatives)
  - F1-Score: **0.9990**
- **Negative Class (`No Asthma = 0`)**:
  - Precision: **1.0000**
  - Recall: **0.9993** (1,512 / 1,513 negative controls correctly classified; only 1 false positive)
  - F1-Score: **0.9997**
- **Confusion Matrix**:
  ```text
  [[TN = 1512,  FP = 1],
   [FN =    0,  TP = 487]]
  ```

---

## 9. Prediction Interface & Software Risk Tiers
- **Interface**: [`predict.py`](predict.py) provides a validated, stateless prediction interface.
- **Software Risk Tiers**:
  The prediction interface outputs a `risk_level` (`Low`, `Moderate`, `High`) based on predicted probability ranges (<0.30 = Low, 0.30–0.70 = Moderate, ≥0.70 = High).
  - **Important Notice**: These risk levels are **software-defined heuristic categories** for UI display and screening prioritization. They are **NOT** clinical diagnoses, medical thresholds, or clinical severity classifications.

---

## 10. Automated Unit Tests
- Test file: [`tests/test_asthma.py`](tests/test_asthma.py)
- All **10 unit tests** pass with `pytest` and `unittest`:
  1. `test_01_artifacts_exist`: PASSED
  2. `test_02_artifacts_load`: PASSED
  3. `test_03_expected_feature_validation`: PASSED
  4. `test_04_valid_sample_prediction`: PASSED
  5. `test_05_output_format_is_correct`: PASSED
  6. `test_06_probability_range_is_valid`: PASSED
  7. `test_07_missing_feature_rejected`: PASSED
  8. `test_08_unexpected_and_excluded_features_handled`: PASSED
  9. `test_09_repeated_prediction_is_deterministic`: PASSED
  10. `test_10_batch_prediction_works`: PASSED

---

## 11. Known Limitations & Synthetic Dataset Warning
1. **Synthetic Data Warning**: The dataset (`synthetic_asthma_dataset.csv`) is synthetic. The near-perfect evaluation metrics reflect deterministic synthetic generation rules and **do not** reflect real-world clinical performance or human physiological variability.
2. **Clinical Marker Disconnect**: Key clinical biomarkers such as Peak Expiratory Flow (PEF) and FeNO have near-zero correlation with asthma status in this synthetic generator, contrary to real clinical medicine.
3. **No Clinical Source Metadata**: No medical center or validation protocol is attached to this dataset.

---

## 12. Clinical Safety Disclaimer
> **IMPORTANT MEDICAL NOTICE**:
> This model is an **AI-based asthma risk-screening research component** and is **not a clinically validated diagnostic system**. It is intended strictly for risk screening demonstration and must never be used for self-diagnosis, clinical prescription, or emergency decisions. All asthma symptoms must be evaluated by licensed healthcare professionals and verified through spirometry and clinical pulmonary function testing.
