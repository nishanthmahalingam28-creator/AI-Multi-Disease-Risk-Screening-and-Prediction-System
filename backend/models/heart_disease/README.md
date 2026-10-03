# Heart Disease Prediction Model

## 1. Overview

This module provides the heart disease risk screening and prediction model within the AI Multi-Disease Risk Screening and Prediction System. The model produces an automated, AI-based screening estimate and a predicted class (Class 0 or Class 1) along with predicted probabilities for the defined dataset classes.

This component is designed solely for empirical risk screening experimentation. It does not diagnose heart disease, provide medical conclusions, or establish clinical effectiveness. Model performance on this dataset does not constitute clinical validation.

## 2. Dataset

* **Dataset Path**: `datasets/heart_disease/heart.csv`
* **Shape**: 270 rows × 14 columns
* **Target Column**: `Heart Disease`
* **Target Mapping**:
  * `Absence` = `0` (Class 0)
  * `Presence` = `1` (Class 1)
* **Class Distribution**:
  * `Absence`: 150 rows (55.56%)
  * `Presence`: 120 rows (44.44%)
* **Data Completeness**:
  * 13 predictor features
  * 0 standard missing/null values across all columns
  * 0 duplicate rows
  * 0 constant predictor columns

The 13 predictor features in their canonical dataset order:

1. `Age`
2. `Sex`
3. `Chest pain type`
4. `BP`
5. `Cholesterol`
6. `FBS over 120`
7. `EKG results`
8. `Max HR`
9. `Exercise angina`
10. `ST depression`
11. `Slope of ST`
12. `Number of vessels fluro`
13. `Thallium`

### Observed Zero Values (EDA Findings)
The continuous vital measurement columns (`Age`, `BP`, `Cholesterol`, `Max HR`) contain strictly positive readings with zero count of 0. Observed zero values in the dataset correspond to binary indicator states or discrete counts:

* `Sex`: 87 zeros (32.22%)
* `FBS over 120`: 230 zeros (85.19%)
* `EKG results`: 131 zeros (48.52%)
* `Exercise angina`: 181 zeros (67.04%)
* `ST depression`: 85 zeros (31.48%)
* `Number of vessels fluro`: 160 zeros (59.26%)

These counts reflect valid nominal/ordinal/count observations and were preserved as documented observations rather than modified.

## 3. EDA Summary

An exploratory data analysis was conducted on the raw dataset:

* **Target Distribution**: A balanced distribution with an imbalance ratio of 1.25 : 1 (150 Absence vs 120 Presence).
* **Feature Distributions**: Vitals (`BP`, `Cholesterol`, `Max HR`) exhibit standard continuous distributions; `Cholesterol` (skewness +1.18) and `ST depression` (skewness +1.26) display positive skewness with high-value right tails.
* **Outlier Inspection**: Statistical IQR diagnostic flagging identified potential outliers in `BP` (9 records, upper bound 170.0), `Cholesterol` (5 records, upper bound 380.5), `ST depression` (4 records, upper bound 4.0), and `Max HR` (1 record, lower bound 83.5). These represent statistical outlier flags, not confirmed errors, and were retained.
* **Leakage Inspection**: No obvious target leakage was identified during feature and target inspection. Correlation analysis was used as a diagnostic check and does not by itself prove the absence of leakage or proxy variables.

### Observed Correlations

Linear correlation (Pearson $r$) with the binary target variable (Presence = 1, Absence = 0):

* `Thallium`: 0.5250
* `Number of vessels fluro`: 0.4553
* `Exercise angina`: 0.4193
* `Max HR`: -0.4185
* `ST depression`: 0.4180
* `Chest pain type`: 0.4174
* `Slope of ST`: 0.3376
* `Sex`: 0.2977
* `Age`: 0.2123
* `EKG results`: 0.1821
* `BP`: 0.1554
* `Cholesterol`: 0.1180
* `FBS over 120`: -0.0163

**Strongest Feature-to-Feature Correlation**:
* `ST depression` vs `Slope of ST`: $r = 0.6097$

No feature pair exceeded $|r| \ge 0.85$, and no feature demonstrated correlation exceeding 0.5250 with the target. These observations represent statistical associations only and do not establish clinical causation.

## 4. Train/Test Split

* **Split Configuration**: 80% training, 20% test
* **Stratification**: Stratified by target variable `Heart Disease` (`stratify=y`)
* **Random State**: `random_state=42`
* **Total Samples**: 270
* **Training Rows**: 216
  * Absence: 120 rows (55.56%)
  * Presence: 96 rows (44.44%)
* **Test Rows**: 54
  * Absence: 30 rows (55.56%)
  * Presence: 24 rows (44.44%)

The 54-row test set was strictly quarantined and not accessed during candidate model cross-validation, hyperparameter selection, or preprocessing comparison.

## 5. Preprocessing

The preprocessing pipeline follows the selected **Approach B** design implemented via a scikit-learn `ColumnTransformer`:

* **Continuous Features (5)**:
  * `Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`
  * Preprocessing: `SimpleImputer(strategy='median')` followed by `RobustScaler()`
  * Training partition learned medians: `Age = 54.0`, `BP = 130.0`, `Cholesterol = 243.0`, `Max HR = 153.5`, `ST depression = 0.8`
  * `RobustScaler` scales using the median and interquartile range (IQR), providing resistance against extreme values in cholesterol and ST depression.
* **Binary Features (3)**:
  * `Sex`, `FBS over 120`, `Exercise angina`
  * Preprocessing: `passthrough` (exact binary values 0 and 1 preserved).
* **Discrete-Coded Features (5)**:
  * `Chest pain type`, `EKG results`, `Slope of ST`, `Number of vessels fluro`, `Thallium`
  * Preprocessing: `passthrough` (observed integer levels preserved without artificial continuous scaling).

**Integrity Constraints**:
* No outlier removal or value clipping was performed.
* No SMOTE, random resampling, or class weighting was applied.
* No PCA or dimensionality reduction was utilized.
* No manual feature engineering or interaction terms were introduced.
* The target variable was strictly excluded from preprocessing.
* Preprocessor parameters were fitted exclusively on training data ($X_{\text{train}}$).

## 6. Candidate Models

Five candidate classifiers were evaluated using 5-fold StratifiedKFold cross-validation exclusively on the 216-row training partition (`shuffle=True, random_state=42`). Preprocessing was fitted fold-by-fold strictly inside each cross-validation fold to guarantee zero data leakage.

### 5-Fold Cross-Validation Performance Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8192 ± 0.0523 | 0.8070 ± 0.0824 | 0.7921 ± 0.0928 | 0.7946 ± 0.0629 | 0.9007 ± 0.0388 | 0.9060 ± 0.0340 |
| **Random Forest** | 0.8332 ± 0.0478 | 0.8410 ± 0.0713 | 0.7811 ± 0.1076 | 0.8035 ± 0.0670 | 0.9064 ± 0.0378 | 0.9078 ± 0.0378 |
| **SVC** | 0.8147 ± 0.0217 | 0.8315 ± 0.0577 | 0.7400 ± 0.0724 | 0.7788 ± 0.0330 | 0.8989 ± 0.0187 | 0.9035 ± 0.0233 |
| **Gradient Boosting** | 0.7823 ± 0.0351 | 0.7304 ± 0.0438 | 0.8126 ± 0.0530 | 0.7683 ± 0.0381 | 0.8816 ± 0.0507 | 0.8761 ± 0.0542 |
| **XGBoost** | 0.8099 ± 0.0601 | 0.7932 ± 0.0824 | 0.7811 ± 0.0846 | 0.7846 ± 0.0698 | 0.8837 ± 0.0605 | 0.8807 ± 0.0647 |

**Validation Error Counts**:
* **Random Forest**: False Negatives (FN) = 21, False Positives (FP) = 15
* **Logistic Regression**: FN = 20, FP = 19
* **Support Vector Classifier**: FN = 25, FP = 15
* **Gradient Boosting**: FN = 18, FP = 29
* **XGBoost**: FN = 21, FP = 20

*(Total positive validation cases across 5 folds: 96; total negative validation cases: 120).*

## 7. Final Model Selection

* **Selected Model**: **Random Forest**
* **Model Configuration**:
  ```python
  RandomForestClassifier(
      n_estimators=300,
      random_state=42,
      n_jobs=-1
  )
  ```

**Selection Basis**:
Random Forest was selected based on the documented empirical cross-validation comparison in Step 6. Among the five evaluated candidates, it achieved the highest mean Accuracy (0.8332), highest Precision (0.8410), highest F1-Score (0.8035), highest ROC-AUC (0.9064), and highest PR-AUC (0.9078). It also produced the lowest cumulative False Positives (15).

While Gradient Boosting recorded fewer false negatives in the cross-validation aggregate (18 vs 21), it incurred nearly double the false positives (29 vs 15) and lower values across the other reported metrics (Accuracy 0.7823, Precision 0.7304, F1 0.7683, ROC-AUC 0.8816). Logistic Regression was competitive (ROC-AUC 0.9007, F1 0.7946, 20 FNs, 19 FPs), but Random Forest provided a higher overall metric profile on the training data.

## 8. Final Test Evaluation

The selected pipeline (ColumnTransformer Approach B + RandomForestClassifier) was trained on all 216 training samples and evaluated exactly once on the quarantined 54-row test set.

### Test Metrics ($N = 54$)

* **Accuracy**: **0.8148** (81.48%)
* **Precision**: **0.7692** (76.92%)
* **Recall (Sensitivity)**: **0.8333** (83.33%)
* **F1-Score**: **0.8000** (80.00%)
* **ROC-AUC**: **0.8729**
* **PR-AUC**: **0.8013**
* **False Positive Rate (FPR)**: **0.2000**
* **False Negative Rate (FNR)**: **0.1667**

### Test Confusion Matrix

| | Predicted Absence (0) | Predicted Presence (1) |
| :--- | :---: | :---: |
| **Actual Absence (0)** | **24** (TN) | **6** (FP) |
| **Actual Presence (1)** | **4** (FN) | **20** (TP) |

* **True Negatives (TN)**: 24
* **False Positives (FP)**: 6
* **False Negatives (FN)**: 4
* **True Positives (TP)**: 20

Out of 24 actual positive cases in the quarantined test partition, 20 were correctly identified by the model with 4 false negatives. These metrics are dataset-specific test results and do not represent clinical diagnostic performance.

## 9. Prediction Module

* **Module File**: `backend/models/heart_disease/predict.py`
* **Primary Interface**: `predict_heart_disease(input_data)`
* **Convenience Aliases**: `predict(input_data)`, `batch_predict(records)`

### Supported Input Types
* Python `dict`
* `pandas.Series`
* Single-row `pandas.DataFrame`

### Validation Safeguards
The prediction module enforces strict schema validation and rejects:
* Missing required features
* Unexpected or extra features
* Non-numeric values (e.g., strings)
* `NaN` values
* Infinite values (`inf` / `-inf`)
* Boolean values
* Multi-row DataFrames when single-sample prediction is expected

The module loads `model.joblib` and passes raw validated feature values directly to the embedded pipeline, avoiding manual duplication of preprocessing.

### Example Usage

```python
from backend.models.heart_disease.predict import predict_heart_disease

sample_patient = {
    "Age": 70,
    "Sex": 1,
    "Chest pain type": 4,
    "BP": 130,
    "Cholesterol": 322,
    "FBS over 120": 0,
    "EKG results": 2,
    "Max HR": 109,
    "Exercise angina": 0,
    "ST depression": 2.4,
    "Slope of ST": 2,
    "Number of vessels fluro": 3,
    "Thallium": 3
}

result = predict_heart_disease(sample_patient)
print(result)
```

### Example Model Output

```json
{
  "status": "success",
  "model_name": "Random_Forest",
  "predicted_class": 1,
  "predicted_probability": 0.8967,
  "class_probabilities": {
    "0": 0.1033,
    "1": 0.8967
  },
  "screening_estimate": "Class 1"
}
```

*(This output represents an AI model-generated screening estimate for the defined dataset classes and does not constitute a medical diagnosis).*

## 10. Saved Artifacts

```text
backend/models/heart_disease/
├── inspect_dataset.py          # Step 1: Read-only dataset inspection script
├── eda.py                      # Step 2: Exploratory data analysis execution script
├── eda_summary.json            # Step 2: Factual summary of EDA metrics and distributions
├── plots/                      # Step 2: Generated visual plots (PNG)
│   ├── 01_target_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_feature_boxplots.png
│   ├── 04_correlation_heatmap.png
│   └── 05_feature_vs_target.png
├── define_features.py          # Step 3: Feature and target extraction module
├── feature_schema.json         # Step 3: Canonical 13-feature schema and target definition
├── split_data.py               # Step 4: Stratified 80/20 train/test split generator
├── preprocessing.py            # Step 5: Preprocessing pipeline definition (Approach B)
├── test_preprocessing.py       # Step 5: Unit test suite for preprocessing (13 tests)
├── candidate_cv.py             # Step 6: 5-fold cross-validation candidate evaluation script
├── candidate_cv_results.json   # Step 6: Empirical CV comparison metrics for 5 candidates
├── final_evaluation.py         # Step 7: Final model training and quarantined test evaluation
├── model.joblib                # Step 7: Serialized production pipeline artifact (1.81 MB)
├── metrics.json                # Step 7: Final test evaluation metrics and confusion matrix
├── model_selection.json        # Step 7: Model selection justification and CV report
├── predict.py                  # Step 8: Stateless prediction and input validation module
└── tests/                      # Step 9: Automated pytest test suite
    ├── __init__.py
    └── test_heart_disease.py
```

## 11. Testing

Automated testing was conducted using pytest:

```bash
python -m pytest -q backend/models/heart_disease/tests
```

**Results**:
```text
36 passed in 4.65s
```

* **Tests Collected**: 36
* **Passed**: 36
* **Failed**: 0
* **Skipped**: 0
* **Warnings / Errors**: 0

The test suite systematically verifies:
* Artifact file existence and loadability
* Pipeline architecture (ColumnTransformer and RandomForestClassifier)
* Canonical feature schema, types, and ordering
* Prediction output structure and probability calibration bounds ($[0.0, 1.0]$)
* Input container flexibility (`dict`, `pd.Series`, 1-row `pd.DataFrame`)
* Rejection of invalid inputs (missing, extra, non-numeric, NaN, infinite, boolean, multi-row)
* Input container immutability during prediction
* Deterministic output consistency
* Batch prediction interface
* Final metrics and confusion matrix validation
* Model selection quarantine confirmation
* Dataset file integrity and non-modification
* Neutral output wording and absence of diagnostic claims

## 12. Limitations

* **Sample Size**: The model was trained on 216 training samples and evaluated on 54 test samples from a single dataset of 270 records.
* **Population Representativeness**: The dataset reflects specific clinical research cohorts and may not represent other patient populations, demographics, or clinical environments.
* **Dataset-Specific Performance**: Cross-validation and test metrics reflect performance on this specific dataset partition and cannot be assumed to generalize without broader validation.
* **Absence of Clinical Validation**: No clinical trials, prospective evaluations, or medical device certifications have been conducted.
* **Screening Estimates Only**: Predictions represent model-generated screening estimates for dataset classes (`Class 0` and `Class 1`) and do not constitute a medical diagnosis.
* **Sensitivity to Extreme Inputs**: Model behavior on extreme, out-of-distribution, or missing input measurements has not been clinically validated.
* **Threshold Calibration**: Output probabilities are generated using Random Forest vote distributions under the default 0.50 decision threshold; application-specific calibration may be required.

## 13. Integration Notes

Downstream services integrating with the heart disease screening model should note:

* The saved model (`model.joblib`) is a complete pipeline encapsulating both preprocessing (ColumnTransformer) and classification (Random Forest).
* Integration should invoke `predict_heart_disease(input_data)` from `backend/models/heart_disease/predict.py`.
* Downstream callers must supply all 13 features matching the exact names and order defined in `feature_schema.json`.
* Callers should validate inputs before invocation and handle structured error responses.
* Returned predictions must be presented to users as AI-generated screening estimates, using neutral language and explicitly disclaiming medical diagnosis.

*(Note: API router and endpoint integration will be addressed in subsequent backend development steps).*

## 14. Reproducibility

* **Environment**: Python 3.14.6
* **Core Libraries**: `scikit-learn==1.9.1`, `xgboost==3.4.1`, `pandas`, `numpy`, `joblib`
* **Random Seed**: `random_state = 42` (applied to train/test split, cross-validation, and classifier initialization)
* **Partition Protocol**: Stratified 80/20 split (`stratify=y`, 216 train / 54 test)
* **Cross-Validation**: 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` with fold-local preprocessing

## 15. Important Safety and Scope Note

This software module is developed strictly for research, educational, and risk-screening experimentation as part of the AI Multi-Disease Risk Screening and Prediction System. It is not an approved medical device, does not provide medical advice or diagnosis, and should never be used as a substitute for professional healthcare evaluation.
