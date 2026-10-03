# Diabetes Prediction Model

## Overview

This module provides the diabetes risk screening model within the AI Multi-Disease Risk Screening and Prediction System. The model produces an AI-based screening estimate and a predicted class (Class 0 or Class 1) along with predicted probabilities. It is designed solely as an automated screening component and does not diagnose diabetes or provide medical conclusions.

## Dataset

* **Dataset file**: `datasets/diabetes/diabetes.csv`
* **Rows**: 768
* **Columns**: 9 (8 predictor features and 1 target column)
* **Target column**: `Outcome`
* **Target values**: `0` and `1`
* **Class distribution**:
  * Class 0: 500 rows (65.10%)
  * Class 1: 268 rows (34.90%)
* **Missing values**: 0 standard null or NaN values across all 9 columns.
* **Duplicates**: 0 duplicate rows.
* **Data quality observation**: Numerical zero values were observed in several continuous measurement columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`). These zeros were addressed through documented preprocessing assumptions rather than altering the raw dataset.

## Features

The model uses exactly 8 predictor features in the following canonical order:

1. `Pregnancies`
2. `Glucose`
3. `BloodPressure`
4. `SkinThickness`
5. `Insulin`
6. `BMI`
7. `DiabetesPedigreeFunction`
8. `Age`

* **Target**: `Outcome`

## Exploratory Data Analysis

A comprehensive exploratory analysis was performed without altering the raw dataset:

* **Target distribution**: 500 Class 0 observations (65.10%) and 268 Class 1 observations (34.90%), reflecting an imbalance ratio of approximately 1.87:1.
* **Zero-value counts**:
  * `Insulin`: 374 zeros (48.70%)
  * `SkinThickness`: 227 zeros (29.56%)
  * `BloodPressure`: 35 zeros (4.56%)
  * `BMI`: 11 zeros (1.43%)
  * `Glucose`: 5 zeros (0.65%)
  * `Pregnancies`: 111 zeros (14.45%)
* **Feature distributions & skewness**:
  * `Insulin` (+2.27), `DiabetesPedigreeFunction` (+1.92), `Age` (+1.13), and `Pregnancies` (+0.90) exhibit positive skewness.
  * `BloodPressure` (-1.84) exhibits negative skewness due to the recorded zero values.
* **IQR outlier counts**:
  * `BloodPressure`: 45 outliers (5.86%)
  * `Insulin`: 34 outliers (4.43%)
  * `DiabetesPedigreeFunction`: 29 outliers (3.78%)
  * `BMI`: 19 outliers (2.47%)
  * `Age`: 9 outliers (1.17%)
  * `Glucose`: 5 outliers (0.65%)
  * `Pregnancies`: 4 outliers (0.52%)
  * `SkinThickness`: 1 outlier (0.13%)
* **Feature-target correlations (Pearson $r$ with `Outcome`)**:
  * `Glucose`: $r = 0.4666$
  * `BMI`: $r = 0.2927$
  * `Age`: $r = 0.2384$
  * `Pregnancies`: $r = 0.2219$
  * `DiabetesPedigreeFunction`: $r = 0.1738$
  * `Insulin`: $r = 0.1305$
  * `SkinThickness`: $r = 0.0748$
  * `BloodPressure`: $r = 0.0651$
* **Strongest feature-feature correlation**: `Pregnancies` vs `Age` ($r = 0.5443$).
* **Target leakage**: No obvious target leakage was identified during the feature and target inspection. Correlation analysis was used as one diagnostic check and does not by itself prove the absence of leakage or proxy identifiers.

## Train/Test Split

* **Split ratio**: 80% training, 20% test
* **Stratification**: Stratified by target variable `Outcome` (`stratify=y`)
* **Random state**: `random_state=42`
* **Training rows**: 614 (400 Class 0, 214 Class 1)
* **Test rows**: 154 (100 Class 0, 54 Class 1)
* **Quarantine**: The 154-row test set was kept strictly untouched during exploratory data analysis, candidate model cross-validation, and model selection.

## Preprocessing

The preprocessing pipeline is implemented inside a unified scikit-learn pipeline:
`ZeroToNan -> SimpleImputer(strategy='median') -> RobustScaler -> Classifier`

* `Pregnancies` zero values are preserved as valid observations.
* Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are converted to `np.nan` within the pipeline.
* Missing values are imputed using median values computed exclusively on the training partition (`X_train` medians: `Pregnancies`: 3.0, `Glucose`: 117.0, `BloodPressure`: 72.0, `SkinThickness`: 29.0, `Insulin`: 125.0, `BMI`: 32.4, `DiabetesPedigreeFunction`: 0.38, `Age`: 29.0).
* `RobustScaler` scales features using median and interquartile range (IQR).
* All preprocessing transformers are fitted only on the training partition and applied without refitting to test or new incoming records.
* No SMOTE, random sampling, PCA, outlier removal, or feature engineering was utilized.

## Candidate Models

Five candidate models were evaluated using 5-fold stratified cross-validation on the 614-row training partition only:

* Logistic Regression (`max_iter=1000, random_state=42`)
* Random Forest (`n_estimators=300, random_state=42, n_jobs=-1`)
* Support Vector Classifier (`kernel='rbf', probability=True, random_state=42`)
* Gradient Boosting (`random_state=42`)
* XGBoost (`n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'`)

The cross-validation results from `backend/models/diabetes/candidate_cv_results.json`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.7899** | **0.7652** | 0.5794 | **0.6583** | **0.8434** |
| **Random Forest** | 0.7655 | 0.6965 | 0.5935 | 0.6391 | 0.8209 |
| **Support Vector Classifier (SVC)** | 0.7279 | 0.6794 | 0.3822 | 0.4714 | 0.8145 |
| **Gradient Boosting** | 0.7557 | 0.6721 | **0.5981** | 0.6308 | 0.8195 |
| **XGBoost** | 0.7589 | 0.6877 | 0.5842 | 0.6285 | 0.8205 |

## Selected Model

**Logistic Regression** was selected based on the documented cross-validation comparison:

* Highest mean Accuracy (0.7899)
* Highest mean Precision (0.7652)
* Highest mean F1-score (0.6583)
* Highest mean ROC-AUC (0.8434)
* Across folds, its recall standard deviation was 0.0152. While tree-based ensembles (Gradient Boosting and Random Forest) yielded marginally higher recall (0.5981 and 0.5935 vs 0.5794, representing 4 and 3 fewer false negatives across 614 validation samples), they incurred higher false positive counts (64 and 57 vs 39) and lower mean ROC-AUC (0.8195 and 0.8209 vs 0.8434).

## Final Test Evaluation

The selected pipeline was trained on all 614 training rows and evaluated once on the untouched 154-row test partition.

The test metrics recorded in `backend/models/diabetes/metrics.json`:

* **Accuracy**: 0.7013
* **Precision**: 0.5870
* **Recall / Sensitivity**: 0.5000
* **F1-Score**: 0.5400
* **ROC-AUC**: 0.8122
* **PR-AUC**: 0.6731

### Confusion Matrix (Test Set, $N = 154$)

| | Predicted Class 0 | Predicted Class 1 | Total Actual |
| :--- | :--- | :--- | :--- |
| **Actual Class 0** | **81** (TN) | **19** (FP) | 100 |
| **Actual Class 1** | **27** (FN) | **27** (TP) | 54 |
| **Total Predicted** | 108 | 46 | 154 |

* **False Positive Rate (FPR)**: 0.1900 (19 / 100)
* **False Negative Rate (FNR)**: 0.5000 (27 / 54)

## Prediction

The prediction module in `backend/models/diabetes/predict.py`:

* Loads `backend/models/diabetes/model.joblib`.
* Loads `backend/models/diabetes/feature_schema.json`.
* Validates incoming inputs: requires all 8 features in the canonical order, checks numeric types, and rejects missing features, unknown keys, NaNs, and infinite values.
* Passes raw inputs directly into the loaded pipeline (which executes zero-handling, imputation, scaling, and classification).
* Returns a dictionary containing `status`, `model_name`, `predicted_class`, `predicted_probability`, `class_probabilities`, and `screening_estimate`.
* Does not retrain the model.

### Example Usage

```bash
python backend/models/diabetes/predict.py
```

Python usage:

```python
from backend.models.diabetes.predict import predict_diabetes

sample_input = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}

result = predict_diabetes(sample_input)
print(result)
# Output:
# {
#   "status": "success",
#   "model_name": "Logistic_Regression",
#   "predicted_class": 1,
#   "predicted_probability": 0.7026,
#   "class_probabilities": {"0": 0.2974, "1": 0.7026},
#   "screening_estimate": "Class 1"
# }
```

## Artifacts

All model artifacts are stored in repository-relative paths:

* `backend/models/diabetes/model.joblib` — Serialized production pipeline
* `backend/models/diabetes/preprocessing.py` — Preprocessing pipeline definitions
* `backend/models/diabetes/predict.py` — Inference module and validation logic
* `backend/models/diabetes/feature_schema.json` — Canonical schema of features and data types
* `backend/models/diabetes/metrics.json` — Test evaluation metrics and confusion matrix
* `backend/models/diabetes/model_selection.json` — Model selection justification and CV comparison
* `backend/models/diabetes/candidate_cv_results.json` — 5-fold cross-validation metrics for all candidates
* `backend/models/diabetes/eda_summary.json` — Exploratory data analysis summary
* `backend/models/diabetes/inspect_dataset.py` — Dataset inspection script
* `backend/models/diabetes/define_features.py` — Feature and target extraction script
* `backend/models/diabetes/split_data.py` — Train/test split generator
* `backend/models/diabetes/final_evaluation.py` — Final evaluation and model serialization script
* `backend/models/diabetes/test_preprocessing.py` — Unit tests for preprocessing
* `backend/models/diabetes/tests/test_diabetes.py` — Automated integration and unit test suite
* `backend/models/diabetes/plots/` — Generated EDA plots

## Testing

Automated testing was conducted using pytest:

* **Test Framework**: `pytest 9.1.1` (Python 3.14.6)
* **Test File**: `backend/models/diabetes/tests/test_diabetes.py`
* **Test Results**: **32 passed, 0 failed**

Command to run the test suite:

```bash
python -m pytest backend/models/diabetes/tests/test_diabetes.py -v
```

## Limitations

* **Screening Estimate**: This model is a machine-learning component trained on tabular dataset measurements; predictions are model-generated screening estimates for the defined dataset classes and do not constitute a medical diagnosis.
* **Sample Size**: The dataset contains 768 total observations, with a 154-sample test set.
* **Sensitivity / False Negatives**: On the 154-sample test set, the false negative rate was 0.5000 (27 missed positive cases out of 54 actual Class 1 samples) under the default 0.50 classification threshold.
* **Input Sensitivity**: Input records with extreme or out-of-range measurements may affect prediction reliability.

## Integration Notes

Downstream services integrating with the diabetes model should adhere to the following:

* Supply all 8 features matching the names and order specified in `backend/models/diabetes/feature_schema.json`.
* Use `predict_diabetes` from `backend/models/diabetes/predict.py` for inference.
* Rely on `backend/models/diabetes/model.joblib` for pipeline execution.
* Ensure inputs contain valid numeric values without NaN or infinite values.
