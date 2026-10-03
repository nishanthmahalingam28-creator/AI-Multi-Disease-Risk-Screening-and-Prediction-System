# Stroke Disease Prediction Model

## 1. Title

**Stroke Disease Prediction Model**

This module provides the stroke risk screening and prediction model within the project's multi-disease risk screening and prediction system. It functions as one disease-specific component designed to produce an automated, AI-based screening/prediction estimate and a predicted class (Class 0 or Class 1) along with predicted probabilities for the defined dataset classes.

This component is designed solely for empirical risk screening research and statistical modeling. It does not diagnose stroke, does not provide medical conclusions, and does not establish clinical effectiveness.

---

## 2. Purpose

The module:

* accepts the defined stroke-related dataset features
* applies the saved preprocessing pipeline
* uses the trained machine-learning model
* returns a predicted class and predicted probability
* is intended as a model component for the larger AI screening/prediction system

All outputs from this model are structured as an "AI-based screening/prediction estimate" and a "model prediction". The module does not provide a confirmed diagnosis, medical diagnosis, or clinically validated diagnosis, and it does not classify any individual as a healthy patient or unhealthy patient, nor does it state that a patient definitely has stroke or definitely does not have stroke.

---

## 3. Dataset

The model utilizes the actual dataset located at:

`datasets/stroke/stroke.csv`

Dataset characteristics established during inspection:

* **Row Count**: 5,110 rows
* **Column Count**: 12 raw columns
* **Target Column**: `stroke`
* **Target Class Distribution**:
  * Class 0: 4,861 rows (95.13%)
  * Class 1: 249 rows (4.87%)
* **Class Ratio**: approximately 19.52:1 (severe class imbalance)
* **Missing Values**: 201 missing values, all located in the `bmi` column (3.93% of `bmi` entries)
* **Duplicate Rows**: 0 duplicate rows
* **Identifier Column**: `id` (5,110 unique integers across 5,110 rows)
* **Dataset SHA-256**: `aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815`

The `id` column is excluded from all predictive features because it is an arbitrary record identifier rather than a predictive input. No external source or license is asserted, as none is documented in the repository files.

---

## 4. Features

The model accepts exactly 10 predictive features in this exact order:

1. `gender` (categorical: `Female`, `Male`, `Other`)
2. `age` (continuous numeric: patient age in years)
3. `hypertension` (binary numeric: `0` or `1`)
4. `heart_disease` (binary numeric: `0` or `1`)
5. `ever_married` (categorical: `No`, `Yes`)
6. `work_type` (categorical: `Govt_job`, `Never_worked`, `Private`, `Self-employed`, `children`)
7. `Residence_type` (categorical: `Rural`, `Urban`)
8. `avg_glucose_level` (continuous numeric: blood glucose reading)
9. `bmi` (continuous numeric: body mass index)
10. `smoking_status` (categorical: `Unknown`, `formerly smoked`, `never smoked`, `smokes`)

**Target**: `stroke` (`0` for Class 0, `1` for Class 1)

Constraints:
* `id` is excluded from predictive features.
* `stroke` is the target and is not an input feature.
* No invented features were added.
* No feature engineering or synthetic feature construction was performed.

---

## 5. Exploratory Data Analysis

Exploratory data analysis on the raw dataset identified the following findings:

* **Class Imbalance**: 95.13% Class 0 vs. 4.87% Class 1 (imbalance ratio 19.52:1).
* **Missingness**: 201 missing values isolated strictly to `bmi`.
* **Duplicates**: 0 duplicate records.
* **Distributions**:
  * Continuous variables (`age`, `avg_glucose_level`, `bmi`) were examined; `avg_glucose_level` exhibits positive skewness (+1.57) with an extended high-value tail.
  * Categorical features were audited for cardinality and frequency, identifying rare representation for `gender` = `Other` (1 record) and widespread usage of `smoking_status` = `Unknown` (1,544 records, 30.22%).
* **Outliers**: Statistical IQR inspection identified 503 sample flags in `avg_glucose_level` (values up to 271.74) and 110 sample flags in `bmi` (values up to 97.60). These were retained as valid observational measurements.
* **Correlations**: Correlation analysis was performed for numeric features.

Notable linear correlations with the target `stroke`:
* `age` vs. `stroke`: approximately `0.2453`
* `heart_disease` vs. `stroke`: approximately `0.1349`
* `avg_glucose_level` vs. `stroke`: approximately `0.1319`
* `hypertension` vs. `stroke`: approximately `0.1279`
* `bmi` vs. `stroke`: approximately `0.0424`

These figures represent statistical dataset correlations only and must not be interpreted as medical causation or clinical importance.

EDA plots are saved under:
`backend/models/stroke/plots/`

---

## 6. Train/Test Split

The dataset was partitioned following a strict protocol:

* **Split Configuration**: Stratified 80% train / 20% test
* **Random State**: `random_state = 42`
* **Training Rows**: 4,088 samples
  * Class 0: 3,889 rows (95.13%)
  * Class 1: 199 rows (4.87%)
* **Test Rows**: 1,022 samples
  * Class 0: 972 rows (95.11%)
  * Class 1: 50 rows (4.89%)

The 1,022-row test partition was kept quarantined during all preprocessing design, candidate model comparison, cross-validation, and model selection. It was evaluated strictly once after model selection was completed.

---

## 7. Preprocessing

The preprocessing pipeline is implemented via a scikit-learn `ColumnTransformer` with three specialized pipelines:

1. **Continuous Features** (`age`, `avg_glucose_level`, `bmi`):
   * Median imputation: `SimpleImputer(strategy='median')`
   * Learned training medians: `age` = 45.0, `avg_glucose_level` = 91.945, `bmi` = 28.0
   * Scaling: `RobustScaler()` (centers by median and scales by IQR to accommodate positive skewness in glucose and BMI)
2. **Binary Features** (`hypertension`, `heart_disease`):
   * Median imputation: `SimpleImputer(strategy='median')`
   * Passthrough representation (unscaled binary values preserved)
3. **Categorical Features** (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`):
   * Most-frequent imputation: `SimpleImputer(strategy='most_frequent')`
   * One-hot encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

The resulting transformed feature representation has **21 columns**.

Operational rules:
* Preprocessing parameters were fitted exclusively on the 4,088 training rows.
* Test partition data was transformed using the fitted training preprocessor.
* The saved `model.joblib` artifact contains the complete end-to-end pipeline (preprocessing + classifier).
* Preprocessing choices reflect empirical data characteristics and are not described as clinically required.

---

## 8. Candidate Models and Cross-Validation

Five candidate architectures were evaluated using 5-fold StratifiedKFold (`shuffle=True`, `random_state=42`) on the 4,088-row training partition only, with fold-local preprocessing fitted independently inside each fold.

### Step 6 Baseline Cross-Validation (Unweighted, Threshold 0.50)

Because of the severe 19.54:1 class imbalance, unweighted models evaluated at threshold 0.50 exhibited near-zero positive-class recall:

| Model Architecture | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.9513 \pm 0.0005$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.8395 \pm 0.0105$ | $0.1923 \pm 0.0411$ |
| **Random Forest** | $0.9513 \pm 0.0009$ | $0.2667 \pm 0.3887$ | $0.0100 \pm 0.0122$ | $0.0191 \pm 0.0234$ | $0.8143 \pm 0.0201$ | $0.1678 \pm 0.0352$ |
| **SVC** | $0.9513 \pm 0.0005$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.6708 \pm 0.0300$ | $0.1173 \pm 0.0205$ |
| **Gradient Boosting** | $0.9499 \pm 0.0015$ | $0.3000 \pm 0.1871$ | $0.0201 \pm 0.0101$ | $0.0375 \pm 0.0188$ | $0.8426 \pm 0.0148$ | $0.1951 \pm 0.0304$ |
| **XGBoost** | $0.9494 \pm 0.0012$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.8431 \pm 0.0168$ | $0.2077 \pm 0.0724$ |

### Step 7 Imbalance-Aware Training-Only Cross-Validation (Threshold 0.50)

Imbalance-aware model variants were evaluated strictly on the training partition:

| Model Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (`class_weight='balanced'`)** | $0.7387 \pm 0.0097$ | $0.1329 \pm 0.0074$ | $0.7890 \pm 0.0337$ | $0.2274 \pm 0.0121$ | $0.8388 \pm 0.0112$ | $0.1935 \pm 0.0392$ |
| **Random Forest (`class_weight='balanced'`)** | $0.9366 \pm 0.0076$ | $0.1788 \pm 0.1359$ | $0.0701 \pm 0.0484$ | $0.0999 \pm 0.0706$ | $0.8172 \pm 0.0105$ | $0.1745 \pm 0.0426$ |
| **SVC (`class_weight='balanced'`)** | $0.7605 \pm 0.0101$ | $0.1257 \pm 0.0049$ | $0.6583 \pm 0.0431$ | $0.2110 \pm 0.0083$ | $0.7950 \pm 0.0090$ | $0.1553 \pm 0.0226$ |
| **XGBoost (`scale_pos_weight=19.5427`)** | $0.7723 \pm 0.0117$ | $0.1415 \pm 0.0074$ | $0.7237 \pm 0.0210$ | $0.2366 \pm 0.0107$ | $0.8304 \pm 0.0147$ | $0.1951 \pm 0.0412$ |

**Selection Basis**:
The selection record in `model_selection.json` documents the measured criteria used to select the final variant. Logistic Regression (`class_weight='balanced'`) demonstrated the highest mean cross-validation sensitivity/recall ($0.7890 \pm 0.0337$), detecting 157 of 199 positive stroke cases across the validation folds (reducing false negatives to 42), while sustaining a mean ROC-AUC of $0.8388 \pm 0.0112$ and PR-AUC of $0.1935 \pm 0.0392$.

---

## 9. Final Model

* **Selected Model**: `Logistic Regression`
* **Configuration**:
  ```text
  penalty = l2
  C = 1.0
  class_weight = balanced
  solver = lbfgs
  max_iter = 1000
  random_state = 42
  ```
* **Decision Threshold**: `0.50`

No test-set threshold tuning was performed. The model is an empirical machine-learning classifier and is not described as clinically interpretable.

---

## 10. Final Test Evaluation

The selected pipeline was trained on all 4,088 training samples and evaluated once on the quarantined 1,022-row test set.

* **Test Set Size**: 1,022 samples (Class 0: 972, Class 1: 50)

### Performance Metrics

| Metric | Value |
| :--- | :---: |
| Accuracy | 0.7446 |
| Precision | 0.1375 |
| Recall / Sensitivity | 0.8000 |
| F1-Score | 0.2346 |
| ROC-AUC | 0.8436 |
| PR-AUC | 0.2684 |

### Confusion Matrix

```text
TN = 721
FP = 251
FN = 10
TP = 40
```

* **False Positive Rate (FPR)**: `0.2582` (251 false positives out of 972 negative cases)
* **False Negative Rate (FNR)**: `0.2000` (10 false negatives out of 50 positive cases)

These values represent evaluation metrics on this dataset's held-out test partition only and do not represent real-world clinical performance.

---

## 11. Prediction Interface

Module file: `backend/models/stroke/predict.py`

### Usage Example

```python
from backend.models.stroke.predict import predict_one

sample = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

result = predict_one(sample)
print(result)
```

### Example Structured Output

```json
{
  "status": "success",
  "model_name": "Logistic_Regression",
  "predicted_class": 1,
  "predicted_probability": 0.8254,
  "class_probabilities": {
    "0": 0.1746,
    "1": 0.8254
  },
  "threshold": 0.5,
  "model_prediction": "Class 1"
}
```

The output contains the predicted class, predicted probability for Class 1, class probabilities dictionary, decision threshold, and neutral model prediction label. A prediction of Class 1 does not mean a confirmed stroke; it is an algorithmic model output.

---

## 12. Input Validation

The `predict.py` module enforces strict validation:

* Missing features are rejected (`ValueError`).
* Extra or unexpected features (such as `stroke` or `id`) are rejected (`ValueError`).
* `NaN`, null, and None values are rejected (`ValueError`).
* Infinite numeric values (`inf`, `-inf`) are rejected (`ValueError`).
* Invalid numeric types or non-numeric strings are rejected (`TypeError`).
* Empty input containers are rejected (`ValueError`).
* Multiple rows passed to `predict_one()` are rejected (`ValueError`).
* `pandas.Series` is supported.
* Single-row `pandas.DataFrame` is supported.
* Feature ordering is normalized to canonical order regardless of input dictionary key order.
* Caller's input object is not mutated.
* Repeated predictions with identical input are deterministic.

---

## 13. Artifacts

The following files constitute the stroke disease prediction model module:

```text
backend/models/stroke/
├── inspect_dataset.py
├── eda.py
├── eda_summary.json
├── define_features.py
├── feature_schema.json
├── split_data.py
├── split_summary.json
├── preprocessing.py
├── preprocessing_summary.json
├── preprocessor.joblib
├── test_preprocessing.py
├── candidate_cv.py
├── candidate_cv_results.json
├── final_evaluation.py
├── model.joblib
├── model_selection.json
├── metrics.json
├── predict.py
├── tests/
│   ├── __init__.py
│   └── test_stroke.py
└── plots/
```

All listed files are verified to exist in the repository workspace.

---

## 14. Testing

Automated testing was conducted using pytest:

```text
pytest -q backend/models/stroke/tests
```

**Results**:
```text
34 passed in 2.19s
```

* **Total Tests Collected**: 34
* **Passed**: 34
* **Failed**: 0
* **Skipped**: 0
* **Material Warnings**: None
* **Direct Root Import**: Verified (`from backend.models.stroke.predict import predict_one`)

The test suite systematically verifies artifact existence and loadability, schema definition, prediction interface contracts, invalid input rejection, determinism, input immutability, dataset hash integrity, independent metric consistency on the test split, confusion matrix internal counts, model selection documentation, and absence of model refitting.

---

## 15. Limitations

* This model was evaluated strictly on the available dataset and its held-out test partition.
* The dataset exhibits severe class imbalance (19.52:1 negative to positive ratio).
* The reported metrics are dataset-specific and do not establish clinical effectiveness.
* The model output is an algorithmic prediction/screening estimate, not a confirmed diagnosis.
* External validation was not performed in this workflow.
* The dataset source and license are not asserted beyond existing repository documentation.
* The standard 0.50 threshold was used and was not tuned on the test set.
* The model is an experimental software component and should not be described as medically validated.

---

## 16. Reproducibility

Reproducibility parameters applied across the pipeline:

* `random_state = 42` applied to dataset split, cross-validation splitting, and classifier initialization.
* Stratified 80/20 train/test partition protocol.
* 5-fold `StratifiedKFold` cross-validation with fold-local preprocessing.
* Saved preprocessing pipeline (`preprocessor.joblib`).
* Saved model pipeline artifact (`model.joblib`).
* Raw dataset SHA-256: `aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815`.
