# Lung Cancer Risk-Screening Model

## 1. Dataset Overview
- **Dataset File**: `datasets/lung_cancer/survey_lung_cancer.csv`
- **Raw Dimensions**: 309 rows × 16 columns
- **Preprocessed Dimensions**: 276 rows × 16 columns (after documented deduplication)
- **Missing Values**: 0 (0.00% missing values across all columns)
- **Provenance Notice**: The dataset is a survey-based questionnaire capturing demographic information, self-reported symptoms, and lifestyle factors. No formal institutional metadata or source clinical papers are packaged within the repository.

---

## 2. Features and Target

### Target Variable
- **Raw Header**: `LUNG_CANCER`
- **Standardized Name**: `lung_cancer`
- **Classes**:
  - `YES` (`1`): Positive risk indicator (270 raw / 238 post-deduplication, ~86.23%)
  - `NO` (`0`): Negative risk indicator (39 raw / 38 post-deduplication, ~13.77%)

### Predictor Features (15 Total)
| Feature Name | Raw Header | Data Type | Encoding / Scaling | Description |
|---|---|---|---|---|
| `gender` | `GENDER` | Categorical | `MALE` $\rightarrow$ 1, `FEMALE` $\rightarrow$ 0 | Patient biological sex |
| `age` | `AGE` | Numerical | `StandardScaler` (fit on train only) | Age in years (range 21–87) |
| `smoking` | `SMOKING` | Binary | Passthrough `{0, 1}` | Smoking history |
| `yellow_fingers` | `YELLOW_FINGERS` | Binary | Passthrough `{0, 1}` | Nicotine staining on fingers |
| `anxiety` | `ANXIETY` | Binary | Passthrough `{0, 1}` | Chronic anxiety / nervousness |
| `peer_pressure` | `PEER_PRESSURE` | Binary | Passthrough `{0, 1}` | Peer pressure influencing lifestyle |
| `chronic_disease` | `CHRONIC DISEASE` | Binary | Passthrough `{0, 1}` | Underlying chronic illness |
| `fatigue` | `FATIGUE ` *(trailing space)* | Binary | Passthrough `{0, 1}` | Persistent unexplained fatigue |
| `allergy` | `ALLERGY ` *(trailing space)* | Binary | Passthrough `{0, 1}` | History of allergies / hypersensitivity |
| `wheezing` | `WHEEZING` | Binary | Passthrough `{0, 1}` | Wheezing respiratory sounds |
| `alcohol_consuming` | `ALCOHOL CONSUMING` | Binary | Passthrough `{0, 1}` | Regular alcohol consumption |
| `coughing` | `COUGHING` | Binary | Passthrough `{0, 1}` | Persistent cough |
| `shortness_of_breath` | `SHORTNESS OF BREATH` | Binary | Passthrough `{0, 1}` | Dyspnea / breathlessness |
| `swallowing_difficulty` | `SWALLOWING DIFFICULTY` | Binary | Passthrough `{0, 1}` | Dysphagia / difficulty swallowing |
| `chest_pain` | `CHEST PAIN` | Binary | Passthrough `{0, 1}` | Chest pain / tightness |

---

## 3. Data Preprocessing & Sanitization
1. **Header Sanitization**: Raw headers containing trailing spaces (`'FATIGUE '`, `'ALLERGY '`) and internal spaces (`'CHRONIC DISEASE'`, `'ALCOHOL CONSUMING'`, etc.) are standardized to clean lowercase `snake_case`.
2. **Encoding**:
   - `gender`: Mapped to binary `{0, 1}`.
   - `lung_cancer`: Mapped to binary `{0, 1}`.
3. **Scaling**: Continuous variable `age` is scaled using `StandardScaler` fitted strictly on the training partition.
4. **No Target Leakage**: The preprocessor is fit solely on feature matrix $X_{train}$ without access to target labels.

---

## 4. Duplicate & Conflicting-Label Handling Policy
- **Exact Full-Row Duplicates**:
  - The raw dataset contained 33 exact duplicate records (10.68%).
  - **Policy**: All 33 duplicate records were removed at ingestion prior to partitioning ($309 \rightarrow 276$ rows).
  - **Rationale**: Retaining identical records across a randomized train/test split causes identical observations to leak into both partitions, producing artificially inflated evaluation metrics.
- **Conflicting-Label Profile**:
  - One feature combination (64-year-old male with smoking, allergy, wheezing, alcohol, coughing, chest pain, but negative for yellow fingers, anxiety, peer pressure, chronic disease, fatigue, shortness of breath, and swallowing difficulty) had contradictory target outcomes:
    - Present with `LUNG_CANCER = YES` (3 instances in raw data, 1 in deduplicated data)
    - Present with `LUNG_CANCER = NO` (1 instance)
  - **Policy**: Both instances are preserved in the dataset because they represent legitimate clinical outcome divergence for identical questionnaire symptom profiles.

---

## 5. Partitioning & Cross-Validation Strategy
- **Train / Test Split**: 80% Train ($N = 220$), 20% Test ($N = 56$).
- **Partitioning Method**: `train_test_split(..., stratify=y, random_state=42)`.
- **Untouched Test Set**: The test set ($N=56$; 48 positive, 8 negative) remained completely quarantined until final evaluation.
- **Cross-Validation**: 5-Fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` applied to $X_{train}$. The ColumnTransformer preprocessor was fitted independently inside each fold to guarantee zero leakage.

---

## 6. Candidate Model Exploration & Comparison

Four candidate estimators with balanced class weighting were evaluated across 5 stratified folds:

| Candidate Model | CV Accuracy | CV Balanced Acc | CV F1 (YES=1) | CV ROC-AUC | CV PR-AUC | CV Minority Recall (NO=0) |
|---|---|---|---|---|---|---|
| **Logistic Regression (balanced)** | 0.8409 ± 0.0431 | **0.8096 ± 0.0425** | 0.9017 ± 0.0295 | **0.9298 ± 0.0220** | **0.9888 ± 0.0035** | **0.7667 ± 0.0816** |
| **Random Forest (balanced)** *(Selected)* | **0.8955 ± 0.0369** | 0.7570 ± 0.0637 | **0.9395 ± 0.0225** | 0.9035 ± 0.0356 | 0.9843 ± 0.0057 | 0.5667 ± 0.1333 |
| **HistGradientBoosting (balanced)** | 0.8773 ± 0.0340 | 0.7465 ± 0.1024 | 0.9285 ± 0.0209 | 0.8421 ± 0.0961 | 0.9678 ± 0.0220 | 0.5667 ± 0.2261 |
| **Support Vector Classifier (balanced)** | 0.8591 ± 0.0485 | 0.7781 ± 0.1128 | 0.9159 ± 0.0289 | 0.8939 ± 0.0567 | 0.9827 ± 0.0095 | 0.6667 ± 0.2108 |

---

## 7. Model Selection Rationale & Trade-off Analysis
- **Selected Model**: `RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=4, random_state=42)`
- **Detailed Metric Trade-offs**:
  Cross-validation comparisons demonstrated clear trade-offs between the top two candidate models:
  - **Logistic Regression (balanced)** demonstrated superior **Balanced Accuracy (0.8096 ± 0.0425 vs 0.7570 ± 0.0637)**, higher **ROC-AUC (0.9298 ± 0.0220 vs 0.9035 ± 0.0356)**, higher **PR-AUC (0.9888 ± 0.0035 vs 0.9843 ± 0.0057)**, and significantly higher **minority class recall for negative controls (0.7667 ± 0.0816 vs 0.5667 ± 0.1333 for NO=0)**.
  - **Random Forest (balanced)** achieved higher **overall F1-score for positive risk (0.9395 ± 0.0225 vs 0.9017 ± 0.0295)**, higher **overall accuracy (0.8955 ± 0.0369 vs 0.8409 ± 0.0431)**, and lower false positive rate on the majority class.
- **Selection Decision**:
  Random Forest was selected for this disease model baseline because:
  1. It captures non-linear symptom co-occurrences (e.g. combined coughing, allergy, and swallowing difficulty) without requiring explicit interaction terms.
  2. The depth restriction (`max_depth=4`) regularizes the ensemble, successfully preventing overfitting while maximizing positive-class screening F1-score.
  3. **Important Note**: Random Forest did *not* outperform Logistic Regression on all metrics. In settings where identifying the minority negative class (`NO=0`) is prioritized over overall F1, Logistic Regression remains a compelling, highly competitive linear alternative.

---

## 8. Final Evaluation on Untouched Test Set ($N = 56$)
- **Accuracy**: **94.64%** (53 / 56 correct)
- **Balanced Accuracy**: **91.67%**
- **ROC-AUC**: **0.9740**
- **PR-AUC (Average Precision)**: **0.9961**
- **Majority Class (`YES = 1`)**:
  - Precision: **0.9787**
  - Recall: **0.9583** (46 / 48 cases detected)
  - F1-Score: **0.9684**
- **Minority Class (`NO = 0`)**:
  - Precision: **0.7778**
  - Recall: **0.8750** (7 / 8 negative controls correctly identified)
  - F1-Score: **0.8235**
- **Confusion Matrix**:
  ```text
  [[TN=7,  FP=1],
   [FN=2,  TP=46]]
  ```

---

## 9. Top Feature Importances (Random Forest)
1. `allergy`: 0.1699
2. `alcohol_consuming`: 0.1146
3. `peer_pressure`: 0.0959
4. `swallowing_difficulty`: 0.0772
5. `age`: 0.0748
6. `fatigue`: 0.0618
7. `wheezing`: 0.0603
8. `chronic_disease`: 0.0593
9. `chest_pain`: 0.0577
10. `yellow_fingers`: 0.0572

---

## 10. Model Artifacts & Architecture
- [`model.joblib`](model.joblib): Serialized trained `RandomForestClassifier`.
- [`preprocessor.joblib`](preprocessor.joblib): Serialized `ColumnTransformer` with `StandardScaler` fitted on training age.
- [`feature_schema.json`](feature_schema.json): Specification of the 15 input features, value bounds, and column mappings.
- [`metrics.json`](metrics.json): Complete machine-readable record of 5-fold CV metrics and test set performance.
- [`predict.py`](predict.py): Stateless inference interface with input schema validation.
  - **Risk Tier Definition**: The risk levels assigned by the prediction interface (`Low`, `Moderate`, `High`) represent heuristic software categorization rules (<0.35 = Low, 0.35–0.70 = Moderate, ≥0.70 = High) for UI display. They are **not** clinically validated diagnostic thresholds.
- [`tests/test_lung_cancer.py`](tests/test_lung_cancer.py): 9 comprehensive automated tests.

---

## 11. Known Limitations
1. **Sample Size**: Small cohort size ($N = 309$ raw, $N = 276$ deduplicated).
2. **Severe Class Imbalance**: High proportion of positive cancer cases (~86.2%) suggests clinical referral bias.
3. **Survey Self-Reported Metrics**: Lack of quantitative clinical biomarkers (e.g., pack-years, histopathology, molecular biomarkers like EGFR/KRAS, or CT imaging).

---

## 12. Clinical Screening Disclaimer
> **IMPORTANT MEDICAL NOTICE**:
> This machine learning model is strictly an **exploratory risk-screening questionnaire tool** designed to flag potential risk for further medical investigation. It does **NOT** provide a medical diagnosis, pathology assessment, or definitive confirmation of lung cancer. All predictions must be evaluated by licensed medical professionals alongside diagnostic radiology and clinical pathology.
