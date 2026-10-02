# Parkinson's Disease Risk-Screening Model

## 1. Overview & Dataset Provenance
This module implements an AI-based risk screening component for Parkinson's disease based on biomedical acoustic measurements of sustained vowel phonations (`/a/`).

- **Dataset**: Oxford Parkinsons Disease Detection Dataset
- **Source**: UCI Machine Learning Repository, Dataset 174
- **Original Study**: Max A. Little, Patrick E. McSharry, Stephen J. Roberts, Declan A. E. Costello, Irene M. Moroz. *"Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection"*, BioMedical Engineering OnLine 2007, 6:23.
- **Total Records**: 195 phonation recordings
- **Independent Sampling Units**: 32 unique human subjects (24 diagnosed with Parkinson's disease, 8 healthy control subjects)
- **Recordings per Subject**: 6 to 7 phonations per subject

---

## 2. Target Variable
- **Target Column**: `status`
- **Encoding**:
  - `0`: Healthy control subject
  - `1`: Parkinson's disease patient
- **Distribution (Record-Level)**: 147 PD (75.38%), 48 Healthy (24.62%) — Imbalance ratio: ~3.06:1
- **Distribution (Subject-Level)**: 24 PD (75.00%), 8 Healthy (25.00%) — Imbalance ratio: 3.00:1

---

## 3. Predictor Features & Grouping Architecture

### Excluded Columns (Target & Leakage Prevention)
- `name`: Administrative recording identifier (e.g. `phon_R01_S01_1`). Excluded from predictor matrix $X$ to prevent identifier memorization.
- `subject_id`: Extracted token (`S01`, `S02`, etc.) used **strictly** as the grouping variable for patient-disjoint data splitting and cross-validation. Never passed into feature matrix $X$.
- `status`: Ground-truth target column.

### 22 Continuous Predictor Features
1. **Fundamental Frequency / Pitch**:
   - `MDVP:Fo(Hz)`: Average fundamental frequency
   - `MDVP:Fhi(Hz)`: Maximum fundamental frequency
   - `MDVP:Flo(Hz)`: Minimum fundamental frequency
2. **Frequency Perturbation (Jitter)**:
   - `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`
3. **Amplitude Perturbation (Shimmer)**:
   - `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`
4. **Noise Ratios**:
   - `NHR`: Noise-to-harmonics ratio
   - `HNR`: Harmonics-to-noise ratio
5. **Nonlinear Dynamical Complexity Measures**:
   - `RPDE`: Recurrence period density entropy
   - `DFA`: Detrended fluctuation analysis fractal exponent
   - `spread1`: Logarithmic fundamental frequency variation spread 1
   - `spread2`: Fundamental frequency variation spread 2
   - `D2`: Correlation dimension
   - `PPE`: Pitch period entropy

---

## 4. Patient-Level Grouped Validation Architecture

### Critical Safeguard Against Leakage
Each subject contributed 6–7 repeated recordings. A standard randomized train/test split or standard `StratifiedKFold` would place recordings from the **same patient** into both training and validation/test folds. Because individual vocal tract anatomy makes repeated phonations from the same speaker highly correlated, naive splitting allows models to memorize individual voices rather than learn disease-generalizable patterns, yielding artificially inflated metrics (>95%).

To guarantee zero subject leakage:
1. **Outer Holdout**: Held out 6 subjects (18.75% of subjects, 37 recordings: 2 Healthy [`S10`, `S50`], 4 PD [`S19`, `S20`, `S27`, `S39`]) strictly for final evaluation.
   - Verification: $\text{set}(\text{train\_subjects}) \cap \text{set}(\text{test\_subjects}) = \emptyset$.
2. **Inner Validation**: 5-Fold `StratifiedGroupKFold` applied exclusively to the 26 training subjects (158 recordings) with `groups=subject_id`.

---

## 5. Preprocessing Pipeline
- **Missing Values**: 0 missing values across all columns.
- **Scaling**: Scikit-learn `StandardScaler` embedded within a pipeline, fitted strictly on training subjects in each fold and applied to validation/test subjects.

---

## 6. Candidate Model Evaluation (5-Fold Grouped CV)

All models were evaluated strictly using `StratifiedGroupKFold` on the 26 training subjects:

| Model Family | Balanced Accuracy | ROC-AUC | PR-AUC | PD Recall | Healthy Specificity | F1 (PD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (C=0.1, balanced)** | **0.6363 $\pm$ 0.1862** | **0.8268 $\pm$ 0.1886** | **0.9384 $\pm$ 0.0803** | 0.7060 $\pm$ 0.1584 | **0.5667 $\pm$ 0.4655** | 0.7490 $\pm$ 0.0934 |
| **SVC (RBF kernel, C=1.0, balanced)** | 0.6360 $\pm$ 0.2130 | 0.6801 $\pm$ 0.2805 | 0.8836 $\pm$ 0.1116 | 0.7387 $\pm$ 0.1332 | 0.5333 $\pm$ 0.5055 | 0.7863 $\pm$ 0.0607 |
| **Random Forest (max_depth=4, balanced)**| 0.4638 $\pm$ 0.0884 | 0.7450 $\pm$ 0.1936 | 0.9206 $\pm$ 0.0821 | 0.8933 $\pm$ 0.0984 | 0.0833 $\pm$ 0.1443 | 0.8415 $\pm$ 0.0460 |
| **HistGradientBoosting (max_depth=3)** | 0.5343 $\pm$ 0.1349 | 0.7015 $\pm$ 0.1574 | 0.8683 $\pm$ 0.1506 | 0.9020 $\pm$ 0.1052 | 0.1667 $\pm$ 0.2887 | 0.8394 $\pm$ 0.0621 |
| **KNN (k=5, distance-weighted)** | 0.6050 $\pm$ 0.1454 | 0.7509 $\pm$ 0.1158 | 0.9082 $\pm$ 0.0435 | 0.9100 $\pm$ 0.1034 | 0.3000 $\pm$ 0.2981 | 0.8653 $\pm$ 0.0457 |

### Selection Rationale
`Logistic Regression` with $L_2$ regularization ($C=0.1$) was selected as the optimal model based strictly on grouped cross-validation:
1. **Highest Area Under Curves**: Highest ROC-AUC (0.8268) and highest PR-AUC (0.9384).
2. **Best Balanced Generalization**: Highest Balanced Accuracy (0.6363) and highest Healthy Specificity (0.5667), avoiding the majority-class collapse observed in tree ensembles on this small-sample dataset.
3. **Robustness to Multicollinearity**: Strong $L_2$ shrinkage stabilizes coefficients across the highly collinear jitter, shimmer, and entropy metrics.

---

## 7. Held-Out Test Set Performance

Evaluated exactly once on the quarantined 6 test subjects (37 recordings):

| Metric | Value |
| :--- | :--- |
| **Test Subjects** | 6 unique subjects (`['S10', 'S19', 'S20', 'S27', 'S39', 'S50']`) |
| **Test Recordings** | 37 (12 Healthy, 25 PD) |
| **Accuracy** | 75.68% (0.7568) |
| **Balanced Accuracy** | 71.17% (0.7117) |
| **ROC-AUC** | 0.8533 |
| **PR-AUC (Average Precision)** | 0.9281 |
| **PD Sensitivity / Recall** | 84.00% (21 / 25 recordings) |
| **Healthy Specificity** | 58.33% (7 / 12 recordings) |
| **Precision (PD)** | 80.77% (21 / 26 recordings) |
| **F1-Score (PD)** | 0.8235 |

### Test Confusion Matrix
```
                    Predicted Healthy (0)    Predicted Parkinson's (1)
Actual Healthy (0)            7                        5
Actual Parkinson's (1)        4                       21
```

---

## 8. Software-Defined Risk Tiers

The prediction module maps model output probabilities into software-defined heuristic risk tiers:

| Risk Tier | Probability Range | Intended UI Purpose |
| :--- | :--- | :--- |
| **Low** | $p < 0.30$ | Low statistical likelihood of phonatory perturbation |
| **Moderate**| $0.30 \le p < 0.70$ | Indeterminate / intermediate acoustic features; monitoring suggested |
| **High** | $p \ge 0.70$ | High statistical likelihood; priority specialist evaluation advised |

> **IMPORTANT DISCLAIMER**: These tiers are software-defined heuristic categories intended exclusively for visual presentation and screening prioritization. They are **NOT** clinically validated thresholds, medical severity classifications, or neurological diagnoses.

---

## 9. Limitations & Clinical Safety

1. **Independent Subject Count**:
   While the dataset contains 195 recordings, they stem from only 32 unique subjects. Statistical sample size is fundamentally determined by the 32 patients, meaning fold-to-fold variance is naturally high.
2. **Acoustic Phonation Scope**:
   Features capture sustained phonation of `/a/` in sound-treated settings. They do not capture continuous conversational speech, motor tremors, bradykinesia, rigidity, or cognitive symptoms.
3. **Absence of Demographic Controls**:
   Age and biological sex are not provided in the feature schema, preventing normalization for sex-linked pitch differences.
4. **Clinical Distinction**:
   This pipeline is an exploratory screening component designed to support clinical workflows. It is **not** an automated diagnostic system and must not replace evaluation by a licensed neurologist.
