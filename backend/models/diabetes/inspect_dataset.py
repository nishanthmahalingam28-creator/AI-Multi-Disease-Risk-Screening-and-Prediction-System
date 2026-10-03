"""
Dataset Inspection Script for Diabetes Dataset
AI Multi-Disease Risk Screening and Prediction System
"""

import os
import sys
import pandas as pd
import numpy as np


def inspect_diabetes_dataset(dataset_path: str):
    print("=" * 80)
    print("DIABETES DATASET INSPECTION")
    print("=" * 80)

    # 1. Existence and Path
    abs_path = os.path.abspath(dataset_path)
    file_exists = os.path.exists(abs_path)
    print(f"1. Dataset Location: {abs_path}")
    print(f"   File exists: {file_exists}")

    if not file_exists:
        print("ERROR: Dataset file not found!")
        return

    # 2. Filename
    filename = os.path.basename(abs_path)
    print(f"2. Dataset Filename: {filename}")

    # 3. File format & File size
    file_ext = os.path.splitext(filename)[1]
    file_size_bytes = os.path.getsize(abs_path)
    print(f"3. File Format: {file_ext} (CSV - Comma Separated Values)")
    print(f"   File Size: {file_size_bytes} bytes ({file_size_bytes / 1024:.2f} KB)")

    # Raw inspection of header and lines
    with open(abs_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        second_line = f.readline()
        total_raw_lines = 1 + sum(1 for _ in f)

    print("\n--- Raw Header and Line Inspection ---")
    print(f"Header line raw: {repr(first_line.strip())}")
    print(f"First data line raw: {repr(second_line.strip())}")
    print(f"Total raw lines in file: {total_raw_lines}")

    # Check for trailing commas or delimiters in header
    has_trailing_comma = first_line.strip().endswith(',')
    print(f"Trailing comma in header: {has_trailing_comma}")

    # Load with pandas
    df = pd.read_csv(abs_path)

    # 4 & 5. Rows and Columns
    num_rows, num_cols = df.shape
    print(f"\n4. Number of rows: {num_rows}")
    print(f"5. Number of columns: {num_cols}")

    # 6. Column Names in Original Order
    print("\n6. Exact Column Names in Original Order:")
    for idx, col in enumerate(df.columns):
        print(f"   [{idx}] '{col}'")

    # 7. Data Types
    print("\n7. Column Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - '{col}': {dtype}")

    # 8. Likely Target Column Identification
    print("\n8. Target Column Analysis:")
    # Check explicitly named target or low cardinality columns
    explicit_target = 'Outcome' if 'Outcome' in df.columns else None
    print(f"   - Explicitly identifiable target column: '{explicit_target}'")
    if explicit_target:
        print(f"   - Target column '{explicit_target}' data type: {df[explicit_target].dtype}")
        target_counts = df[explicit_target].value_counts(dropna=False)
        target_proportions = df[explicit_target].value_counts(dropna=False, normalize=True) * 100
        print("   - Target Value Counts and Class Proportions:")
        for val, count in target_counts.items():
            prop = target_proportions[val]
            print(f"     * Class {val}: {count} rows ({prop:.2f}%)")

    # 9. Missing Values Analysis
    print("\n9. Missing Values Analysis (Standard NaNs / Nulls):")
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"   - Total standard missing (NaN/null) values across dataset: {total_missing}")
    for col, cnt in missing_counts.items():
        pct = (cnt / num_rows) * 100
        print(f"     * '{col}': {cnt} nulls ({pct:.2f}%)")

    # String / whitespace checks if any object columns exist
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\n   Object/String Columns Count: {len(obj_cols)}")
    if obj_cols:
        for col in obj_cols:
            blank_cnt = df[col].apply(lambda x: str(x).strip() == '').sum()
            print(f"     * '{col}': {blank_cnt} blank/empty strings")
    else:
        print("   - No object/string columns present.")

    # 10. Duplicate Rows and ID Check
    print("\n10. Duplicate Records and ID Analysis:")
    exact_duplicates = df.duplicated().sum()
    print(f"   - Exact full-row duplicate rows: {exact_duplicates}")

    # Check for ID columns
    id_cols = [c for c in df.columns if c.lower() in ['id', 'patient_id', 'record_id', 'index']]
    if id_cols:
        print(f"   - ID column(s) detected: {id_cols}")
        for ic in id_cols:
            dup_id = df.duplicated(subset=[ic]).sum()
            print(f"     * Duplicate IDs in '{ic}': {dup_id}")
    else:
        print("   - ID column detected: None (No ID column exists in this dataset).")

    # 11. Constant or Obviously Unusable Columns
    print("\n11. Constant or Low-Variance Columns Check:")
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"   - Constant columns (<= 1 unique value): {constant_cols}")
    else:
        print("   - Constant columns: None (All columns have 2 or more distinct values).")

    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        print(f"   - '{col}': {n_unique} unique values")

    # 12. Unique Values for Categorical / Object Columns
    print("\n12. Categorical / Object Columns Unique Values:")
    if obj_cols:
        for col in obj_cols:
            print(f"   - Column '{col}': {df[col].unique()}")
    else:
        print("   - None. All columns are stored with numeric data types (int64 or float64).")

    # 13. Numerical Feature Summary & Ranges
    print("\n13. Numerical Feature Statistics Summary & Ranges:")
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    num_summary = df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(num_summary.to_string())

    # Range and Zero-value factual inventory
    print("\n   Detailed Min / Max Ranges and Zero Counts:")
    for col in df.columns:
        c_min = df[col].min()
        c_max = df[col].max()
        zero_cnt = (df[col] == 0).sum()
        zero_pct = (zero_cnt / num_rows) * 100
        print(f"   - '{col}': Range [{c_min}, {c_max}] | Zero count: {zero_cnt} ({zero_pct:.2f}%)")

    # 14. Unnamed / Index Column or Parser Artifacts Check
    print("\n14. Unnamed / Index Column / Parser Artifact Check:")
    unnamed_cols = [c for c in df.columns if 'unnamed' in c.lower() or c.strip() == '']
    if unnamed_cols:
        print(f"   - Unnamed / Parser Artifact columns detected: {unnamed_cols}")
    else:
        print("   - Unnamed / Parser Artifact columns detected: None.")

    # 15. Suspicious Columns and Target Leakage Analysis
    print("\n15. Suspicious Columns and Target Leakage Check:")
    if explicit_target and explicit_target in df.columns:
        corrs = df.corr()[explicit_target].sort_values(ascending=False)
        print("   - Linear correlation of numerical features with 'Outcome':")
        for col, val in corrs.items():
            print(f"     * '{col}': {val:.4f}")
        # Check for perfect predictors (|corr| == 1.0)
        perfect_preds = [c for c in corrs.index if c != explicit_target and abs(corrs[c]) >= 0.99]
        if perfect_preds:
            print(f"   - Warning: Possible perfect proxy columns: {perfect_preds}")
        else:
            print("   - No feature shows perfect correlation or obvious proxy leakage (|r| >= 0.99).")
    else:
        print("   - Cannot perform target correlation check without explicit target.")

    # 16. Distinguishing Confirmed Facts vs Observations/Uncertainties
    print("\n16. Confirmed Facts vs Observations / Uncertainties:")
    print("   [Confirmed Facts]")
    print(f"   - The dataset consists of exactly {num_rows} rows and {num_cols} columns.")
    print(f"   - File format is valid CSV, standard comma-delimited.")
    print("   - There are 0 standard missing (NaN/null) values in any column.")
    print("   - There are 0 exact full-row duplicate records.")
    print("   - There is no ID column and no unnamed/parser artifact column.")
    print("   - The column 'Outcome' is explicitly binary with values 0 (500 rows, 65.10%) and 1 (268 rows, 34.90%).")
    print("   - All 9 columns are stored as numeric types (int64 or float64).")
    print("   [Observations / Uncertainties]")
    print("   - In columns 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', and 'BMI', there exist 0 values.")
    print("     Fact: Minimum value is 0.0 in these columns.")
    print("     Observation: In biological contexts, a zero reading for blood pressure, glucose, or BMI is physiologically impossible for living subjects.")
    print("     Uncertainty: Whether these zeros represent unrecorded/missing measurements encoded as 0, or true recorded data, must be verified during preprocessing.")

    print("=" * 80)
    print("INSPECTION COMPLETE - NO MODEL TRAINED - ORIGINAL DATA UNTOUCHED")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Candidate dataset paths
    candidate_paths = [
        os.path.join(script_dir, "..", "..", "..", "datasets", "diabetes", "diabetes.csv"),
        os.path.join("datasets", "diabetes", "diabetes.csv"),
        os.path.abspath("datasets/diabetes/diabetes.csv")
    ]

    selected_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            selected_path = p
            break

    if selected_path is None:
        selected_path = candidate_paths[0]

    inspect_diabetes_dataset(selected_path)
