"""
Dataset Inspection Script for Breast Cancer Dataset
Member 1: AI Multi-Disease Risk Screening and Prediction System
"""

import os
import sys
import json
import pandas as pd
import numpy as np

def inspect_breast_cancer_dataset(dataset_path: str):
    print("=" * 80)
    print("BREAST CANCER DATASET INSPECTION")
    print("=" * 80)
    
    # 1. Existence and Path
    abs_path = os.path.abspath(dataset_path)
    print(f"1. Dataset Location: {abs_path}")
    print(f"   File exists: {os.path.exists(abs_path)}")
    
    if not os.path.exists(abs_path):
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
    
    # Raw inspection of header
    with open(abs_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        second_line = f.readline()
    print("\n--- Raw Header Inspection ---")
    print(f"Header line raw: {repr(first_line.strip())}")
    print(f"First data line raw: {repr(second_line.strip())}")

    # Load with pandas
    df = pd.read_csv(abs_path)
    
    # 4 & 5. Rows and Columns
    num_rows, num_cols = df.shape
    print(f"\n4. Number of rows: {num_rows}")
    print(f"5. Number of columns: {num_cols}")
    
    # 6. Column Names
    print("\n6. Column Names:")
    for idx, col in enumerate(df.columns):
        print(f"   [{idx:02d}] '{col}'")
        
    # 7. Data Types
    print("\n7. Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col}: {dtype}")
        
    # 8. Candidate Target Column(s)
    # Check categorical / object columns or columns with few unique values
    candidate_targets = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 5 or df[col].dtype == 'object':
            candidate_targets.append((col, len(unique_vals), unique_vals[:10]))
    print("\n8. Candidate Target Column(s) based on actual data:")
    for col, n_u, sample_u in candidate_targets:
        print(f"   - Candidate: '{col}' | Unique count: {n_u} | Values: {sample_u}")
        
    # 9. Missing Values
    print("\n9. Missing Values:")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if missing_cols.empty:
        print("   No standard NaN / null values found in any column.")
    else:
        for col, cnt in missing_cols.items():
            pct = (cnt / num_rows) * 100
            print(f"   - {col}: {cnt} missing ({pct:.2f}%)")
            
    # Check for empty string or whitespace in object columns
    print("\n   Checking for blank/whitespace strings in object columns:")
    for col in df.select_dtypes(include=['object']).columns:
        blank_cnt = df[col].apply(lambda x: str(x).strip() == '').sum()
        if blank_cnt > 0:
            print(f"   - {col}: {blank_cnt} blank/empty strings")
        else:
            print(f"   - {col}: 0 blank strings")

    # 10. Duplicate Records
    exact_duplicates = df.duplicated().sum()
    print(f"\n10. Duplicate Records:")
    print(f"   - Exact full-row duplicates: {exact_duplicates}")
    if 'id' in df.columns:
        id_duplicates = df.duplicated(subset=['id']).sum()
        print(f"   - Duplicate 'id' values: {id_duplicates}")
        feature_cols_only = [c for c in df.columns if c not in ['id']]
        feature_duplicates = df.duplicated(subset=feature_cols_only).sum()
        print(f"   - Duplicates ignoring 'id' column: {feature_duplicates}")

    # 11. Target Class Distribution
    print("\n11. Class Distribution of Candidate Target(s):")
    for target_col in ['diagnosis']:
        if target_col in df.columns:
            vc = df[target_col].value_counts(dropna=False)
            vp = df[target_col].value_counts(dropna=False, normalize=True) * 100
            print(f"   Target '{target_col}':")
            for val, cnt in vc.items():
                print(f"     - {val}: {cnt} ({vp[val]:.2f}%)")
                
    # 12. Unique Values for Categorical / Object Columns
    print("\n12. Categorical / Non-numeric Column Unique Values:")
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        u_vals = df[col].unique()
        print(f"   - Column '{col}': {len(u_vals)} unique values -> {u_vals}")

    # 13. Numerical Feature Statistics
    num_cols_df = df.select_dtypes(include=[np.number])
    print(f"\n13. Numerical Feature Statistics ({len(num_cols_df.columns)} numerical columns):")
    stats_df = num_cols_df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    # Format and print
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    print(stats_df.to_string())

    # 14. Suspicious or Invalid Values
    print("\n14. Suspicious or Invalid Values Check:")
    # Check for NaN / Inf
    inf_cols = {}
    for col in num_cols_df.columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_cols[col] = inf_count
    print(f"   - Columns with Infinite values: {inf_cols if inf_cols else 'None'}")
    
    # Check for negative values in biological/morphological measurements
    neg_cols = {}
    for col in num_cols_df.columns:
        if col != 'id':
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                neg_cols[col] = neg_count
    print(f"   - Morphological columns with negative values: {neg_cols if neg_cols else 'None'}")
    
    # Check for zero values in morphological columns
    zero_cols = {}
    for col in num_cols_df.columns:
        if col != 'id':
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                zero_cols[col] = zero_count
    print(f"   - Morphological columns with zero values:")
    for col, z_cnt in zero_cols.items():
        print(f"     * {col}: {z_cnt} rows ({z_cnt/num_rows*100:.2f}%) have value == 0.0")

    # Check for trailing empty column often named "Unnamed: 32"
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
    print(f"   - Unnamed columns: {unnamed_cols}")
    for col in unnamed_cols:
        print(f"     * {col}: non-null count = {df[col].notnull().sum()}, all null = {df[col].isnull().all()}")

    # 15. Possible Data Leakage Columns
    print("\n15. Data Leakage Analysis:")
    # Check correlations or identifying columns
    print("   - 'id': Patient identification number. Arbitrary key, non-clinical, risk of memorization/leakage.")
    # Check if any feature perfectly predicts or has high correlation with target
    if 'diagnosis' in df.columns:
        y_binary = df['diagnosis'].map({'M': 1, 'B': 0})
        corr_series = num_cols_df.apply(lambda s: s.corr(y_binary) if s.std() > 0 else np.nan)
        print("   - Correlation with target (diagnosis = M):")
        top_corr = corr_series.dropna().sort_values(ascending=False)
        print("     Top 5 positively correlated:")
        for col, cval in top_corr.head(5).items():
            print(f"       * {col}: {cval:.4f}")
        print("     Lowest / Top negative correlated:")
        for col, cval in top_corr.tail(5).items():
            print(f"       * {col}: {cval:.4f}")

    # 16. Potential Exclusions
    print("\n16. Potential Exclusions & Rationale:")
    if 'id' in df.columns:
        print("   - Column 'id': Exclude. It is an arbitrary patient ID identifier, not a physiological or clinical feature.")
    if unnamed_cols:
        for c in unnamed_cols:
            print(f"   - Column '{c}': Exclude. Generated due to trailing comma in CSV, contains 100% missing values.")

    # 17. Suitability for Binary Classification
    print("\n17. Suitability for Binary Classification Screening Model:")
    print("   - Task: Distinguish Benign (B) from Malignant (M) breast masses based on cell nucleus characteristics from FNA (Fine Needle Aspirate).")
    print(f"   - Sample size: {num_rows} records.")
    print("   - Features: Real-valued morphological measurements of cell nuclei.")
    print("   - Suitability: Highly suitable for binary classification.")
    print("=" * 80)

if __name__ == "__main__":
    dataset_file = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "datasets", "breast_cancer", "breast.csv"
    )
    # Also support direct relative path or workspace path
    if not os.path.exists(dataset_file):
        dataset_file = os.path.join("datasets", "breast_cancer", "breast.csv")
    inspect_breast_cancer_dataset(dataset_file)
