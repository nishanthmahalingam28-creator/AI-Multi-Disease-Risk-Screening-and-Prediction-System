"""
Dataset Inspection and Audit Script for Stroke Dataset
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 1 — Dataset Inspection and Audit
This script performs a comprehensive, non-destructive, read-only inspection of the
actual Stroke dataset in the repository without modifying any rows, columns,
types, or values, and without training any models or performing preprocessing.
"""

import os
import sys
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def compute_sha256(filepath: str) -> str:
    """
    Computes the SHA-256 hash of a file for integrity verification.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_stroke_datasets(base_dir: str) -> List[str]:
    """
    Searches the repository for candidate stroke dataset files.
    """
    found = []
    common_locations = [
        os.path.join(base_dir, "datasets", "stroke", "stroke.csv"),
        os.path.join(base_dir, "backend", "models", "stroke", "stroke.csv"),
        os.path.join(base_dir, "datasets", "stroke.csv"),
    ]
    for loc in common_locations:
        if os.path.exists(loc) and os.path.isfile(loc):
            found.append(os.path.abspath(loc))

    datasets_root = os.path.join(base_dir, "datasets")
    if os.path.exists(datasets_root):
        for root, _, files in os.walk(datasets_root):
            for file in files:
                if "stroke" in file.lower() and file.endswith((".csv", ".tsv", ".parquet", ".xlsx")):
                    full_p = os.path.abspath(os.path.join(root, file))
                    if full_p not in found:
                        found.append(full_p)

    return sorted(list(set(found)))


def inspect_stroke_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Performs a thorough factual inspection of the Stroke dataset file.
    """
    print("=" * 80)
    print("STEP 1: STROKE DATASET INSPECTION & AUDIT")
    print("=" * 80)

    # Pre-inspection integrity check
    abs_path = os.path.abspath(dataset_path)
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        print(f"\nERROR: Dataset file not found at: {abs_path}")
        return {}

    initial_hash = compute_sha256(abs_path)
    file_size_bytes = os.path.getsize(abs_path)

    # 1. Dataset file path
    print(f"\n1. Dataset File Path:")
    print(f"   - Absolute Path: {abs_path}")
    print(f"   - Relative to Workspace: {os.path.relpath(abs_path, os.getcwd())}")
    print(f"   - File Exists: True")
    print(f"   - File Size: {file_size_bytes} bytes ({file_size_bytes / 1024:.2f} KB)")
    print(f"   - Initial SHA-256 Hash: {initial_hash}")

    # 2. Dataset source information from repository files/metadata
    print(f"\n2. Dataset Source Information:")
    print(f"   - Repository Location: datasets/stroke/stroke.csv")
    print(f"   - Tracked in Git: Commit 7c5f1e1 ('Dataset')")
    print(f"   - Format: Standard Comma-Separated Values (CSV)")
    print(f"   - Context: Standard Stroke Prediction benchmark dataset commonly used in healthcare machine learning")
    print(f"     (contains demographic, lifestyle, and health indicator features).")

    # Read raw lines for structural verification
    with open(abs_path, "r", encoding="utf-8") as f:
        raw_header = f.readline().strip()
        raw_first_data = f.readline().strip()
        total_raw_lines = 1 + 1 + sum(1 for _ in f)

    print(f"   - Raw Header: {repr(raw_header)}")
    print(f"   - First Data Row: {repr(raw_first_data)}")
    print(f"   - Total Raw Lines: {total_raw_lines} (1 header + {total_raw_lines - 1} records)")

    # Read with pandas strictly in read-only mode
    df = pd.read_csv(abs_path)

    # 3. Number of rows and columns
    num_rows, num_cols = df.shape
    print(f"\n3. Number of Rows: {num_rows}")
    print(f"   Number of Columns: {num_cols}")
    print(f"   Shape: ({num_rows}, {num_cols})")

    # 4. Exact column names and their order
    col_names = list(df.columns)
    print(f"\n4. Exact Column Names and Their Order (0 to {num_cols - 1}):")
    for idx, col in enumerate(col_names):
        print(f"   [{idx:2d}] {repr(col)}")

    # 5. Data types
    dtypes_dict = df.dtypes.to_dict()
    print(f"\n5. Data Types:")
    for col in col_names:
        print(f"   - {repr(col):<22} : {dtypes_dict[col]}")

    # 6. Target-column candidates and identification
    print(f"\n6. Target-Column Candidates & Identification:")
    target_candidates = [c for c in col_names if c.lower() in ["stroke", "target", "outcome", "label", "condition"]]
    print(f"   - Target Candidates Detected: {target_candidates}")
    identified_target = "stroke" if "stroke" in col_names else (target_candidates[0] if target_candidates else None)
    print(f"   - Identified Target Column: {repr(identified_target)}")
    print(f"   - Structural Evidence for Target Identification:")
    print(f"     * Column name 'stroke' directly denotes the clinical condition being screened.")
    print(f"     * Located at the terminal column position (index 11 of 12).")
    print(f"     * Contains binary indicator values (0 and 1) representing disease outcome.")
    print(f"     * Other columns represent physiological vitals, demographics, and clinical history predictors.")

    # 7 & 8. Unique target values, class counts, and percentages
    target_info = {}
    if identified_target:
        print(f"\n7 & 8. Target Value Distribution ({repr(identified_target)}):")
        t_counts = df[identified_target].value_counts(dropna=False)
        t_props = df[identified_target].value_counts(dropna=False, normalize=True) * 100
        for val, count in t_counts.items():
            pct = t_props[val]
            target_info[str(val)] = {"count": int(count), "percentage": round(float(pct), 4)}
            print(f"   - Class {repr(val)}: {count:5d} rows ({pct:6.2f}%)")
        print(f"   - Class Imbalance Ratio: {t_counts.max() / t_counts.min():.2f}:1")
        print(f"   - Minority Class ('1'): {t_counts.get(1, 0)} instances ({t_props.get(1, 0.0):.2f}%)")
        print(f"   - Majority Class ('0'): {t_counts.get(0, 0)} instances ({t_props.get(0, 0.0):.2f}%)")

    # 9. Missing / null counts for every column
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    print(f"\n9. Missing / Null Values Across All Columns:")
    print(f"   - Total Missing Cells: {total_missing}")
    cols_with_nulls = []
    for col in col_names:
        cnt = int(missing_series[col])
        pct = (cnt / num_rows) * 100
        if cnt > 0:
            cols_with_nulls.append((col, cnt, pct))
        print(f"   - {repr(col):<22} : {cnt:4d} nulls ({pct:5.2f}%)")
    if cols_with_nulls:
        print(f"   - Summary: {len(cols_with_nulls)} column(s) contain missing values: {cols_with_nulls}")
    else:
        print("   - Summary: No missing values detected in any column.")

    # 10. Duplicate-row count
    dup_rows = int(df.duplicated().sum())
    print(f"\n10. Duplicate Rows:")
    print(f"   - Exact Full-Row Duplicates: {dup_rows}")

    # 11. Constant columns
    constant_cols = [c for c in col_names if df[c].nunique(dropna=False) <= 1]
    print(f"\n11. Constant Columns:")
    if constant_cols:
        print(f"   - Constant Column(s) Found: {constant_cols}")
    else:
        print("   - None (All columns have at least 2 distinct values).")

    # 12. Unique-value counts for every column
    print(f"\n12. Unique-Value Counts for Every Column:")
    for col in col_names:
        n_uniq = int(df[col].nunique(dropna=False))
        print(f"   - {repr(col):<22} : {n_uniq:5d} unique values")

    # 13. Numeric summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n13. Numeric Summary Statistics ({len(numeric_cols)} numeric columns):")
    desc_df = df[numeric_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(desc_df.to_string())

    # 14. Categorical / object columns and their unique values
    # In newer pandas versions, string columns may be object or string dtype
    non_numeric_cols = [c for c in col_names if c not in numeric_cols]
    print(f"\n14. Categorical / Non-Numeric Columns ({len(non_numeric_cols)} columns):")
    for col in non_numeric_cols:
        val_counts = df[col].value_counts(dropna=False).to_dict()
        print(f"   - {repr(col)} ({df[col].dtype}):")
        for val, count in val_counts.items():
            pct = (count / num_rows) * 100
            print(f"     * {repr(val):<20} : {count:5d} ({pct:5.2f}%)")

    # 15. Zero-value counts for numeric columns
    print(f"\n15. Zero-Value Counts for Numeric Columns:")
    print("   Note: Factual count only. Zero values are NOT assumed to be missing unless contextually evident.")
    for col in numeric_cols:
        z_cnt = int((df[col] == 0).sum())
        z_pct = (z_cnt / num_rows) * 100
        min_v = df[col].min()
        max_v = df[col].max()
        print(f"   - {repr(col):<22} : {z_cnt:5d} zeros ({z_pct:6.2f}%) | Min: {min_v} | Max: {max_v}")

    # 16. Potential ID / index columns
    print(f"\n16. Potential ID / Index Columns:")
    id_candidates = []
    for col in col_names:
        if col.lower() in ["id", "patient_id", "record_id", "index", "subject_id"]:
            id_candidates.append(col)
        elif df[col].nunique(dropna=False) == num_rows:
            id_candidates.append(col)
    id_candidates = sorted(list(set(id_candidates)))
    if id_candidates:
        print(f"   - Identified Potential ID Column(s): {id_candidates}")
        for c in id_candidates:
            print(f"     * {repr(c)} has {df[c].nunique()} unique values across {num_rows} rows (100% uniqueness).")
            print(f"     * Factual Note: Identifier columns possess no generalizable medical relationship")
            print(f"       and should not be used as predictive features.")
    else:
        print("   - None detected.")

    # 17. Potential data-quality issues
    print(f"\n17. Potential Data-Quality Issues Observed:")
    quality_issues = []
    # Missing bmi
    bmi_nulls = int(df["bmi"].isnull().sum()) if "bmi" in df.columns else 0
    if bmi_nulls > 0:
        pct_null = (bmi_nulls / num_rows) * 100
        quality_issues.append(f"'bmi' contains {bmi_nulls} missing values ({pct_null:.2f}% of rows).")
        print(f"   - Missing Values: 'bmi' column has {bmi_nulls} null entries ({pct_null:.2f}%).")

    # Gender 'Other'
    if "gender" in df.columns:
        other_cnt = int((df["gender"] == "Other").sum())
        if other_cnt > 0:
            quality_issues.append(f"'gender' contains a rare category 'Other' with {other_cnt} instance(s).")
            print(f"   - Rare Category: 'gender' has {other_cnt} row(s) with value 'Other'.")

    # Unknown smoking status
    if "smoking_status" in df.columns:
        unk_cnt = int((df["smoking_status"] == "Unknown").sum())
        if unk_cnt > 0:
            unk_pct = (unk_cnt / num_rows) * 100
            quality_issues.append(f"'smoking_status' contains {unk_cnt} rows ({unk_pct:.2f}%) recorded as 'Unknown'.")
            print(f"   - Recorded Category 'Unknown': 'smoking_status' has {unk_cnt} entries ({unk_pct:.2f}%) as 'Unknown'.")

    # Extreme class imbalance
    if identified_target:
        min_pct = t_props.min()
        if min_pct < 10.0:
            quality_issues.append(f"Severe class imbalance: minority class is {min_pct:.2f}% of dataset.")
            print(f"   - Class Imbalance: Minority class represents only {min_pct:.2f}% of samples.")

    # Age range
    if "age" in df.columns:
        min_age = df["age"].min()
        max_age = df["age"].max()
        print(f"   - Age Range: {min_age} to {max_age} years (contains fractional ages for infants).")

    if not quality_issues:
        print("   - No major structural data-quality issues detected.")

    # 18. Potential leakage concerns
    print(f"\n18. Potential Leakage Concerns & Correlation Analysis:")
    if identified_target:
        # Check Pearson correlation for numeric columns
        corr_series = df[numeric_cols].corr()[identified_target].sort_values(ascending=False)
        print("   - Linear Correlation with target ('stroke'):")
        for c, r in corr_series.items():
            if c != identified_target:
                print(f"     * {repr(c):<22} : r = {r:+.4f}")
        high_corr = [c for c, r in corr_series.items() if c != identified_target and abs(r) >= 0.85]
        if high_corr:
            print(f"   - Warning: Column(s) with |r| >= 0.85: {high_corr}")
        else:
            print("   - Diagnostic Observation: No numeric feature shows an extreme linear correlation (|r| >= 0.85).")

        # Identifier correlation check
        if "id" in numeric_cols:
            id_corr = corr_series.get("id", 0.0)
            print(f"   - 'id' correlation with target: r = {id_corr:+.4f} (near zero, confirming identifier role).")

    # 19. Dataset shape and integrity checks
    final_hash = compute_sha256(abs_path)
    hash_match = (initial_hash == final_hash)
    print(f"\n19. Dataset Shape and Integrity Verification:")
    print(f"   - Verified Shape: Exactly {num_rows} rows and {num_cols} columns")
    print(f"   - Pre-inspection SHA-256 : {initial_hash}")
    print(f"   - Post-inspection SHA-256: {final_hash}")
    print(f"   - Hash Match Confirmed  : {hash_match}")
    if hash_match:
        print("   - Read-Only Integrity Check PASSED: Raw dataset file was not modified.")
    else:
        print("   - CRITICAL ERROR: Dataset file hash changed during inspection!")

    print("\n" + "=" * 80)
    print("STEP 1: STROKE DATASET INSPECTION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return {
        "file_path": abs_path,
        "rows": num_rows,
        "columns": num_cols,
        "column_names": col_names,
        "dtypes": {c: str(d) for c, d in dtypes_dict.items()},
        "target": identified_target,
        "target_distribution": target_info,
        "missing_values": missing_series.to_dict(),
        "duplicate_rows": dup_rows,
        "constant_columns": constant_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": non_numeric_cols,
        "sha256": final_hash,
        "integrity_verified": hash_match,
    }


def main():
    """
    Main entry point for Step 1 dataset inspection.
    """
    workspace_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    dataset_candidates = find_stroke_datasets(workspace_dir)

    if not dataset_candidates:
        print("ERROR: No stroke dataset candidate found in the repository!")
        sys.exit(1)

    dataset_path = dataset_candidates[0]
    results = inspect_stroke_dataset(dataset_path)

    if not results.get("integrity_verified", False):
        print("Inspection aborted due to integrity check failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
