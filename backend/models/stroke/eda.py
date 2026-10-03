"""
Exploratory Data Analysis (EDA) Script for Stroke Prediction Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Stroke Prediction Model

Step 2 — Exploratory Data Analysis
This script performs a purely descriptive, non-destructive exploratory data analysis
of the actual Stroke dataset without modifying any rows, columns, or values,
and without training any model or performing preprocessing.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt

# Canonical dataset path
DATASET_PATH = os.path.join("datasets", "stroke", "stroke.csv")

# Feature classifications based on Step 1 dataset inspection
ALL_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]

CONTINUOUS_NUMERIC_COLUMNS = ["age", "avg_glucose_level", "bmi"]
DISCRETE_NUMERIC_COLUMNS = ["hypertension", "heart_disease"]
CATEGORICAL_COLUMNS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
IDENTIFIER_COLUMNS = ["id"]
TARGET_COLUMN = "stroke"

# Plot styling palette
COLOR_CLASS_0 = "#2b5c8f"    # Deep Slate / Navy
COLOR_CLASS_1 = "#d9534f"    # Crimson / Coral Red
COLOR_PRIMARY = "#1f77b4"
COLOR_ACCENT = "#2ca02c"
COLOR_MUTED = "#6c757d"
COLOR_BOX_FILL = "#d0e1f9"
COLOR_BOX_LINE = "#1c3b70"


def compute_sha256(filepath: str) -> str:
    """
    Computes the SHA-256 hash of a file for integrity verification.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_iqr_outliers(series: pd.Series) -> Dict[str, Any]:
    """
    Computes IQR-based statistical outlier bounds and counts for a numeric series.
    Note: These are statistical diagnostic flags, not clinical or quality defects.
    """
    clean_s = series.dropna()
    q1 = float(clean_s.quantile(0.25))
    q3 = float(clean_s.quantile(0.75))
    iqr = float(q3 - q1)
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)
    outliers = clean_s[(clean_s < lower_bound) | (clean_s > upper_bound)]
    count = int(len(outliers))
    pct = float((count / len(clean_s)) * 100) if len(clean_s) > 0 else 0.0
    return {
        "valid_count": int(len(clean_s)),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_count": count,
        "outlier_percentage": round(pct, 4)
    }


def run_eda(dataset_path: str, output_dir: str):
    """
    Executes comprehensive EDA and exports eda_summary.json and visual plots.
    """
    print("=" * 80)
    print("STEP 2: STROKE DATASET EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # Resolve paths
    abs_dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(abs_dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {abs_dataset_path}")

    # Pre-EDA integrity hash
    pre_sha256 = compute_sha256(abs_dataset_path)
    print(f"Pre-EDA Dataset SHA-256 : {pre_sha256}")

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "eda_summary.json")

    # Read original dataset in read-only manner
    df = pd.read_csv(abs_dataset_path)
    num_rows, num_cols = df.shape

    print(f"1. Dataset Shape: {num_rows} rows x {num_cols} columns")

    # 2. Target Distribution
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    target_props = (df[TARGET_COLUMN].value_counts(dropna=False, normalize=True) * 100).to_dict()
    c0_count = int(target_counts.get(0, 0))
    c1_count = int(target_counts.get(1, 0))
    c0_pct = round(float(target_props.get(0, 0.0)), 4)
    c1_pct = round(float(target_props.get(1, 0.0)), 4)
    imbalance_ratio = round(c0_count / c1_count, 4) if c1_count > 0 else None

    target_dist_summary = {
        "target_column": TARGET_COLUMN,
        "classes": {
            "0": {"count": c0_count, "percentage": c0_pct, "label": "No Stroke (Negative)"},
            "1": {"count": c1_count, "percentage": c1_pct, "label": "Stroke (Positive)"}
        },
        "imbalance_ratio": imbalance_ratio,
        "imbalance_classification": "severe_class_imbalance"
    }

    print(f"\n2. Target Distribution ({TARGET_COLUMN}):")
    print(f"   - Class 0 (No Stroke): {c0_count:5d} rows ({c0_pct:5.2f}%)")
    print(f"   - Class 1 (Stroke)   : {c1_count:5d} rows ({c1_pct:5.2f}%)")
    print(f"   - Imbalance Ratio    : {imbalance_ratio:.2f}:1")

    # 3. Missing-Value Distribution
    missing_dict = df.isnull().sum().to_dict()
    missing_pct_dict = {c: round(float((cnt / num_rows) * 100), 4) for c, cnt in missing_dict.items()}
    total_missing = sum(missing_dict.values())
    missing_summary = {
        "total_missing_cells": total_missing,
        "columns_with_missing_values": [c for c, cnt in missing_dict.items() if cnt > 0],
        "missing_counts_per_column": missing_dict,
        "missing_percentages_per_column": missing_pct_dict,
        "missing_mechanism_note": (
            "Column 'bmi' contains 201 missing values (3.93%). "
            "All other 11 columns contain 0 missing values."
        )
    }

    print(f"\n3. Missing-Value Analysis:")
    print(f"   - Total missing cells across entire dataset: {total_missing}")
    for col, cnt in missing_dict.items():
        if cnt > 0:
            print(f"   - {col:<20} : {cnt} missing ({missing_pct_dict[col]:.2f}%)")
        else:
            print(f"   - {col:<20} : 0 missing (0.00%)")

    # 4. Duplicate Rows
    dup_rows = int(df.duplicated().sum())
    print(f"\n4. Duplicate Rows: Total = {dup_rows}")

    # 5. Descriptive Statistics for Continuous and Discrete Numeric Features
    eval_numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
    desc_stats = {}
    print("\n5. Descriptive Statistics for Numeric Features:")
    for col in eval_numeric_cols:
        s = df[col]
        desc_stats[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "25%": round(float(s.quantile(0.25)), 4),
            "50%_median": round(float(s.median()), 4),
            "75%": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4)
        }
        st = desc_stats[col]
        print(f"   - {col:<20} | Count: {st['count']} | Mean: {st['mean']:>8.2f} | Std: {st['std']:>7.2f} | Median: {st['50%_median']:>6.2f} | Range: [{st['min']}, {st['max']}]")

    # 6. Skewness Analysis
    skew_analysis = {}
    print("\n6. Skewness Analysis for Numeric Variables:")
    for col in eval_numeric_cols:
        sk = float(df[col].skew())
        abs_sk = abs(sk)
        skew_class = (
            "highly_skewed" if abs_sk > 1.0
            else ("moderately_skewed" if abs_sk >= 0.5 else "approximately_symmetric")
        )
        skew_analysis[col] = {
            "skewness": round(sk, 4),
            "skew_classification": skew_class
        }
        print(f"   - {col:<20} : {sk:>+7.4f} ({skew_class})")

    # 7. Outlier Analysis (IQR Method)
    outlier_analysis = {}
    print("\n7. Statistical Outlier Analysis (IQR Method, Diagnostic Only):")
    for col in CONTINUOUS_NUMERIC_COLUMNS:
        res = compute_iqr_outliers(df[col])
        outlier_analysis[col] = res
        print(f"   - {col:<20} : {res['outlier_count']:4d} outliers ({res['outlier_percentage']:5.2f}%) | Bounds: [{res['lower_bound']}, {res['upper_bound']}]")

    # 8. Categorical Feature Distributions
    cat_summary = {}
    print("\n8. Categorical Feature Distributions:")
    for col in CATEGORICAL_COLUMNS:
        counts = df[col].value_counts(dropna=False).to_dict()
        props = (df[col].value_counts(dropna=False, normalize=True) * 100).to_dict()
        cat_summary[col] = {
            val: {
                "count": int(counts[val]),
                "percentage": round(float(props[val]), 4)
            }
            for val in counts
        }
        print(f"   - {col} (distinct values: {len(counts)}):")
        for val, st in cat_summary[col].items():
            print(f"     * {repr(val):<20} : {st['count']:5d} ({st['percentage']:5.2f}%)")

    # 9. Correlation Analysis (Numeric Features & Target)
    corr_numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
    corr_matrix = df[corr_numeric_cols].corr()
    corr_matrix_dict = {
        row_c: {col_c: round(float(corr_matrix.loc[row_c, col_c]), 4) for col_c in corr_numeric_cols}
        for row_c in corr_numeric_cols
    }

    # Top pairwise feature-to-feature correlations (excluding target)
    predictor_numeric = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    pairwise_corrs = []
    for i in range(len(predictor_numeric)):
        for j in range(i + 1, len(predictor_numeric)):
            c1, c2 = predictor_numeric[i], predictor_numeric[j]
            r = float(corr_matrix.loc[c1, c2])
            pairwise_corrs.append({
                "feature_1": c1,
                "feature_2": c2,
                "pearson_r": round(r, 4),
                "abs_r": round(abs(r), 4)
            })
    pairwise_corrs.sort(key=lambda x: x["abs_r"], reverse=True)

    print("\n9. Feature-to-Feature Correlations (Top pairs):")
    for p in pairwise_corrs:
        print(f"   - {p['feature_1']} vs {p['feature_2']}: r = {p['pearson_r']:+.4f}")

    # Feature-to-Target correlations
    target_corrs = []
    for col in predictor_numeric:
        r = float(corr_matrix.loc[col, TARGET_COLUMN])
        target_corrs.append({
            "feature": col,
            "pearson_r_with_target": round(r, 4),
            "abs_r": round(abs(r), 4)
        })
    target_corrs.sort(key=lambda x: x["abs_r"], reverse=True)

    print(f"\n10. Feature-to-Target Correlations (with '{TARGET_COLUMN}'):")
    for tc in target_corrs:
        print(f"   - {tc['feature']:<20} : r = {tc['pearson_r_with_target']:+.4f}")

    # 11. Potential Leakage Diagnostic
    leakage_diagnostic = {
        "highest_absolute_feature_correlation": pairwise_corrs[0]["abs_r"],
        "highest_feature_pair": f"{pairwise_corrs[0]['feature_1']} vs {pairwise_corrs[0]['feature_2']}",
        "highest_target_correlation": target_corrs[0]["abs_r"],
        "highest_target_feature": target_corrs[0]["feature"],
        "threshold_checked": 0.85,
        "exceeds_threshold": bool(target_corrs[0]["abs_r"] >= 0.85),
        "identifier_column": "id",
        "id_unique_count": int(df["id"].nunique()),
        "id_total_rows": num_rows,
        "leakage_statement": (
            "No obvious target leakage was identified during the feature and target inspection. "
            "The identifier column 'id' is 100% unique (5,110 values across 5,110 rows) and must be excluded from feature vectors. "
            "Correlation analysis was used as one diagnostic check and does not by itself prove the absence of leakage or proxy identifiers."
        )
    }

    print(f"\n11. Target Leakage Diagnostic:")
    print(f"   - Max correlation with target: {target_corrs[0]['feature']} (r = {target_corrs[0]['pearson_r_with_target']:+.4f})")
    print(f"   - Identifier 'id' uniqueness: {leakage_diagnostic['id_unique_count']} / {num_rows} (must be excluded from modeling)")
    print(f"   - {leakage_diagnostic['leakage_statement']}")

    # 12. Potential Data Quality Observations
    data_quality_observations = [
        f"The dataset contains exactly {num_rows} rows and {num_cols} columns.",
        "Zero full-row duplicate records exist across the entire dataset.",
        "Column 'bmi' contains 201 missing values (3.93% of rows), which requires principled imputation (e.g. median).",
        "Column 'id' represents a unique arbitrary patient identifier (5,110 distinct integers) with no predictive utility.",
        "Column 'gender' contains a rare category 'Other' with exactly 1 record, alongside 'Female' (58.59%) and 'Male' (41.39%).",
        "Column 'smoking_status' contains an explicit 'Unknown' category for 1,544 rows (30.22%), which represents a recorded category rather than a null token.",
        "Severe class imbalance is present: 'stroke' = 1 constitutes only 4.87% (249 cases) versus 95.13% (4,861 cases) for 'stroke' = 0.",
        "Positive skewness is observed in 'avg_glucose_level' (+1.5723) with 627 statistical IQR outliers (12.27%) and 'bmi' (+1.0553) with 110 IQR outliers (2.24%).",
        "No continuous predictor exhibits near-perfect linear correlation (|r| >= 0.85) with the target."
    ]

    # Save JSON summary
    summary_dict = {
        "dataset_name": "Stroke Prediction Dataset",
        "dataset_path": "datasets/stroke/stroke.csv",
        "dataset_shape": {
            "rows": num_rows,
            "columns": num_cols
        },
        "target_distribution": target_dist_summary,
        "missing_values": missing_summary,
        "duplicate_rows": dup_rows,
        "numeric_descriptive_statistics": desc_stats,
        "skewness": skew_analysis,
        "iqr_outlier_analysis": outlier_analysis,
        "categorical_distributions": cat_summary,
        "numeric_correlation_matrix": corr_matrix_dict,
        "pairwise_numeric_feature_correlations": pairwise_corrs,
        "feature_to_target_correlations": target_corrs,
        "leakage_diagnostic": leakage_diagnostic,
        "data_quality_observations": data_quality_observations
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\nSaved EDA summary to: {summary_path}")

    # =========================================================================
    # PLOT GENERATION
    # =========================================================================
    print("\nGenerating EDA Plots...")

    # Plot 1: 01_target_distribution.png
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = [c0_count, c1_count]
    labels = ["No Stroke (0)", "Stroke (1)"]
    colors = [COLOR_CLASS_0, COLOR_CLASS_1]

    bars = axes[0].bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    axes[0].set_title("Stroke Target Distribution (Counts)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Observations", fontsize=11)
    axes[0].set_xlabel("Target Class (stroke)", fontsize=11)
    axes[0].set_ylim(0, max(counts) * 1.18)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / num_rows) * 100
        axes[0].annotate(f"{h:,}\n({pct:.2f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=11, fontweight="bold")

    axes[1].pie(counts, labels=labels, autopct="%1.2f%%", startangle=90, colors=colors,
                explode=(0.04, 0.08), textprops={"fontsize": 11, "fontweight": "bold"},
                wedgeprops={"edgecolor": "black", "linewidth": 1.2, "width": 0.6})
    axes[1].set_title(f"Stroke Class Proportions (Imbalance: {imbalance_ratio:.2f}:1)", fontsize=13, fontweight="bold", pad=12)

    plt.suptitle("Stroke Dataset — Target Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_target_distribution.png")
    plt.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p1}")

    # Plot 2: 02_missing_values.png
    fig, ax = plt.subplots(figsize=(12, 6))
    col_names = ALL_COLUMNS
    null_counts = [missing_dict[c] for c in col_names]
    null_pcts = [missing_pct_dict[c] for c in col_names]
    bar_colors = [COLOR_CLASS_1 if cnt > 0 else COLOR_PRIMARY for cnt in null_counts]

    bars = ax.bar(col_names, null_counts, color=bar_colors, edgecolor="black", linewidth=1.1, width=0.6)
    ax.set_title("Missing / Null Values Per Column in Stroke Dataset", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Missing Cell Count", fontsize=11)
    ax.set_xlabel("Dataset Columns", fontsize=11)
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, max(null_counts) * 1.3 if max(null_counts) > 0 else 10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, cnt, pct in zip(bars, null_counts, null_pcts):
        if cnt > 0:
            ax.annotate(f"{cnt} nulls\n({pct:.2f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, cnt),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#a94442")
        else:
            ax.annotate("0", xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color="#555555")

    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_missing_values.png")
    plt.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p2}")

    # Plot 3: 03_categorical_distributions.png
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, col in enumerate(CATEGORICAL_COLUMNS):
        ax = axes_flat[idx]
        vc = df[col].value_counts(dropna=False)
        cat_labels = [str(k) for k in vc.index]
        cat_vals = vc.values
        cat_colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(cat_vals)))

        b = ax.bar(cat_labels, cat_vals, color=cat_colors, edgecolor="black", linewidth=1.1, width=0.55)
        ax.set_title(f"Categorical Feature: {col}", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_ylim(0, max(cat_vals) * 1.22)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=25)

        for bar in b:
            h = bar.get_height()
            pct = (h / num_rows) * 100
            ax.annotate(f"{h:,}\n({pct:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="semibold")

    # The 6th slot: show binary discrete features hypertension and heart_disease
    ax_last = axes_flat[5]
    discrete_labels = ["No HTN", "HTN", "No HD", "HD"]
    discrete_counts = [
        int((df["hypertension"] == 0).sum()),
        int((df["hypertension"] == 1).sum()),
        int((df["heart_disease"] == 0).sum()),
        int((df["heart_disease"] == 1).sum())
    ]
    disc_colors = ["#74a9cf", "#02818a", "#bcbddc", "#756bb1"]
    b_disc = ax_last.bar(discrete_labels, discrete_counts, color=disc_colors, edgecolor="black", linewidth=1.1, width=0.55)
    ax_last.set_title("Discrete Indicators: Hypertension & Heart Disease", fontsize=11, fontweight="bold", pad=10)
    ax_last.set_ylabel("Count", fontsize=10)
    ax_last.set_ylim(0, max(discrete_counts) * 1.22)
    ax_last.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in b_disc:
        h = bar.get_height()
        pct = (h / num_rows) * 100
        ax_last.annotate(f"{h:,}\n({pct:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, fontweight="semibold")

    plt.suptitle("Stroke Dataset — Categorical and Discrete Predictor Distributions", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_categorical_distributions.png")
    plt.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p3}")

    # Plot 4: 04_numeric_distributions.png
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    plot_num_cols = ["age", "avg_glucose_level", "bmi"]

    for idx, col in enumerate(plot_num_cols):
        ax = axes[idx]
        s = df[col].dropna()
        ax.hist(s, bins=30, color=COLOR_PRIMARY, edgecolor="black", alpha=0.75)
        ax.axvline(s.median(), color="#d9534f", linestyle="--", linewidth=2.0, label=f"Median: {s.median():.1f}")
        ax.axvline(s.mean(), color="#f0ad4e", linestyle="-", linewidth=2.0, label=f"Mean: {s.mean():.1f}")

        sk = s.skew()
        ax.set_title(f"{col}\n(skewness: {sk:+.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Stroke Dataset — Continuous Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "04_numeric_distributions.png")
    plt.savefig(p4, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p4}")

    # Plot 5: 05_numeric_boxplots.png
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, col in enumerate(plot_num_cols):
        ax = axes[idx]
        s = df[col].dropna()
        ax.boxplot(s, orientation="vertical", patch_artist=True,
                   boxprops=dict(facecolor=COLOR_BOX_FILL, color=COLOR_BOX_LINE, linewidth=1.3),
                   medianprops=dict(color=COLOR_CLASS_1, linewidth=2.2),
                   whiskerprops=dict(color=COLOR_BOX_LINE, linewidth=1.3),
                   capprops=dict(color=COLOR_BOX_LINE, linewidth=1.3),
                   flierprops=dict(marker="o", color=COLOR_CLASS_1, alpha=0.5, markersize=4))
        out_cnt = outlier_analysis[col]["outlier_count"]
        out_pct = outlier_analysis[col]["outlier_percentage"]
        ax.set_title(f"{col}\n(IQR Outliers: {out_cnt} | {out_pct:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Observed {col}", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Stroke Dataset — Continuous Numeric Feature Boxplots (IQR Outlier Flags)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p5 = os.path.join(plots_dir, "05_numeric_boxplots.png")
    plt.savefig(p5, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p5}")

    # Plot 6: 06_correlation_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr_numeric_cols)))
    ax.set_yticks(range(len(corr_numeric_cols)))
    ax.set_xticklabels(corr_numeric_cols, rotation=35, ha="left", fontsize=10, fontweight="semibold")
    ax.set_yticklabels(corr_numeric_cols, fontsize=10, fontweight="semibold")

    for i in range(len(corr_numeric_cols)):
        for j in range(len(corr_numeric_cols)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    ax.set_title("Stroke Dataset — Pearson Numeric Correlation Matrix", fontsize=13, fontweight="bold", pad=28)
    plt.tight_layout()
    p6 = os.path.join(plots_dir, "06_correlation_heatmap.png")
    plt.savefig(p6, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p6}")

    # Plot 7: 07_feature_vs_target.png
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    # Panel 1: age vs stroke
    ax0 = axes_flat[0]
    ax0.boxplot([df[df["stroke"] == 0]["age"], df[df["stroke"] == 1]["age"]],
                patch_artist=True, widths=0.5,
                boxprops=dict(facecolor=COLOR_CLASS_0, alpha=0.7),
                medianprops=dict(color="black", linewidth=2.0))
    ax0.set_xticks([1, 2])
    ax0.set_xticklabels(["No Stroke (0)", "Stroke (1)"], fontsize=10, fontweight="semibold")
    ax0.set_title("Age Stratified by Stroke\n(r = +0.2453)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Age (years)", fontsize=10)
    ax0.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: avg_glucose_level vs stroke
    ax1 = axes_flat[1]
    ax1.boxplot([df[df["stroke"] == 0]["avg_glucose_level"], df[df["stroke"] == 1]["avg_glucose_level"]],
                patch_artist=True, widths=0.5,
                boxprops=dict(facecolor=COLOR_PRIMARY, alpha=0.7),
                medianprops=dict(color="black", linewidth=2.0))
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["No Stroke (0)", "Stroke (1)"], fontsize=10, fontweight="semibold")
    ax1.set_title("Avg Glucose Level Stratified by Stroke\n(r = +0.1319)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Avg Glucose Level", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: bmi vs stroke
    ax2 = axes_flat[2]
    bmi_c0 = df[df["stroke"] == 0]["bmi"].dropna()
    bmi_c1 = df[df["stroke"] == 1]["bmi"].dropna()
    ax2.boxplot([bmi_c0, bmi_c1], patch_artist=True, widths=0.5,
                boxprops=dict(facecolor="#74a9cf", alpha=0.7),
                medianprops=dict(color="black", linewidth=2.0))
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["No Stroke (0)", "Stroke (1)"], fontsize=10, fontweight="semibold")
    ax2.set_title("BMI Stratified by Stroke\n(r = +0.0424)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("BMI", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 4: hypertension vs stroke rate
    ax3 = axes_flat[3]
    htn_rates = df.groupby("hypertension")["stroke"].mean() * 100
    b3 = ax3.bar(["No Hypertension (0)", "Hypertension (1)"], htn_rates, color=["#74a9cf", "#02818a"], edgecolor="black", width=0.5)
    ax3.set_title("Stroke Incidence by Hypertension Status", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Stroke Rate (%)", fontsize=10)
    ax3.set_ylim(0, max(htn_rates) * 1.3)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b3:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 5: heart_disease vs stroke rate
    ax4 = axes_flat[4]
    hd_rates = df.groupby("heart_disease")["stroke"].mean() * 100
    b4 = ax4.bar(["No Heart Disease (0)", "Heart Disease (1)"], hd_rates, color=["#bcbddc", "#756bb1"], edgecolor="black", width=0.5)
    ax4.set_title("Stroke Incidence by Heart Disease History", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Stroke Rate (%)", fontsize=10)
    ax4.set_ylim(0, max(hd_rates) * 1.3)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b4:
        h = bar.get_height()
        ax4.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 6: smoking_status vs stroke rate
    ax5 = axes_flat[5]
    smk_rates = df.groupby("smoking_status")["stroke"].mean() * 100
    smk_order = ["never smoked", "Unknown", "formerly smoked", "smokes"]
    smk_vals = [smk_rates.get(k, 0.0) for k in smk_order]
    b5 = ax5.bar(smk_order, smk_vals, color=plt.cm.Oranges(np.linspace(0.4, 0.8, 4)), edgecolor="black", width=0.55)
    ax5.set_title("Stroke Incidence by Smoking Status", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Stroke Rate (%)", fontsize=10)
    ax5.set_ylim(0, max(smk_vals) * 1.3)
    ax5.tick_params(axis="x", rotation=20)
    ax5.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b5:
        h = bar.get_height()
        ax5.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Stroke Dataset — Feature-versus-Target Association Diagnostics", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    p7 = os.path.join(plots_dir, "07_feature_vs_target.png")
    plt.savefig(p7, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p7}")

    # Post-EDA integrity verification
    post_sha256 = compute_sha256(abs_dataset_path)
    hash_match = (pre_sha256 == post_sha256)
    print(f"\nPost-EDA Dataset SHA-256: {post_sha256}")
    print(f"Read-Only Integrity Verified: {hash_match}")
    if not hash_match:
        raise RuntimeError("CRITICAL ERROR: Dataset file was modified during EDA!")

    print("\n" + "=" * 80)
    print("STEP 2 EDA COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    candidate_paths = [
        os.path.join(workspace_root, "datasets", "stroke", "stroke.csv"),
        os.path.join("datasets", "stroke", "stroke.csv"),
        os.path.abspath("datasets/stroke/stroke.csv")
    ]
    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        resolved_path = candidate_paths[0]

    run_eda(resolved_path, script_dir)
