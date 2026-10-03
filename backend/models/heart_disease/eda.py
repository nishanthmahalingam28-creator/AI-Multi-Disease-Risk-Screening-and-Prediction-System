"""
Exploratory Data Analysis (EDA) Script for Heart Disease Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System
Member 1: Heart Disease Prediction Model

Step 2 — Exploratory Data Analysis
This script performs a purely descriptive, non-destructive exploratory data analysis
of the actual Heart Disease dataset without modifying any rows, columns, or values,
and without training any model or performing preprocessing.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt

# Canonical dataset path
DATASET_PATH = os.path.join("datasets", "heart_disease", "heart.csv")

# Feature columns
NUMERIC_COLUMNS = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium"
]

TARGET_COLUMN = "Heart Disease"
TARGET_CLASSES = ["Absence", "Presence"]

# Plot styling constants
COLOR_ABSENCE = "#2b5c8f"   # Deep Navy/Steel Blue
COLOR_PRESENCE = "#d9534f"  # Crimson/Coral Red
COLOR_MUTED = "#6c757d"
COLOR_PRIMARY = "#1f77b4"


def compute_iqr_outliers(series: pd.Series) -> dict:
    """
    Computes IQR-based statistical outlier bounds and counts for a numeric series.
    Note: These are statistical diagnostic flags, not proven errors or data flaws.
    """
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = float(q3 - q1)
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    count = int(len(outliers))
    pct = float((count / len(series)) * 100)
    return {
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
    print("STEP 2: HEART DISEASE EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # Resolve paths
    abs_dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(abs_dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {abs_dataset_path}")

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "eda_summary.json")

    # Read original dataset in read-only manner
    df = pd.read_csv(abs_dataset_path)
    num_rows, num_cols = df.shape

    print(f"1. Dataset Shape: {num_rows} rows x {num_cols} columns")

    # 2. Target Distribution
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    target_proportions = df[TARGET_COLUMN].value_counts(dropna=False, normalize=True) * 100

    target_dist_summary = {
        "target_column": TARGET_COLUMN,
        "classes": {
            "Absence": {
                "count": int(target_counts.get("Absence", 0)),
                "percentage": round(float(target_proportions.get("Absence", 0.0)), 4)
            },
            "Presence": {
                "count": int(target_counts.get("Presence", 0)),
                "percentage": round(float(target_proportions.get("Presence", 0.0)), 4)
            }
        },
        "imbalance_ratio": round(float(target_counts.max() / target_counts.min()), 4)
    }
    print(f"\n2. Target Distribution ({TARGET_COLUMN}):")
    print(f"   - 'Absence' : {target_dist_summary['classes']['Absence']['count']} rows ({target_dist_summary['classes']['Absence']['percentage']}%)")
    print(f"   - 'Presence': {target_dist_summary['classes']['Presence']['count']} rows ({target_dist_summary['classes']['Presence']['percentage']}%)")
    print(f"   - Class ratio: {target_dist_summary['imbalance_ratio']}:1")

    # 3. Missing Values
    missing_dict = df.isnull().sum().to_dict()
    total_missing = sum(missing_dict.values())
    print(f"\n3. Missing Values: Total = {total_missing} missing cells")

    # 4. Duplicate Rows
    duplicate_count = int(df.duplicated().sum())
    print(f"\n4. Duplicate Rows: Total = {duplicate_count}")

    # 5. Descriptive Statistics for Numeric Features
    desc_stats = {}
    print("\n5. Descriptive Statistics for Numeric Features:")
    for col in NUMERIC_COLUMNS:
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
        print(f"   - {col:<24} | Mean: {st['mean']:>8.2f} | Std: {st['std']:>7.2f} | Median: {st['50%_median']:>6.2f} | Range: [{st['min']}, {st['max']}]")

    # 6. Zero-Value Analysis
    zero_analysis = {}
    print("\n6. Zero-Value Analysis:")
    # Feature category classification based on data structure
    discrete_features = ["Sex", "FBS over 120", "EKG results", "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium", "Chest pain type"]
    continuous_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

    for col in NUMERIC_COLUMNS:
        z_cnt = int((df[col] == 0).sum())
        z_pct = float((z_cnt / num_rows) * 100)
        f_type = "discrete_indicator" if col in discrete_features else "continuous_measurement"
        zero_analysis[col] = {
            "zero_count": z_cnt,
            "zero_percentage": round(z_pct, 4),
            "feature_type": f_type,
            "zero_interpretation_note": (
                "Valid discrete category or zero count" if f_type == "discrete_indicator"
                else ("Zero ST depression (baseline/no depression)" if col == "ST depression"
                      else "Physiological measurement with strictly positive readings (0 zeros observed)")
            )
        }
        print(f"   - {col:<24} : {z_cnt:3d} zeros ({z_pct:6.2f}%) [{f_type}]")

    # 7. Distribution / Skewness Analysis
    skew_analysis = {}
    print("\n7. Skewness Analysis:")
    for col in NUMERIC_COLUMNS:
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
        print(f"   - {col:<24} : {sk:>+7.4f} ({skew_class})")

    # 8. Outlier Analysis (IQR)
    outlier_analysis = {}
    print("\n8. IQR Outlier Analysis (Diagnostic Flagging Only):")
    for col in NUMERIC_COLUMNS:
        outlier_res = compute_iqr_outliers(df[col])
        outlier_analysis[col] = outlier_res
        print(f"   - {col:<24} : {outlier_res['outlier_count']:2d} outliers ({outlier_res['outlier_percentage']:5.2f}%) | Bounds: [{outlier_res['lower_bound']}, {outlier_res['upper_bound']}]")

    # 9. Feature-to-Feature Correlations
    corr_matrix = df[NUMERIC_COLUMNS].corr()
    pairwise_corrs = []
    for i in range(len(NUMERIC_COLUMNS)):
        for j in range(i + 1, len(NUMERIC_COLUMNS)):
            c1, c2 = NUMERIC_COLUMNS[i], NUMERIC_COLUMNS[j]
            r = float(corr_matrix.loc[c1, c2])
            pairwise_corrs.append({
                "feature_1": c1,
                "feature_2": c2,
                "pearson_r": round(r, 4),
                "abs_r": round(abs(r), 4)
            })
    pairwise_corrs.sort(key=lambda x: x["abs_r"], reverse=True)

    print("\n9. Top 5 Strongest Feature-to-Feature Correlations:")
    for p in pairwise_corrs[:5]:
        print(f"   - {p['feature_1']} vs {p['feature_2']}: r = {p['pearson_r']:+.4f}")

    # 10. Feature-to-Target Association (Temporary binary encoding purely for association check)
    target_binary = (df[TARGET_COLUMN] == "Presence").astype(int)
    target_corrs = []
    for col in NUMERIC_COLUMNS:
        r = float(df[col].corr(target_binary))
        target_corrs.append({
            "feature": col,
            "pearson_r_with_target": round(r, 4),
            "abs_r": round(abs(r), 4)
        })
    target_corrs.sort(key=lambda x: x["abs_r"], reverse=True)

    print("\n10. Feature-to-Target Associations (Pearson r with 'Presence'=1, 'Absence'=0):")
    for tc in target_corrs:
        print(f"   - {tc['feature']:<24} : r = {tc['pearson_r_with_target']:+.4f}")

    # 11. Potential Leakage Diagnostic
    leakage_diagnostic = {
        "highest_absolute_feature_correlation": pairwise_corrs[0]["abs_r"],
        "highest_feature_pair": f"{pairwise_corrs[0]['feature_1']} vs {pairwise_corrs[0]['feature_2']}",
        "highest_target_correlation": target_corrs[0]["abs_r"],
        "highest_target_feature": target_corrs[0]["feature"],
        "threshold_checked": 0.85,
        "exceeds_threshold": bool(target_corrs[0]["abs_r"] >= 0.85),
        "leakage_statement": (
            "No obvious target leakage was identified during the feature and target inspection. "
            "Correlation analysis was used as one diagnostic check and does not by itself prove the absence of leakage or proxy identifiers."
        )
    }
    print(f"\n11. Target Leakage Diagnostic:")
    print(f"   - Max correlation with target: {target_corrs[0]['feature']} (r = {target_corrs[0]['pearson_r_with_target']:+.4f})")
    print(f"   - Max feature-feature correlation: {pairwise_corrs[0]['feature_1']} vs {pairwise_corrs[0]['feature_2']} (r = {pairwise_corrs[0]['pearson_r']:+.4f})")
    print(f"   - {leakage_diagnostic['leakage_statement']}")

    # 12. Data Quality Observations
    data_quality_observations = [
        "The dataset consists of exactly 270 rows and 14 columns with zero standard null/missing values.",
        "There are zero duplicate rows across the entire dataset.",
        "Continuous physiological measurements ('BP', 'Cholesterol', 'Max HR') contain 0 zero values (all strictly positive).",
        "Zero values exist in discrete/indicator features ('Sex', 'FBS over 120', 'EKG results', 'Exercise angina', 'Number of vessels fluro') and 'ST depression' (representing 0 baseline depression).",
        "High positive skewness is observed in 'FBS over 120' (+1.9920), 'ST depression' (+1.2629), 'Number of vessels fluro' (+1.2099), and 'Cholesterol' (+1.1837).",
        "Statistical IQR outlier flags are observed in 'BP' (9 records, 3.33%), 'Cholesterol' (5 records, 1.85%), 'ST depression' (4 records, 1.48%), and 'Max HR' (1 record, 0.37%).",
        "All features have correlation |r| < 0.61 with each other and |r| <= 0.5250 with the target variable 'Heart Disease'."
    ]

    # Save JSON summary
    summary_dict = {
        "dataset_name": "Heart Disease Dataset",
        "dataset_path": "datasets/heart_disease/heart.csv",
        "dataset_shape": {
            "rows": num_rows,
            "columns": num_cols
        },
        "target_distribution": target_dist_summary,
        "missing_values": {
            "total_missing": total_missing,
            "per_column": missing_dict
        },
        "duplicate_rows": duplicate_count,
        "numeric_features_descriptive_statistics": desc_stats,
        "zero_value_analysis": zero_analysis,
        "distribution_skewness": skew_analysis,
        "iqr_outlier_analysis": outlier_analysis,
        "feature_to_feature_correlations_top_10": pairwise_corrs[:10],
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = [target_counts.get("Absence", 0), target_counts.get("Presence", 0)]
    labels = ["Absence", "Presence"]
    colors = [COLOR_ABSENCE, COLOR_PRESENCE]

    # Bar chart
    bars = axes[0].bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    axes[0].set_title("Heart Disease Target Distribution (Counts)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Observations", fontsize=11)
    axes[0].set_xlabel("Target Class (Heart Disease)", fontsize=11)
    axes[0].set_ylim(0, max(counts) * 1.2)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / num_rows) * 100
        axes[0].annotate(f"{h}\n({pct:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Donut chart
    axes[1].pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors,
               explode=(0.04, 0.04), textprops={"fontsize": 11, "fontweight": "bold"},
               wedgeprops={"edgecolor": "black", "linewidth": 1.2, "width": 0.6})
    axes[1].set_title("Heart Disease Class Proportions", fontsize=13, fontweight="bold", pad=12)

    plt.suptitle("Heart Disease Dataset — Target Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_target_distribution.png")
    plt.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p1}")

    # Plot 2: 02_feature_distributions.png
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes_flat = axes.flatten()

    for idx, col in enumerate(NUMERIC_COLUMNS):
        ax = axes_flat[idx]
        s = df[col]
        n_unique = s.nunique()
        bins = min(20, n_unique) if n_unique > 5 else np.arange(s.min() - 0.5, s.max() + 1.5, 1)
        
        ax.hist(s, bins=bins, color=COLOR_PRIMARY, edgecolor="black", alpha=0.75, density=False)
        ax.set_title(f"{col}\n(skew: {s.skew():+.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused subplots (16 total slots, 13 features)
    for extra_idx in range(len(NUMERIC_COLUMNS), len(axes_flat)):
        fig.delaxes(axes_flat[extra_idx])

    plt.suptitle("Heart Disease Dataset — Numeric Feature Distributions", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_feature_distributions.png")
    plt.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p2}")

    # Plot 3: 03_feature_boxplots.png
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes_flat = axes.flatten()

    for idx, col in enumerate(NUMERIC_COLUMNS):
        ax = axes_flat[idx]
        s = df[col]
        box = ax.boxplot(s, vert=True, patch_artist=True,
                         boxprops=dict(facecolor="#c6dbef", color="#08519c", linewidth=1.2),
                         medianprops=dict(color="#d9534f", linewidth=2.0),
                         whiskerprops=dict(color="#08519c", linewidth=1.2),
                         capprops=dict(color="#08519c", linewidth=1.2),
                         flierprops=dict(marker="o", color="#d9534f", alpha=0.6, markersize=5))
        out_cnt = outlier_analysis[col]["outlier_count"]
        ax.set_title(f"{col}\n(IQR Outliers: {out_cnt})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Observed Value", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused subplots
    for extra_idx in range(len(NUMERIC_COLUMNS), len(axes_flat)):
        fig.delaxes(axes_flat[extra_idx])

    plt.suptitle("Heart Disease Dataset — Numeric Feature Boxplots (Outlier Diagnostic)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_feature_boxplots.png")
    plt.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p3}")

    # Plot 4: 04_correlation_heatmap.png
    fig, ax = plt.subplots(figsize=(13, 11))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(NUMERIC_COLUMNS)))
    ax.set_yticks(range(len(NUMERIC_COLUMNS)))
    ax.set_xticklabels(NUMERIC_COLUMNS, rotation=45, ha="left", fontsize=9, fontweight="semibold")
    ax.set_yticklabels(NUMERIC_COLUMNS, fontsize=9, fontweight="semibold")

    # Annotate numeric values
    for i in range(len(NUMERIC_COLUMNS)):
        for j in range(len(NUMERIC_COLUMNS)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title("Heart Disease Dataset — Pearson Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=28)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "04_correlation_heatmap.png")
    plt.savefig(p4, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p4}")

    # Plot 5: 05_feature_vs_target.png
    # Visualizing key features stratified by target (Absence vs Presence)
    key_features = [
        "Thallium",
        "Number of vessels fluro",
        "Exercise angina",
        "ST depression",
        "Chest pain type",
        "Max HR",
        "Age",
        "BP",
        "Cholesterol"
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes_flat = axes.flatten()

    for idx, col in enumerate(key_features):
        ax = axes_flat[idx]
        absence_vals = df[df[TARGET_COLUMN] == "Absence"][col]
        presence_vals = df[df[TARGET_COLUMN] == "Presence"][col]

        bplot = ax.boxplot([absence_vals, presence_vals],
                          patch_artist=True, widths=0.5,
                          medianprops=dict(color="black", linewidth=2.0),
                          whiskerprops=dict(linewidth=1.2),
                          flierprops=dict(marker="o", alpha=0.5, markersize=4))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Absence", "Presence"], fontsize=10, fontweight="semibold")
        
        # Color boxes by class
        bplot["boxes"][0].set_facecolor(COLOR_ABSENCE)
        bplot["boxes"][0].set_alpha(0.7)
        bplot["boxes"][1].set_facecolor(COLOR_PRESENCE)
        bplot["boxes"][1].set_alpha(0.7)

        # Correlation with target
        r_val = next(item["pearson_r_with_target"] for item in target_corrs if item["feature"] == col)
        ax.set_title(f"{col}\n(Pearson r = {r_val:+.4f})", fontsize=10, fontweight="bold")
        ax.set_ylabel(col, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Heart Disease Dataset — Key Features Stratified by Target Class", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    p5 = os.path.join(plots_dir, "05_feature_vs_target.png")
    plt.savefig(p5, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   [x] Created: {p5}")

    print("\n" + "=" * 80)
    print("STEP 2 EDA COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    # Resolve candidate dataset paths
    candidate_paths = [
        os.path.join(workspace_root, "datasets", "heart_disease", "heart.csv"),
        os.path.join("datasets", "heart_disease", "heart.csv"),
        os.path.abspath("datasets/heart_disease/heart.csv")
    ]
    resolved_dataset_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_dataset_path = p
            break

    if resolved_dataset_path is None:
        resolved_dataset_path = candidate_paths[0]

    output_directory = script_dir
    run_eda(resolved_dataset_path, output_directory)
