"""
Exploratory Data Analysis (EDA) Script for Breast Cancer Model
Member 1: AI Multi-Disease Risk Screening and Prediction System

Dataset: datasets/breast_cancer/breast.csv
Target: diagnosis (B: Benign, M: Malignant)
Excluded: id, Unnamed: 32
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

# Define feature groups
MEAN_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
]

SE_FEATURES = [
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se"
]

WORST_FEATURES = [
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

ALL_PREDICTORS = MEAN_FEATURES + SE_FEATURES + WORST_FEATURES

# Palette for clinical clarity
PALETTE = {"B": "#2b5c8f", "M": "#d9534f"}  # Blue for Benign, Red for Malignant

def run_eda(dataset_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA) - BREAST CANCER MODEL")
    print("=" * 80)

    # Ingestion
    df = pd.read_csv(dataset_path)
    raw_shape = df.shape
    print(f"Dataset ingested from: {dataset_path}")
    print(f"Raw shape: {raw_shape[0]} rows, {raw_shape[1]} columns")

    # Exclude known non-predictor columns
    excluded_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    print(f"Excluded non-clinical columns: {excluded_cols}")

    # Verify features and target
    assert "diagnosis" in df.columns, "Target 'diagnosis' missing from dataset!"
    missing_features = [f for f in ALL_PREDICTORS if f not in df.columns]
    assert len(missing_features) == 0, f"Missing features: {missing_features}"

    # 1. Target Distribution
    target_counts = df["diagnosis"].value_counts().to_dict()
    target_pcts = (df["diagnosis"].value_counts(normalize=True) * 100).round(2).to_dict()
    print("\n--- 1. Target Distribution ---")
    print(f"Benign (B): {target_counts.get('B', 0)} ({target_pcts.get('B', 0)}%)")
    print(f"Malignant (M): {target_counts.get('M', 0)} ({target_pcts.get('M', 0)}%)")
    print(f"Ratio (B : M): {target_counts.get('B', 1) / target_counts.get('M', 1):.2f} : 1")

    # Plot 1: Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x="diagnosis", hue="diagnosis", data=df, ax=axes[0], palette=PALETTE, order=["B", "M"], legend=False)
    axes[0].set_title("Target Class Counts (diagnosis)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Diagnosis Class", fontsize=11)
    axes[0].set_ylabel("Patient Count", fontsize=11)
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f'{int(height)} ({height/len(df)*100:.1f}%)',
                         (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    axes[1].pie([target_counts["B"], target_counts["M"]],
                labels=[f"Benign (B)\n{target_counts['B']} ({target_pcts['B']}%)",
                        f"Malignant (M)\n{target_counts['M']} ({target_pcts['M']}%)"],
                colors=[PALETTE["B"], PALETTE["M"]],
                autopct='%1.1f%%', startangle=90, explode=(0, 0.05),
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    axes[1].set_title("Target Class Proportion", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p1_path = os.path.join(plots_dir, "01_target_distribution.png")
    fig.savefig(p1_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {p1_path}")

    # 2. Missing Values Check
    missing_summary = df[ALL_PREDICTORS].isnull().sum()
    total_missing = missing_summary.sum()
    print("\n--- 2. Missing Values ---")
    print(f"Total missing values across all 30 predictor features: {total_missing}")

    # 3. Duplicate Records Check
    duplicate_rows = df[ALL_PREDICTORS].duplicated().sum()
    print("\n--- 3. Duplicate Records ---")
    print(f"Duplicate records across the 30 predictor features: {duplicate_rows}")

    # 4. Descriptive Statistics
    desc_df = df[ALL_PREDICTORS].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    desc_df['median'] = df[ALL_PREDICTORS].median()
    desc_df['skewness'] = df[ALL_PREDICTORS].skew()

    # 5. Outlier Investigation (IQR Method)
    outlier_records = {}
    for col in ALL_PREDICTORS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(out_mask.sum())
        outlier_records[col] = {
            "count": n_outliers,
            "percentage": round((n_outliers / len(df)) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4)
        }

    # Plot 2, 3, 4: Feature Distributions grouped into Mean, SE, Worst
    def plot_grouped_distributions(feature_list, title, filename):
        fig, axes = plt.subplots(5, 2, figsize=(14, 16))
        axes = axes.flatten()
        for idx, col in enumerate(feature_list):
            ax = axes[idx]
            sns.histplot(data=df, x=col, hue="diagnosis", kde=True, ax=ax,
                         palette=PALETTE, alpha=0.4, bins=25, stat="density", common_norm=False)
            ax.set_title(f"{col} (Skew: {df[col].skew():.2f})", fontsize=10, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel("Density")
        plt.suptitle(f"Distributions by Diagnosis Class: {title}", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, filename)
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
        return save_path

    p2_path = plot_grouped_distributions(MEAN_FEATURES, "Mean Morphological Features", "02_distributions_mean_features.png")
    p3_path = plot_grouped_distributions(SE_FEATURES, "Standard Error (SE) Features", "03_distributions_se_features.png")
    p4_path = plot_grouped_distributions(WORST_FEATURES, "Worst / Extreme Features", "04_distributions_worst_features.png")
    print(f"Saved: {p2_path}")
    print(f"Saved: {p3_path}")
    print(f"Saved: {p4_path}")

    # Plot 5, 6, 7: Boxplots for Outlier Inspection
    def plot_grouped_boxplots(feature_list, title, filename):
        fig, axes = plt.subplots(5, 2, figsize=(14, 16))
        axes = axes.flatten()
        for idx, col in enumerate(feature_list):
            ax = axes[idx]
            sns.boxplot(x="diagnosis", y=col, hue="diagnosis", data=df, ax=ax, palette=PALETTE, order=["B", "M"], fliersize=3, legend=False)
            out_cnt = outlier_records[col]["count"]
            out_pct = outlier_records[col]["percentage"]
            ax.set_title(f"{col} | Outliers: {out_cnt} ({out_pct}%)", fontsize=10, fontweight='bold')
            ax.set_xlabel("Diagnosis")
            ax.set_ylabel("Value")
        plt.suptitle(f"Boxplots & Outlier Inspection: {title}", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, filename)
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
        return save_path

    p5_path = plot_grouped_boxplots(MEAN_FEATURES, "Mean Morphological Features", "05_boxplots_mean_features.png")
    p6_path = plot_grouped_boxplots(SE_FEATURES, "Standard Error (SE) Features", "06_boxplots_se_features.png")
    p7_path = plot_grouped_boxplots(WORST_FEATURES, "Worst / Extreme Features", "07_boxplots_worst_features.png")
    print(f"Saved: {p5_path}")
    print(f"Saved: {p6_path}")
    print(f"Saved: {p7_path}")

    # 6. Correlation Analysis
    corr_matrix = df[ALL_PREDICTORS].corr()

    # Plot 8: Full Correlation Matrix Heatmap
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="vlag", vmin=-1.0, vmax=1.0,
                annot=False, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap: 30 Predictor Features (Pearson r)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    p8_path = os.path.join(plots_dir, "08_correlation_matrix_heatmap.png")
    fig.savefig(p8_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {p8_path}")

    # Highly collinear pairs (r > 0.90)
    upper_corr = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_pairs = []
    for c1 in upper_corr.index:
        for c2 in upper_corr.columns:
            val = upper_corr.loc[c1, c2]
            if val > 0.90:
                collinear_pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "pearson_r": round(float(val), 4)
                })
    collinear_pairs.sort(key=lambda x: x["pearson_r"], reverse=True)

    # Correlation with Target (Encoded: M=1, B=0)
    target_encoded = (df["diagnosis"] == "M").astype(int)
    target_corrs = df[ALL_PREDICTORS].apply(lambda s: s.corr(target_encoded)).sort_values(ascending=True)

    # Plot 9: Predictor-to-Target Correlation Barplot
    fig, ax = plt.subplots(figsize=(10, 11))
    colors = ['#d9534f' if v > 0.5 else ('#f0ad4e' if v > 0.2 else '#5bc0de') for v in target_corrs.values]
    target_corrs.plot(kind="barh", ax=ax, color=colors, edgecolor='grey', linewidth=0.5)
    ax.set_title("Pearson Correlation with Target Label (Diagnosis = Malignant)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    for idx, v in enumerate(target_corrs.values):
        ax.text(v + (0.01 if v >= 0 else -0.05), idx, f"{v:.3f}", va='center', fontsize=9)
    plt.tight_layout()
    p9_path = os.path.join(plots_dir, "09_target_correlations.png")
    fig.savefig(p9_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {p9_path}")

    # Plot 10: Focused Pairplot of Top Features
    top_features = ["concave points_worst", "perimeter_worst", "radius_worst", "area_mean", "concavity_mean"]
    pairplot_df = df[top_features + ["diagnosis"]].copy()
    g = sns.pairplot(pairplot_df, hue="diagnosis", palette=PALETTE, diag_kind="kde",
                     plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "none"},
                     diag_kws={"common_norm": False})
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle("Pairwise Relationships: Top 5 Target-Correlated Features by Diagnosis", fontsize=13, fontweight='bold')
    p10_path = os.path.join(plots_dir, "10_feature_relationships_pairplot.png")
    g.savefig(p10_path, dpi=160)
    plt.close()
    print(f"Saved: {p10_path}")

    # 7. Summary JSON Generation
    summary_data = {
        "dataset_path": os.path.abspath(dataset_path),
        "raw_shape": {"rows": raw_shape[0], "columns": raw_shape[1]},
        "evaluated_features_count": len(ALL_PREDICTORS),
        "target_variable": "diagnosis",
        "target_distribution": {
            "counts": target_counts,
            "percentages": target_pcts,
            "imbalance_ratio": round(target_counts["B"] / target_counts["M"], 2)
        },
        "missing_values_count": int(total_missing),
        "duplicate_records_count": int(duplicate_rows),
        "skewness_summary": {
            "top_positive_skewed": df[ALL_PREDICTORS].skew().sort_values(ascending=False).head(5).round(3).to_dict(),
            "top_negative_skewed": df[ALL_PREDICTORS].skew().sort_values().head(3).round(3).to_dict()
        },
        "outlier_summary": {
            "top_outlier_features": sorted(outlier_records.items(), key=lambda x: x[1]["count"], reverse=True)[:8]
        },
        "collinearity_summary": {
            "pairs_above_90_count": len(collinear_pairs),
            "top_collinear_pairs": collinear_pairs[:10]
        },
        "target_correlations": {
            "top_5_positive": target_corrs.tail(5).round(4).to_dict(),
            "lowest_correlations": target_corrs.head(5).round(4).to_dict()
        },
        "generated_plots": [
            "01_target_distribution.png",
            "02_distributions_mean_features.png",
            "03_distributions_se_features.png",
            "04_distributions_worst_features.png",
            "05_boxplots_mean_features.png",
            "06_boxplots_se_features.png",
            "07_boxplots_worst_features.png",
            "08_correlation_matrix_heatmap.png",
            "09_target_correlations.png",
            "10_feature_relationships_pairplot.png"
        ]
    }

    json_path = os.path.join(output_dir, "eda_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved summary JSON to: {json_path}")
    print("=" * 80)
    print("EDA EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    dataset_file = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "datasets", "breast_cancer", "breast.csv"
    )
    if not os.path.exists(dataset_file):
        dataset_file = os.path.join("datasets", "breast_cancer", "breast.csv")
    
    out_directory = os.path.dirname(os.path.abspath(__file__))
    run_eda(dataset_file, out_directory)
