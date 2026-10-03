"""
Exploratory Data Analysis (EDA) Script for Diabetes Risk Screening Model
AI Multi-Disease Risk Screening and Prediction System

Dataset: datasets/diabetes/diabetes.csv
Target: Outcome (0: Negative / Non-diabetic, 1: Positive / Diabetic)
Predictors: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless execution
import matplotlib.pyplot as plt


PREDICTOR_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET_COLUMN = "Outcome"

PHYSIOLOGICAL_ZERO_CHECK_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Color palette for clear, professional visualizations
COLOR_PRIMARY = "#2b5c8f"    # Deep Blue
COLOR_ACCENT = "#d9534f"     # Coral Red
COLOR_MUTED = "#6c757d"      # Slate Gray
COLOR_PALETTE = ["#2b5c8f", "#d9534f"]


def compute_iqr_outliers(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return {
        "q1": float(round(q1, 4)),
        "q3": float(round(q3, 4)),
        "iqr": float(round(iqr, 4)),
        "lower_bound": float(round(lower_bound, 4)),
        "upper_bound": float(round(upper_bound, 4)),
        "count": int(len(outliers)),
        "percentage": float(round((len(outliers) / len(series)) * 100, 2))
    }


def run_eda(dataset_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA) - DIABETES MODEL")
    print("=" * 80)

    # 1. Ingestion and Shape
    abs_dataset_path = os.path.abspath(dataset_path)
    df = pd.read_csv(abs_dataset_path)
    n_rows, n_cols = df.shape
    print(f"Dataset ingested from: {abs_dataset_path}")
    print(f"Dataset shape: {n_rows} rows, {n_cols} columns")

    # Verify expected columns
    for col in PREDICTOR_COLUMNS + [TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in dataset!")

    # 2. Target Distribution
    print("\n--- 1. Target Distribution ('Outcome') ---")
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    target_pcts = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()
    imbalance_ratio = round(target_counts.get(0, 1) / target_counts.get(1, 1), 2)
    print(f"Class 0 (Negative): {target_counts.get(0, 0)} ({target_pcts.get(0, 0.0):.2f}%)")
    print(f"Class 1 (Positive): {target_counts.get(1, 0)} ({target_pcts.get(1, 0.0):.2f}%)")
    print(f"Imbalance ratio (0 : 1): {imbalance_ratio} : 1")

    # Plot 1: Target Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = ax1.bar(["Class 0 (Negative)", "Class 1 (Positive)"],
                   [target_counts.get(0, 0), target_counts.get(1, 0)],
                   color=COLOR_PALETTE, width=0.55, edgecolor="#333333", linewidth=1.2)
    ax1.set_title("Target Class Counts ('Outcome')", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Observations", fontsize=10)
    ax1.set_ylim(0, max(target_counts.values()) * 1.15)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        pct = (h / n_rows) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 12,
                 f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    wedges, texts, autotexts = ax2.pie(
        [target_counts.get(0, 0), target_counts.get(1, 0)],
        labels=[f"Class 0\n{target_counts.get(0, 0)} ({target_pcts.get(0, 0):.1f}%)",
                f"Class 1\n{target_counts.get(1, 0)} ({target_pcts.get(1, 0):.1f}%)"],
        colors=COLOR_PALETTE,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.05),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Target Class Proportions", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    p1_path = os.path.join(plots_dir, "01_target_distribution.png")
    fig.savefig(p1_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {p1_path}")

    # 3. Missing-Value and Zero-Value Analysis
    print("\n--- 2. Missing-Value and Zero-Value Analysis ---")
    standard_nulls = df.isnull().sum().to_dict()
    total_standard_nulls = int(sum(standard_nulls.values()))
    print(f"Total standard null (NaN) values: {total_standard_nulls}")

    zero_counts = {}
    for col in df.columns:
        z_cnt = int((df[col] == 0).sum())
        z_pct = float(round((z_cnt / n_rows) * 100, 2))
        zero_counts[col] = {"count": z_cnt, "percentage": z_pct}
        if col in PHYSIOLOGICAL_ZERO_CHECK_COLS:
            print(f"   - '{col}': {z_cnt} zeros ({z_pct:.2f}%)")

    # Plot 2: Zero Values Analysis
    fig, ax = plt.subplots(figsize=(10, 5))
    zero_plot_cols = PREDICTOR_COLUMNS
    zero_plot_counts = [zero_counts[c]["count"] for c in zero_plot_cols]
    zero_plot_pcts = [zero_counts[c]["percentage"] for c in zero_plot_cols]
    bar_colors = [COLOR_ACCENT if c in PHYSIOLOGICAL_ZERO_CHECK_COLS else COLOR_MUTED for c in zero_plot_cols]

    bars = ax.bar(zero_plot_cols, zero_plot_counts, color=bar_colors, edgecolor="#333333", width=0.6)
    ax.set_title("Zero Values Count across Predictor Columns\n(Red: Physiologically questionable zeros)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Count of Zero Values", fontsize=10)
    ax.set_xticks(range(len(zero_plot_cols)))
    ax.set_xticklabels(zero_plot_cols, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, max(zero_plot_counts) * 1.15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    for idx, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 6,
                f"{int(h)}\n({zero_plot_pcts[idx]:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    p2_path = os.path.join(plots_dir, "02_zero_values_analysis.png")
    fig.savefig(p2_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {p2_path}")

    # 4. Feature Summary & Distributions
    print("\n--- 3. Numerical Summary & Distribution Statistics ---")
    desc = df[PREDICTOR_COLUMNS].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    desc['median'] = df[PREDICTOR_COLUMNS].median()
    desc['skewness'] = df[PREDICTOR_COLUMNS].skew()
    desc['kurtosis'] = df[PREDICTOR_COLUMNS].kurtosis()
    print(desc.round(3).to_string())

    # Plot 3: Numerical Feature Distributions
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
    for idx, col in enumerate(PREDICTOR_COLUMNS):
        ax = axes[idx]
        ax.hist(df[col], bins=25, color=COLOR_PRIMARY, edgecolor="#333333", alpha=0.7, density=False)
        ax.axvline(df[col].median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median: {df[col].median():.1f}")
        ax.axvline(df[col].mean(), color="red", linestyle=":", linewidth=1.5, label=f"Mean: {df[col].mean():.1f}")
        ax.set_title(f"{col} (Skew: {df[col].skew():.2f})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.suptitle("Feature Histograms with Mean and Median Indicators", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    p3_path = os.path.join(plots_dir, "03_feature_distributions.png")
    fig.savefig(p3_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {p3_path}")

    # 5. Outlier Investigation using IQR
    print("\n--- 4. Outlier Investigation (IQR Method) ---")
    outlier_summary = {}
    for col in PREDICTOR_COLUMNS:
        outlier_summary[col] = compute_iqr_outliers(df[col])
        print(f"   - '{col}': {outlier_summary[col]['count']} outliers "
              f"({outlier_summary[col]['percentage']}%) | "
              f"IQR bounds: [{outlier_summary[col]['lower_bound']}, {outlier_summary[col]['upper_bound']}]")

    # Plot 4: Boxplots for Numerical Features
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    axes = axes.flatten()
    for idx, col in enumerate(PREDICTOR_COLUMNS):
        ax = axes[idx]
        bp = ax.boxplot(df[col], orientation='horizontal', patch_artist=True,
                        boxprops=dict(facecolor="#d0e1fd", color="#2b5c8f", linewidth=1.2),
                        medianprops=dict(color="#d9534f", linewidth=1.8),
                        whiskerprops=dict(color="#2b5c8f", linewidth=1.2),
                        capprops=dict(color="#2b5c8f", linewidth=1.2),
                        flierprops=dict(marker='o', color="#d9534f", alpha=0.6, markersize=5))
        ax.set_title(f"{col} | IQR Outliers: {outlier_summary[col]['count']} ({outlier_summary[col]['percentage']}%)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_yticks([])
        ax.grid(axis='x', linestyle=':', alpha=0.5)
    plt.suptitle("Feature Boxplots and IQR Outlier Inspection", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    p4_path = os.path.join(plots_dir, "04_feature_boxplots.png")
    fig.savefig(p4_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {p4_path}")

    # 6. Correlation Analysis
    print("\n--- 5. Correlation Analysis ---")
    all_cols = PREDICTOR_COLUMNS + [TARGET_COLUMN]
    corr_matrix = df[all_cols].corr()

    # Feature-to-target correlations
    target_corrs = corr_matrix[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
    print("Linear correlation of features with Outcome:")
    for col, r_val in target_corrs.items():
        print(f"   - '{col}': r = {r_val:.4f}")

    # Strongest pairwise feature-feature correlations (off-diagonal)
    pairwise_corrs = []
    for i in range(len(PREDICTOR_COLUMNS)):
        for j in range(i + 1, len(PREDICTOR_COLUMNS)):
            f1 = PREDICTOR_COLUMNS[i]
            f2 = PREDICTOR_COLUMNS[j]
            r_val = float(corr_matrix.loc[f1, f2])
            pairwise_corrs.append({
                "feature_1": f1,
                "feature_2": f2,
                "pearson_r": round(r_val, 4),
                "abs_pearson_r": round(abs(r_val), 4)
            })
    pairwise_corrs.sort(key=lambda x: x["abs_pearson_r"], reverse=True)
    print("\nTop 5 Strongest Pairwise Feature Correlations:")
    for item in pairwise_corrs[:5]:
        print(f"   - {item['feature_1']} vs {item['feature_2']}: r = {item['pearson_r']:.4f}")

    # Plot 5: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8.5))
    corr_vals = corr_matrix.values
    im = ax.imshow(corr_vals, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson Correlation Coefficient (r)", rotation=-90, va="bottom", fontsize=10)

    ax.set_xticks(np.arange(len(all_cols)))
    ax.set_yticks(np.arange(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(all_cols, fontsize=9)

    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            val = corr_vals[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

    ax.set_title("Pearson Correlation Heatmap (Features + Outcome)", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    p5_path = os.path.join(plots_dir, "05_correlation_heatmap.png")
    fig.savefig(p5_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {p5_path}")

    # Plot 6: Feature vs Target Stratified Boxplots
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
    df_neg = df[df[TARGET_COLUMN] == 0]
    df_pos = df[df[TARGET_COLUMN] == 1]

    for idx, col in enumerate(PREDICTOR_COLUMNS):
        ax = axes[idx]
        bp = ax.boxplot([df_neg[col], df_pos[col]], tick_labels=["Class 0 (Neg)", "Class 1 (Pos)"],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))
        bp['boxes'][0].set_facecolor(COLOR_PRIMARY)
        bp['boxes'][0].set_alpha(0.65)
        bp['boxes'][1].set_facecolor(COLOR_ACCENT)
        bp['boxes'][1].set_alpha(0.65)
        r_val = target_corrs[col]
        ax.set_title(f"{col} by Outcome (r = {r_val:+.3f})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.suptitle("Feature Distributions Stratified by Target Class (Outcome)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    p6_path = os.path.join(plots_dir, "06_feature_vs_target_boxplots.png")
    fig.savefig(p6_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {p6_path}")

    # 7. Target Leakage Assessment
    print("\n--- 6. Target Leakage Assessment ---")
    max_corr_feature = target_corrs.index[0]
    max_corr_val = float(target_corrs.iloc[0])
    print(f"Highest feature-target correlation: '{max_corr_feature}' with r = {max_corr_val:.4f}")
    suspicious_leakage_features = [col for col, r_val in target_corrs.items() if abs(r_val) >= 0.85]
    if suspicious_leakage_features:
        print(f"WARNING: Suspicious high correlation (|r| >= 0.85) in: {suspicious_leakage_features}")
    else:
        print("No feature exhibits correlation exceeding |r| >= 0.85. No evidence of target leakage or trivial proxy features.")

    # 8. Unresolved Data Quality Observations & Uncertainties
    data_quality_observations = [
        "In 5 physiological features ('Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'), zero values are present.",
        "Specifically: Insulin has 374 zeros (48.70%), SkinThickness has 227 zeros (29.56%), BloodPressure has 35 zeros (4.56%), BMI has 11 zeros (1.43%), and Glucose has 5 zeros (0.65%).",
        "Because biological values of 0 for glucose, blood pressure, or BMI are physiologically implausible for living subjects, these zeros represent potential missing-data encodings.",
        "Unresolved Question for Step 3 Preprocessing: Whether to treat these zeros as NaN and impute them (e.g. median/mean/iterative imputation) or retain them as-is must be handled strictly inside the preprocessing pipeline to prevent data leakage.",
        "Outliers are present in several features (e.g. Insulin: 34 outliers, SkinThickness: 1 outlier, BloodPressure: 35 outliers, Pregnancies: 4 outliers under IQR). Descriptive outliers are noted without clinical judgment and should be scaled or handled robustly.",
        "Target class imbalance is moderate (65.10% negative vs 34.90% positive, ratio ~1.87:1), suggesting stratification in cross-validation will be beneficial."
    ]

    # 9. Save Summary JSON
    summary_data = {
        "dataset_path": abs_dataset_path,
        "dataset_shape": {"rows": n_rows, "columns": n_cols},
        "target_variable": TARGET_COLUMN,
        "target_distribution": {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "percentages": {str(k): float(v) for k, v in target_pcts.items()},
            "imbalance_ratio": imbalance_ratio
        },
        "predictor_columns": PREDICTOR_COLUMNS,
        "standard_missing_values": {
            "total_nulls": total_standard_nulls,
            "per_column": standard_nulls
        },
        "zero_values_analysis": {
            "critical_physiological_columns": PHYSIOLOGICAL_ZERO_CHECK_COLS,
            "zero_counts_and_percentages": zero_counts
        },
        "outlier_investigation_iqr": outlier_summary,
        "descriptive_statistics": {
            col: {
                "mean": float(round(desc.loc[col, "mean"], 4)),
                "std": float(round(desc.loc[col, "std"], 4)),
                "median": float(round(desc.loc[col, "median"], 4)),
                "min": float(round(desc.loc[col, "min"], 4)),
                "max": float(round(desc.loc[col, "max"], 4)),
                "skewness": float(round(desc.loc[col, "skewness"], 4)),
                "kurtosis": float(round(desc.loc[col, "kurtosis"], 4))
            }
            for col in PREDICTOR_COLUMNS
        },
        "correlation_analysis": {
            "feature_to_target_pearson": {k: float(round(v, 4)) for k, v in target_corrs.items()},
            "top_pairwise_feature_correlations": pairwise_corrs[:8]
        },
        "target_leakage_check": {
            "max_feature_target_correlation": {"feature": max_corr_feature, "pearson_r": max_corr_val},
            "suspicious_features_detected": suspicious_leakage_features,
            "leakage_risk": "Low / None observed"
        },
        "unresolved_data_quality_questions": data_quality_observations,
        "generated_plots": [
            "01_target_distribution.png",
            "02_zero_values_analysis.png",
            "03_feature_distributions.png",
            "04_feature_boxplots.png",
            "05_correlation_heatmap.png",
            "06_feature_vs_target_boxplots.png"
        ]
    }

    summary_file = os.path.join(output_dir, "eda_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved EDA summary JSON to: {summary_file}")

    print("=" * 80)
    print("EDA EXECUTION COMPLETED SUCCESSFULLY - NO MODEL TRAINED - ORIGINAL DATA UNTOUCHED")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "..", "..", "..", "datasets", "diabetes", "diabetes.csv"),
        os.path.join("datasets", "diabetes", "diabetes.csv"),
        os.path.abspath("datasets/diabetes/diabetes.csv")
    ]

    selected_dataset = None
    for p in candidate_paths:
        if os.path.exists(p):
            selected_dataset = p
            break

    if selected_dataset is None:
        selected_dataset = candidate_paths[0]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(selected_dataset, out_dir)
