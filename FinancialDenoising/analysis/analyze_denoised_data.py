"""
Denoised Data Characteristics Analysis

Comprehensive analysis comparing Original vs Denoised datasets:
- Signal-to-Noise Ratio (SNR)
- Feature Correlation Changes
- Temporal Dependencies (ACF/PACF)
- Predictability (Information Coefficient)
- Cluster-specific Characteristics
- Statistical Tests (Stationarity, Normality, Variance)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def get_feature_columns(df):
    """Extract feature columns (exclude metadata)."""
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def calculate_snr(signal):
    """
    Calculate Signal-to-Noise Ratio.
    SNR = mean² / variance
    """
    mean_power = np.mean(signal) ** 2
    variance = np.var(signal)
    if variance == 0:
        return np.inf
    snr = mean_power / variance
    return snr


def snr_analysis(df_original, df_denoised, feature_cols, output_dir):
    """Analyze SNR improvement for each feature."""
    print("\n" + "="*80)
    print("1. SIGNAL-TO-NOISE RATIO ANALYSIS")
    print("="*80)

    snr_original = {}
    snr_denoised = {}
    snr_improvement = {}

    for col in feature_cols:
        orig_data = df_original[col].fillna(0).values
        denoise_data = df_denoised[col].fillna(0).values

        snr_orig = calculate_snr(orig_data)
        snr_den = calculate_snr(denoise_data)

        snr_original[col] = snr_orig
        snr_denoised[col] = snr_den
        snr_improvement[col] = ((snr_den - snr_orig) / abs(snr_orig) * 100) if snr_orig != 0 else 0

    # Summary statistics
    avg_snr_orig = np.mean(list(snr_original.values()))
    avg_snr_denoise = np.mean(list(snr_denoised.values()))
    avg_improvement = np.mean(list(snr_improvement.values()))

    print(f"\nAverage SNR (Original): {avg_snr_orig:.6f}")
    print(f"Average SNR (Denoised): {avg_snr_denoise:.6f}")
    print(f"Average Improvement: {avg_improvement:.2f}%")

    # Top 10 improved features
    top_improved = sorted(snr_improvement.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Improved Features:")
    for feat, imp in top_improved:
        print(f"  {feat}: +{imp:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # SNR comparison
    features = list(snr_original.keys())[:30]  # Top 30 for visibility
    x = np.arange(len(features))
    width = 0.35

    axes[0].bar(x - width/2, [snr_original[f] for f in features], width, label='Original', alpha=0.8)
    axes[0].bar(x + width/2, [snr_denoised[f] for f in features], width, label='Denoised', alpha=0.8)
    axes[0].set_xlabel('Features (Top 30)')
    axes[0].set_ylabel('SNR')
    axes[0].set_title('SNR Comparison: Original vs Denoised')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=90, labelsize=6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(features, rotation=90, ha='right')

    # Improvement distribution
    improvements = list(snr_improvement.values())
    axes[1].hist(improvements, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(avg_improvement, color='red', linestyle='--', label=f'Mean: {avg_improvement:.2f}%')
    axes[1].set_xlabel('SNR Improvement (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of SNR Improvements')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'snr_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'snr_original': snr_original,
        'snr_denoised': snr_denoised,
        'snr_improvement': snr_improvement,
        'avg_improvement': avg_improvement
    }


def correlation_analysis(df_original, df_denoised, feature_cols, output_dir):
    """Analyze feature correlation changes."""
    print("\n" + "="*80)
    print("2. FEATURE CORRELATION ANALYSIS")
    print("="*80)

    # Compute correlation matrices
    corr_orig = df_original[feature_cols].fillna(0).corr()
    corr_denoise = df_denoised[feature_cols].fillna(0).corr()
    corr_diff = corr_denoise - corr_orig

    # Statistics
    avg_corr_orig = np.abs(corr_orig.values[np.triu_indices_from(corr_orig.values, k=1)]).mean()
    avg_corr_denoise = np.abs(corr_denoise.values[np.triu_indices_from(corr_denoise.values, k=1)]).mean()

    print(f"\nAverage Absolute Correlation (Original): {avg_corr_orig:.4f}")
    print(f"Average Absolute Correlation (Denoised): {avg_corr_denoise:.4f}")
    print(f"Change: {(avg_corr_denoise - avg_corr_orig):.4f}")

    # Count strong correlations
    strong_threshold = 0.7
    strong_orig = (np.abs(corr_orig.values) > strong_threshold).sum() / 2  # Upper triangle
    strong_denoise = (np.abs(corr_denoise.values) > strong_threshold).sum() / 2

    print(f"\nStrong Correlations (|r| > {strong_threshold}):")
    print(f"  Original: {int(strong_orig)}")
    print(f"  Denoised: {int(strong_denoise)}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Original correlation (sample 30x30 for visibility)
    sample_features = feature_cols[:30]
    corr_orig_sample = df_original[sample_features].fillna(0).corr()
    sns.heatmap(corr_orig_sample, ax=axes[0], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Original Correlation (30x30 sample)')

    # Denoised correlation
    corr_denoise_sample = df_denoised[sample_features].fillna(0).corr()
    sns.heatmap(corr_denoise_sample, ax=axes[1], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Denoised Correlation (30x30 sample)')

    # Difference
    corr_diff_sample = corr_denoise_sample - corr_orig_sample
    sns.heatmap(corr_diff_sample, ax=axes[2], cmap='RdBu_r', center=0,
                square=True, cbar_kws={'label': 'Correlation Difference'})
    axes[2].set_title('Correlation Change (Denoised - Original)')

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'avg_corr_original': avg_corr_orig,
        'avg_corr_denoised': avg_corr_denoise,
        'strong_correlations_original': int(strong_orig),
        'strong_correlations_denoised': int(strong_denoise)
    }


def temporal_dependency_analysis(df_original, df_denoised, feature_cols, output_dir):
    """Analyze autocorrelation and temporal dependencies."""
    print("\n" + "="*80)
    print("3. TEMPORAL DEPENDENCY ANALYSIS")
    print("="*80)

    # Select a few representative features
    sample_features = feature_cols[:5]
    lags = 40

    fig, axes = plt.subplots(len(sample_features), 2, figsize=(14, 3*len(sample_features)))

    acf_results = {'original': {}, 'denoised': {}}

    for idx, feat in enumerate(sample_features):
        orig_data = df_original[feat].fillna(0).values
        denoise_data = df_denoised[feat].fillna(0).values

        # ACF
        acf_orig = acf(orig_data, nlags=lags, fft=True)
        acf_denoise = acf(denoise_data, nlags=lags, fft=True)

        acf_results['original'][feat] = acf_orig
        acf_results['denoised'][feat] = acf_denoise

        # Plot
        axes[idx, 0].plot(acf_orig, label='Original', alpha=0.7)
        axes[idx, 0].plot(acf_denoise, label='Denoised', alpha=0.7)
        axes[idx, 0].axhline(y=0, linestyle='--', color='gray', alpha=0.5)
        axes[idx, 0].fill_between(range(lags+1), -1.96/np.sqrt(len(orig_data)),
                                   1.96/np.sqrt(len(orig_data)), alpha=0.2)
        axes[idx, 0].set_title(f'ACF: {feat}')
        axes[idx, 0].set_xlabel('Lag')
        axes[idx, 0].set_ylabel('Autocorrelation')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        # ACF Difference
        acf_diff = acf_denoise - acf_orig
        axes[idx, 1].bar(range(lags+1), acf_diff, alpha=0.7)
        axes[idx, 1].axhline(y=0, linestyle='--', color='red')
        axes[idx, 1].set_title(f'ACF Change (Denoised - Original)')
        axes[idx, 1].set_xlabel('Lag')
        axes[idx, 1].set_ylabel('ACF Difference')
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_dependency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Summary statistics
    avg_acf_decay_orig = np.mean([np.mean(np.abs(acf_results['original'][f][1:10]))
                                   for f in sample_features])
    avg_acf_decay_denoise = np.mean([np.mean(np.abs(acf_results['denoised'][f][1:10]))
                                      for f in sample_features])

    print(f"\nAverage ACF (lag 1-10):")
    print(f"  Original: {avg_acf_decay_orig:.4f}")
    print(f"  Denoised: {avg_acf_decay_denoise:.4f}")

    return {
        'avg_acf_original': avg_acf_decay_orig,
        'avg_acf_denoised': avg_acf_decay_denoise,
        'acf_results': acf_results
    }


def predictability_analysis(df_original, df_denoised, feature_cols, output_dir):
    """Analyze predictability using Information Coefficient."""
    print("\n" + "="*80)
    print("4. PREDICTABILITY ANALYSIS")
    print("="*80)

    target_col = 'forward_returns'

    if target_col not in df_original.columns:
        print("Warning: forward_returns not found, skipping predictability analysis")
        return {}

    ic_original = {}
    ic_denoised = {}

    for feat in feature_cols:
        # Original IC
        valid_mask_orig = ~(df_original[feat].isna() | df_original[target_col].isna())
        if valid_mask_orig.sum() > 0:
            ic_orig = np.corrcoef(df_original.loc[valid_mask_orig, feat],
                                  df_original.loc[valid_mask_orig, target_col])[0, 1]
            ic_original[feat] = ic_orig

        # Denoised IC
        valid_mask_den = ~(df_denoised[feat].isna() | df_denoised[target_col].isna())
        if valid_mask_den.sum() > 0:
            ic_den = np.corrcoef(df_denoised.loc[valid_mask_den, feat],
                                 df_denoised.loc[valid_mask_den, target_col])[0, 1]
            ic_denoised[feat] = ic_den

    # Calculate statistics
    avg_ic_orig = np.mean([abs(ic) for ic in ic_original.values() if not np.isnan(ic)])
    avg_ic_denoise = np.mean([abs(ic) for ic in ic_denoised.values() if not np.isnan(ic)])

    print(f"\nAverage Absolute IC:")
    print(f"  Original: {avg_ic_orig:.4f}")
    print(f"  Denoised: {avg_ic_denoise:.4f}")
    print(f"  Improvement: {((avg_ic_denoise - avg_ic_orig) / avg_ic_orig * 100):.2f}%")

    # Top predictive features
    ic_improvement = {feat: (abs(ic_denoised.get(feat, 0)) - abs(ic_original.get(feat, 0)))
                      for feat in ic_original.keys()}
    top_improved = sorted(ic_improvement.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTop 10 Features with Improved Predictability:")
    for feat, imp in top_improved:
        print(f"  {feat}: +{imp:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IC distribution
    axes[0].hist([abs(ic) for ic in ic_original.values() if not np.isnan(ic)],
                 bins=50, alpha=0.6, label='Original', edgecolor='black')
    axes[0].hist([abs(ic) for ic in ic_denoised.values() if not np.isnan(ic)],
                 bins=50, alpha=0.6, label='Denoised', edgecolor='black')
    axes[0].axvline(avg_ic_orig, color='blue', linestyle='--', label=f'Mean Original: {avg_ic_orig:.4f}')
    axes[0].axvline(avg_ic_denoise, color='orange', linestyle='--', label=f'Mean Denoised: {avg_ic_denoise:.4f}')
    axes[0].set_xlabel('Absolute Information Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('IC Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IC scatter
    common_features = set(ic_original.keys()) & set(ic_denoised.keys())
    ic_orig_vals = [abs(ic_original[f]) for f in common_features if not np.isnan(ic_original[f])]
    ic_den_vals = [abs(ic_denoised[f]) for f in common_features if not np.isnan(ic_denoised[f])]

    axes[1].scatter(ic_orig_vals, ic_den_vals, alpha=0.5)
    axes[1].plot([0, max(ic_orig_vals)], [0, max(ic_orig_vals)], 'r--', label='y=x')
    axes[1].set_xlabel('Original |IC|')
    axes[1].set_ylabel('Denoised |IC|')
    axes[1].set_title('IC Comparison (Denoised vs Original)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'predictability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'avg_ic_original': avg_ic_orig,
        'avg_ic_denoised': avg_ic_denoise,
        'ic_improvement_pct': ((avg_ic_denoise - avg_ic_orig) / avg_ic_orig * 100) if avg_ic_orig > 0 else 0
    }


def stationarity_analysis(df_original, df_denoised, feature_cols, output_dir):
    """Test stationarity using Augmented Dickey-Fuller test."""
    print("\n" + "="*80)
    print("5. STATIONARITY ANALYSIS")
    print("="*80)

    # Sample features for testing
    sample_features = feature_cols[:20]

    stationary_orig = 0
    stationary_denoise = 0

    adf_results = {'original': {}, 'denoised': {}}

    for feat in sample_features:
        # Original ADF test
        try:
            orig_data = df_original[feat].fillna(0).values
            adf_orig = adfuller(orig_data, maxlag=10)
            adf_results['original'][feat] = adf_orig[1]  # p-value
            if adf_orig[1] < 0.05:  # Stationary
                stationary_orig += 1
        except:
            pass

        # Denoised ADF test
        try:
            denoise_data = df_denoised[feat].fillna(0).values
            adf_denoise = adfuller(denoise_data, maxlag=10)
            adf_results['denoised'][feat] = adf_denoise[1]  # p-value
            if adf_denoise[1] < 0.05:  # Stationary
                stationary_denoise += 1
        except:
            pass

    print(f"\nStationary Features (p < 0.05):")
    print(f"  Original: {stationary_orig}/{len(sample_features)} ({stationary_orig/len(sample_features)*100:.1f}%)")
    print(f"  Denoised: {stationary_denoise}/{len(sample_features)} ({stationary_denoise/len(sample_features)*100:.1f}%)")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    features = list(adf_results['original'].keys())
    x = np.arange(len(features))
    width = 0.35

    p_vals_orig = [adf_results['original'][f] for f in features]
    p_vals_denoise = [adf_results['denoised'][f] for f in features]

    ax.bar(x - width/2, p_vals_orig, width, label='Original', alpha=0.8)
    ax.bar(x + width/2, p_vals_denoise, width, label='Denoised', alpha=0.8)
    ax.axhline(y=0.05, color='red', linestyle='--', label='Significance (p=0.05)')
    ax.set_xlabel('Features')
    ax.set_ylabel('ADF Test p-value')
    ax.set_title('Stationarity Test (ADF) Results')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=90, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'stationarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'stationary_pct_original': stationary_orig / len(sample_features) * 100,
        'stationary_pct_denoised': stationary_denoise / len(sample_features) * 100
    }


def generate_report(all_results, output_dir):
    """Generate comprehensive markdown report."""
    print("\n" + "="*80)
    print("GENERATING ANALYSIS REPORT")
    print("="*80)

    report = []
    report.append("# Denoised Data Characteristics Analysis Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("---\n\n")
    report.append("## Executive Summary\n\n")

    # SNR
    if 'snr' in all_results:
        snr_imp = all_results['snr']['avg_improvement']
        report.append(f"### Signal-to-Noise Ratio\n")
        report.append(f"- **Average SNR Improvement**: {snr_imp:.2f}%\n")
        if snr_imp > 20:
            report.append(f"- ✅ **Significant improvement** in signal quality\n\n")
        elif snr_imp > 0:
            report.append(f"- ⚠️ **Marginal improvement** in signal quality\n\n")
        else:
            report.append(f"- ❌ **No improvement** or degradation in signal quality\n\n")

    # Correlation
    if 'correlation' in all_results:
        corr_orig = all_results['correlation']['avg_corr_original']
        corr_den = all_results['correlation']['avg_corr_denoised']
        report.append(f"### Feature Correlations\n")
        report.append(f"- Original: {corr_orig:.4f}\n")
        report.append(f"- Denoised: {corr_den:.4f}\n")
        report.append(f"- Change: {(corr_den - corr_orig):.4f}\n\n")

    # Temporal
    if 'temporal' in all_results:
        acf_orig = all_results['temporal']['avg_acf_original']
        acf_den = all_results['temporal']['avg_acf_denoised']
        report.append(f"### Temporal Dependencies\n")
        report.append(f"- Original ACF (lag 1-10): {acf_orig:.4f}\n")
        report.append(f"- Denoised ACF (lag 1-10): {acf_den:.4f}\n\n")

    # Predictability
    if 'predictability' in all_results:
        ic_imp = all_results['predictability'].get('ic_improvement_pct', 0)
        report.append(f"### Predictability (Information Coefficient)\n")
        report.append(f"- **IC Improvement**: {ic_imp:.2f}%\n")
        if ic_imp > 20:
            report.append(f"- ✅ **Significant improvement** in predictive power\n\n")
        elif ic_imp > 0:
            report.append(f"- ⚠️ **Marginal improvement** in predictive power\n\n")
        else:
            report.append(f"- ❌ **No improvement** in predictive power\n\n")

    # Stationarity
    if 'stationarity' in all_results:
        stat_orig = all_results['stationarity']['stationary_pct_original']
        stat_den = all_results['stationarity']['stationary_pct_denoised']
        report.append(f"### Stationarity\n")
        report.append(f"- Original: {stat_orig:.1f}% stationary features\n")
        report.append(f"- Denoised: {stat_den:.1f}% stationary features\n\n")

    report.append("---\n\n")
    report.append("## Detailed Results\n\n")
    report.append("See visualization files in `results/` directory:\n")
    report.append("- `snr_analysis.png`\n")
    report.append("- `correlation_analysis.png`\n")
    report.append("- `temporal_dependency_analysis.png`\n")
    report.append("- `predictability_analysis.png`\n")
    report.append("- `stationarity_analysis.png`\n\n")

    report.append("---\n\n")
    report.append("## Recommendations\n\n")

    # Generate recommendations based on results
    if 'predictability' in all_results:
        ic_imp = all_results['predictability'].get('ic_improvement_pct', 0)
        if ic_imp > 20:
            report.append("### ✅ Denoising is Highly Effective\n")
            report.append("- Strong improvement in predictive power\n")
            report.append("- Recommend using denoised data for model training\n")
            report.append("- Consider Transformer-based architecture to leverage clean signals\n\n")
        elif ic_imp > 0:
            report.append("### ⚠️ Denoising Shows Moderate Effect\n")
            report.append("- Marginal improvement in predictive power\n")
            report.append("- Consider ensemble approach (original + denoised)\n")
            report.append("- May need hyperparameter tuning for denoiser\n\n")
        else:
            report.append("### ❌ Denoising Not Effective\n")
            report.append("- No improvement or degradation in predictive power\n")
            report.append("- Recommend using original data\n")
            report.append("- Consider different denoising approach or skip denoising\n\n")

    report_text = "".join(report)

    # Save report
    report_path = output_dir / "DENOISED_ANALYSIS_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved: {report_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Analyze Denoised Data Characteristics")
    parser.add_argument("--original", type=str, required=True, help="Path to original CSV")
    parser.add_argument("--denoised", type=str, required=True, help="Path to denoised CSV")
    parser.add_argument("--output", type=str, default="TinyRecursiveModels/analysis/results",
                        help="Output directory for results")

    args = parser.parse_args()

    # Load data
    print("Loading datasets...")
    df_original = pd.read_csv(args.original)
    df_denoised = pd.read_csv(args.denoised)

    print(f"Original: {len(df_original)} rows")
    print(f"Denoised: {len(df_denoised)} rows")

    # Get feature columns
    feature_cols = get_feature_columns(df_original)
    print(f"Features: {len(feature_cols)}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all analyses
    all_results = {}

    all_results['snr'] = snr_analysis(df_original, df_denoised, feature_cols, output_dir)
    all_results['correlation'] = correlation_analysis(df_original, df_denoised, feature_cols, output_dir)
    all_results['temporal'] = temporal_dependency_analysis(df_original, df_denoised, feature_cols, output_dir)
    all_results['predictability'] = predictability_analysis(df_original, df_denoised, feature_cols, output_dir)
    all_results['stationarity'] = stationarity_analysis(df_original, df_denoised, feature_cols, output_dir)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'avg_snr_improvement': all_results['snr']['avg_improvement'],
        'avg_corr_original': all_results['correlation']['avg_corr_original'],
        'avg_corr_denoised': all_results['correlation']['avg_corr_denoised'],
        'avg_ic_original': all_results['predictability']['avg_ic_original'],
        'avg_ic_denoised': all_results['predictability']['avg_ic_denoised'],
        'ic_improvement_pct': all_results['predictability']['ic_improvement_pct'],
        'stationary_pct_original': all_results['stationarity']['stationary_pct_original'],
        'stationary_pct_denoised': all_results['stationarity']['stationary_pct_denoised']
    }])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    print(f"\nMetrics saved: {output_dir / 'metrics.csv'}")

    # Generate report
    generate_report(all_results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
