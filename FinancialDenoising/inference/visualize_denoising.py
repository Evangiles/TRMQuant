"""
Visualize Denoising Results

Compares original vs denoised data with visualizations and statistics.

Usage:
    python inference/visualize_denoising.py \
        --original train.csv \
        --denoised train_denoised.csv \
        --output_dir visualizations
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats


def load_cluster_config(config_path: Path):
    """Load cluster assignments."""
    with open(config_path, 'r') as f:
        return json.load(f)


def compute_metrics(original: np.ndarray, denoised: np.ndarray):
    """
    Compute denoising quality metrics.

    Args:
        original: Original time series
        denoised: Denoised time series

    Returns:
        Dictionary of metrics
    """
    # Total Variation (smoothness)
    tv_original = np.sum(np.abs(np.diff(original)))
    tv_denoised = np.sum(np.abs(np.diff(denoised)))
    tv_reduction = (tv_original - tv_denoised) / tv_original * 100

    # Correlation (preservation of signal)
    correlation = np.corrcoef(original, denoised)[0, 1]

    # MSE
    mse = np.mean((original - denoised) ** 2)
    rmse = np.sqrt(mse)

    # Signal-to-Noise Ratio (SNR)
    signal_power = np.var(original)
    noise_power = np.var(original - denoised)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))

    return {
        'tv_reduction_pct': tv_reduction,
        'correlation': correlation,
        'mse': mse,
        'rmse': rmse,
        'snr_db': snr
    }


def plot_comparison(
    original: np.ndarray,
    denoised: np.ndarray,
    feature_name: str,
    metrics: dict,
    save_path: Path
):
    """
    Plot original vs denoised comparison.

    Args:
        original: Original time series
        denoised: Denoised time series
        feature_name: Feature name
        metrics: Metrics dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Time series comparison
    ax = axes[0]
    ax.plot(original[:500], label='Original', alpha=0.7, linewidth=1)
    ax.plot(denoised[:500], label='Denoised', alpha=0.7, linewidth=1.5)
    ax.set_title(f'{feature_name} - Time Series Comparison (First 500 points)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Difference
    ax = axes[1]
    diff = original - denoised
    ax.plot(diff[:500], color='red', alpha=0.7, linewidth=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Difference (Original - Denoised)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Difference')
    ax.grid(True, alpha=0.3)

    # Plot 3: Distribution comparison
    ax = axes[2]
    ax.hist(original, bins=50, alpha=0.5, label='Original', density=True)
    ax.hist(denoised, bins=50, alpha=0.5, label='Denoised', density=True)
    ax.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add metrics text
    metrics_text = (
        f"TV Reduction: {metrics['tv_reduction_pct']:.2f}%\n"
        f"Correlation: {metrics['correlation']:.4f}\n"
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"SNR: {metrics['snr_db']:.2f} dB"
    )
    fig.text(0.15, 0.02, metrics_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--denoised", type=str, required=True)
    parser.add_argument("--cluster_config", type=str, default="FinancialDenoising/clustering_results/cluster_assignments.json")
    parser.add_argument("--output_dir", type=str, default="FinancialDenoising/visualizations")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of features to visualize per cluster")

    args = parser.parse_args()

    print("="*80)
    print("DENOISING VISUALIZATION")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df_original = pd.read_csv(args.original)
    df_denoised = pd.read_csv(args.denoised)
    print(f"Original: {len(df_original)} rows")
    print(f"Denoised: {len(df_denoised)} rows")

    # Load cluster configuration
    cluster_config = load_cluster_config(Path(args.cluster_config))
    n_clusters = cluster_config['metadata']['n_clusters']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each cluster
    all_metrics = {}

    for cluster_id in range(n_clusters):
        print(f"\n{'─'*80}")
        print(f"Cluster {cluster_id}")
        print('─'*80)

        cluster_key = f"cluster_{cluster_id}"
        cluster_info = cluster_config['clusters'][cluster_key]
        feature_names = cluster_info['features']
        cluster_type = cluster_info.get('type', 'random_walk')

        print(f"Type: {cluster_type}")
        print(f"Features: {len(feature_names)}")

        cluster_metrics = {}

        # Sample features to visualize
        n_viz = min(args.n_samples, len(feature_names))
        sampled_features = np.random.choice(feature_names, n_viz, replace=False)

        for feature in sampled_features:
            original = df_original[feature].values
            denoised = df_denoised[feature].values

            # Handle NaN
            original = np.nan_to_num(original, nan=0.0)
            denoised = np.nan_to_num(denoised, nan=0.0)

            # Compute metrics
            metrics = compute_metrics(original, denoised)
            cluster_metrics[feature] = metrics

            # Plot
            plot_path = output_dir / f"cluster_{cluster_id}_{feature}.png"
            plot_comparison(original, denoised, feature, metrics, plot_path)

            print(f"  {feature}:")
            print(f"    TV reduction: {metrics['tv_reduction_pct']:.2f}%")
            print(f"    Correlation: {metrics['correlation']:.4f}")
            print(f"    SNR: {metrics['snr_db']:.2f} dB")

        all_metrics[cluster_key] = cluster_metrics

    # Save metrics summary
    summary_path = output_dir / 'metrics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETED")
    print('='*80)
    print(f"Visualizations saved to: {output_dir}")
    print(f"Metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
