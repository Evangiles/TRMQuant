"""
3-way comparison with Phase 1 features:

1. raw + raw_FE: Raw data with features engineered from raw
2. denoised + raw_FE: Denoised data with features from raw (cross-validation)
3. denoised + denoised_FE: Denoised data with features from denoised

Usage:
    python scripts/compare_3way.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.models.lightgbm_model import cross_validate_lightgbm
import json


def main():
    print("="*80)
    print("3-Way Comparison: Phase 1 Features")
    print("="*80)

    datasets = {
        'raw+raw_FE': 'data/features/train_phase1_raw.csv',
        'denoised+raw_FE': 'data/features/train_hybrid_denoised_raw_fe.csv',
        'denoised+denoised_FE': 'data/features/train_phase1_denoised.csv'
    }

    results = {}

    for dataset_name, data_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")

        # Load data
        print(f"Loading {data_path}...")
        df = pd.read_csv(data_path)
        print(f"  Shape: {df.shape}")

        # Count features
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
        n_features = len([c for c in df.columns if c not in exclude_cols])
        print(f"  Features: {n_features}")

        # Train LightGBM with 5-fold CV
        print(f"\nTraining LightGBM with 5-fold CV...")
        model, cv_results = cross_validate_lightgbm(
            df,
            target_col='forward_returns',
            n_splits=5,
            embargo=21,
            config_path="configs/models.yaml"
        )

        # Save results
        results_dir = f"results/phase1_{dataset_name.replace('+', '_')}"
        os.makedirs(results_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(results_dir, "lightgbm_model.txt")
        model.save_model(model_path)

        # Save CV results
        cv_path = os.path.join(results_dir, "cv_results.csv")
        cv_results.to_csv(cv_path, index=False)

        # Store summary metrics
        results[dataset_name] = {
            'mean_ic': float(cv_results['ic'].mean()),
            'std_ic': float(cv_results['ic'].std()),
            'mean_sharpe': float(cv_results['sharpe'].mean()),
            'std_sharpe': float(cv_results['sharpe'].std()),
            'mean_adjusted_sharpe': float(cv_results['adjusted_sharpe'].mean()),
            'std_adjusted_sharpe': float(cv_results['adjusted_sharpe'].std()),
            'mean_vol_penalty': float(cv_results['vol_penalty'].mean()),
            'std_vol_penalty': float(cv_results['vol_penalty'].std()),
            'mean_return_penalty': float(cv_results['return_penalty'].mean()),
            'std_return_penalty': float(cv_results['return_penalty'].std()),
            'mean_max_dd': float(cv_results['max_drawdown'].mean()),
            'std_max_dd': float(cv_results['max_drawdown'].std()),
            'mean_hit_rate': float(cv_results['hit_rate'].mean()),
            'std_hit_rate': float(cv_results['hit_rate'].std()),
            'n_features': n_features,
        }

        # Save metrics
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results[dataset_name], f, indent=2)

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL 3-WAY COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Metric':<30} {'raw+raw_FE':<25} {'denoised+raw_FE':<25} {'denoised+denoised_FE':<25}")
    print("-"*105)

    metrics_to_compare = [
        ('IC', 'mean_ic', 'std_ic'),
        ('Raw Sharpe', 'mean_sharpe', 'std_sharpe'),
        ('Adjusted Sharpe', 'mean_adjusted_sharpe', 'std_adjusted_sharpe'),
        ('Vol Penalty', 'mean_vol_penalty', 'std_vol_penalty'),
        ('Return Penalty', 'mean_return_penalty', 'std_return_penalty'),
        ('Max Drawdown', 'mean_max_dd', 'std_max_dd'),
        ('Hit Rate', 'mean_hit_rate', 'std_hit_rate'),
    ]

    for metric_name, mean_key, std_key in metrics_to_compare:
        row = f"{metric_name:<30}"
        for ds_name in ['raw+raw_FE', 'denoised+raw_FE', 'denoised+denoised_FE']:
            mean_val = results[ds_name][mean_key]
            std_val = results[ds_name][std_key]
            val_str = f"{mean_val:.4f}±{std_val:.4f}"
            row += f" {val_str:<25}"
        print(row)

    # Dataset info
    print(f"\n{'Dataset Info':<30} {'raw+raw_FE':<25} {'denoised+raw_FE':<25} {'denoised+denoised_FE':<25}")
    print("-"*105)
    row = f"{'Features':<30}"
    for ds_name in ['raw+raw_FE', 'denoised+raw_FE', 'denoised+denoised_FE']:
        row += f" {results[ds_name]['n_features']:<25}"
    print(row)

    # Improvement analysis
    print(f"\n{'='*80}")
    print("IMPROVEMENT ANALYSIS (vs raw+raw_FE baseline)")
    print(f"{'='*80}")

    baseline = results['raw+raw_FE']

    for ds_name in ['denoised+raw_FE', 'denoised+denoised_FE']:
        print(f"\n{ds_name}:")
        print("-"*40)

        for metric_name, mean_key, std_key in metrics_to_compare:
            baseline_val = baseline[mean_key]
            current_val = results[ds_name][mean_key]

            # Determine if lower is better
            lower_better = metric_name in ['Vol Penalty', 'Return Penalty', 'Max Drawdown']

            if lower_better:
                improvement = (baseline_val - current_val) / abs(baseline_val) * 100
            else:
                improvement = (current_val - baseline_val) / abs(baseline_val) * 100

            improvement_str = f"{improvement:+.1f}%"
            print(f"  {metric_name:<20}: {improvement_str}")

    # Save comparison
    comparison_path = "results/phase1_3way_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("3-Way Comparison Complete!")
    print(f"{'='*80}")
    print(f"Results saved to:")
    print(f"  - results/phase1_raw_raw_FE/")
    print(f"  - results/phase1_denoised_raw_FE/")
    print(f"  - results/phase1_denoised_denoised_FE/")
    print(f"  - results/phase1_3way_comparison.json")


if __name__ == "__main__":
    main()
