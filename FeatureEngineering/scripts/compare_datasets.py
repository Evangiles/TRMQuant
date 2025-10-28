"""
Compare baseline performance between raw and denoised datasets.

Usage:
    python scripts/compare_datasets.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.preprocessing import preprocess_pipeline
from src.models.lightgbm_model import cross_validate_lightgbm
import json


def main():
    print("="*80)
    print("Dataset Comparison: train.csv vs train_denoised.csv")
    print("="*80)

    datasets = {
        'raw': 'data/raw/train.csv',
        'denoised': 'data/raw/train_denoised.csv'
    }

    results = {}

    for dataset_name, input_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'='*80}")

        # 1. Preprocess
        output_path = f"data/processed/train_baseline_{dataset_name}.csv"
        print(f"\nStep 1: Preprocessing {input_path}...")
        df, metadata = preprocess_pipeline(
            input_path=input_path,
            output_path=output_path,
            config_path="configs/preprocessing.yaml"
        )

        # 2. Train LightGBM
        print(f"\nStep 2: Training LightGBM on {dataset_name}...")
        model, cv_results = cross_validate_lightgbm(
            df,
            target_col='forward_returns',
            n_splits=5,
            embargo=21,
            config_path="configs/models.yaml"
        )

        # 3. Save results
        results_dir = f"results/baseline_{dataset_name}"
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
            'n_features': metadata['n_features'],
            'train_samples': metadata['train_samples'],
            'val_samples': metadata['val_samples'],
        }

        # Save metrics
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results[dataset_name], f, indent=2)

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Metric':<30} {'Raw':<25} {'Denoised':<25} {'Improvement':<15}")
    print("-"*95)

    metrics_to_compare = [
        ('IC', 'mean_ic', 'std_ic', False),
        ('Raw Sharpe Ratio', 'mean_sharpe', 'std_sharpe', False),
        ('Adjusted Sharpe', 'mean_adjusted_sharpe', 'std_adjusted_sharpe', False),
        ('Vol Penalty', 'mean_vol_penalty', 'std_vol_penalty', True),  # Lower is better
        ('Return Penalty', 'mean_return_penalty', 'std_return_penalty', True),  # Lower is better
        ('Max Drawdown', 'mean_max_dd', 'std_max_dd', True),  # Lower is better
        ('Hit Rate', 'mean_hit_rate', 'std_hit_rate', False),
    ]

    for metric_name, mean_key, std_key, lower_better in metrics_to_compare:
        raw_mean = results['raw'][mean_key]
        raw_std = results['raw'][std_key]
        denoised_mean = results['denoised'][mean_key]
        denoised_std = results['denoised'][std_key]

        if lower_better:
            improvement = (raw_mean - denoised_mean) / abs(raw_mean) * 100
        else:
            improvement = (denoised_mean - raw_mean) / abs(raw_mean) * 100

        raw_str = f"{raw_mean:.4f}±{raw_std:.4f}"
        denoised_str = f"{denoised_mean:.4f}±{denoised_std:.4f}"
        improvement_str = f"{improvement:+.1f}%"

        print(f"{metric_name:<30} {raw_str:<25} {denoised_str:<25} {improvement_str:<15}")

    print(f"\n{'Dataset Info':<25} {'Raw':<20} {'Denoised':<20}")
    print("-"*80)
    print(f"{'Features':<25} {results['raw']['n_features']:<20} {results['denoised']['n_features']:<20}")
    print(f"{'Train Samples':<25} {results['raw']['train_samples']:<20} {results['denoised']['train_samples']:<20}")
    print(f"{'Val Samples':<25} {results['raw']['val_samples']:<20} {results['denoised']['val_samples']:<20}")

    # Save comparison
    comparison_path = "results/dataset_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("Comparison Complete!")
    print(f"{'='*80}")
    print(f"Results saved to:")
    print(f"  - results/baseline_raw/")
    print(f"  - results/baseline_denoised/")
    print(f"  - results/dataset_comparison.json")


if __name__ == "__main__":
    main()
