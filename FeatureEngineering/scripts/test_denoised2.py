"""
Test train_denoised2.csv (clean + leak-safe denoised data).

Compare with original train_denoised.csv to measure impact of leakage removal.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.models.lightgbm_model import cross_validate_lightgbm
import json
from datetime import datetime


def main():
    print("="*80)
    print("Testing train_denoised2.csv (Clean + Leak-Safe)")
    print("="*80)

    # Load clean denoised data
    data_path = "data/train_denoised2.csv"
    if not os.path.exists(data_path):
        print(f"\nError: {data_path} not found!")
        return

    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")
    print(f"  Rows: {len(df)} (from row 1006, leak-safe)")

    # Train model with cross-validation
    print(f"\n{'='*80}")
    print("Training LightGBM with 5-Fold Time Series CV")
    print(f"{'='*80}")

    model, results_df = cross_validate_lightgbm(
        df,
        target_col='forward_returns',
        n_splits=5,
        embargo=21,
        config_path="configs/models.yaml"
    )

    # Save results
    results_dir = "results/denoised2"
    os.makedirs(results_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(results_dir, "lightgbm_model.txt")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Save CV results
    cv_path = os.path.join(results_dir, "cv_results.csv")
    results_df.to_csv(cv_path, index=False)
    print(f"CV results saved to {cv_path}")

    # Calculate metrics
    metrics = {
        'data': 'train_denoised2.csv (clean + leak-safe)',
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_features': df.shape[1] - 4,  # exclude date_id and 3 targets
        'metrics': {
            'mean_ic': float(results_df['ic'].mean()),
            'std_ic': float(results_df['ic'].std()),
            'mean_sharpe': float(results_df['sharpe'].mean()),
            'std_sharpe': float(results_df['sharpe'].std()),
            'mean_adjusted_sharpe': float(results_df['adjusted_sharpe'].mean()),
            'std_adjusted_sharpe': float(results_df['adjusted_sharpe'].std()),
            'mean_max_dd': float(results_df['max_drawdown'].mean()),
            'std_max_dd': float(results_df['max_drawdown'].std()),
            'mean_hit_rate': float(results_df['hit_rate'].mean()),
            'std_hit_rate': float(results_df['hit_rate'].std()),
        }
    }

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (Clean + Leak-Safe)")
    print(f"{'='*80}")

    m = metrics['metrics']
    print(f"\nInformation Coefficient (IC):")
    print(f"  {m['mean_ic']:.4f} ± {m['std_ic']:.4f}")

    print(f"\nSharpe Ratio:")
    print(f"  {m['mean_sharpe']:.4f} ± {m['std_sharpe']:.4f}")

    print(f"\nAdjusted Sharpe Ratio:")
    print(f"  {m['mean_adjusted_sharpe']:.4f} ± {m['std_adjusted_sharpe']:.4f}")

    print(f"\nMax Drawdown:")
    print(f"  {m['mean_max_dd']:.4f} ± {m['std_max_dd']:.4f}")

    print(f"\nHit Rate:")
    print(f"  {m['mean_hit_rate']:.2%} ± {m['std_hit_rate']:.2%}")

    print(f"\n{'='*80}")
    print("Expected: Lower performance vs. leaked data, but scientifically valid")
    print(f"{'='*80}")
    print(f"\nResults saved to {results_dir}/")
    print(f"  - Model: lightgbm_model.txt")
    print(f"  - CV results: cv_results.csv")
    print(f"  - Metrics: metrics.json")


if __name__ == "__main__":
    main()
