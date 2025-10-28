"""
Train and evaluate all models on specified feature set.

Usage:
    python scripts/train_all_models.py --data baseline --models lightgbm,xgboost,ridge
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.models.lightgbm_model import cross_validate_lightgbm


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--data",
        type=str,
        default="baseline",
        choices=["baseline", "phase1", "phase2", "phase3"],
        help="Which feature set to use"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lightgbm",
        help="Comma-separated list of models (lightgbm,xgboost,ridge)"
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV splits"
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=21,
        help="Embargo days between train/test"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print(f"Training Models on {args.data.upper()} Features")
    print("="*80)
    print(f"Data: {args.data}")
    print(f"Models: {args.models}")
    print(f"CV splits: {args.n_splits}")
    print(f"Embargo: {args.embargo} days")
    print("="*80)

    # Load data
    data_path = f"data/processed/train_{args.data}.csv"
    if not os.path.exists(data_path):
        print(f"\nError: {data_path} not found!")
        print(f"Please run: python scripts/create_{args.data}_features.py")
        return

    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # Parse models
    models_to_train = [m.strip() for m in args.models.split(',')]

    # Results directory
    results_dir = f"results/{args.data}"
    os.makedirs(results_dir, exist_ok=True)

    # Train each model
    all_results = {}

    for model_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*80}")

        if model_name == "lightgbm":
            model, results_df = cross_validate_lightgbm(
                df,
                target_col='forward_returns',
                n_splits=args.n_splits,
                embargo=args.embargo,
                config_path="configs/models.yaml"
            )

            # Save model
            model_path = os.path.join(results_dir, f"{model_name}_model.txt")
            model.save_model(model_path)
            print(f"\nModel saved to {model_path}")

            # Save CV results
            cv_path = os.path.join(results_dir, f"{model_name}_cv_results.csv")
            results_df.to_csv(cv_path, index=False)
            print(f"CV results saved to {cv_path}")

            # Store metrics
            all_results[model_name] = {
                'mean_ic': float(results_df['ic'].mean()),
                'std_ic': float(results_df['ic'].std()),
                'mean_sharpe': float(results_df['sharpe'].mean()),
                'std_sharpe': float(results_df['sharpe'].std()),
                'mean_max_dd': float(results_df['max_drawdown'].mean()),
                'std_max_dd': float(results_df['max_drawdown'].std()),
                'mean_hit_rate': float(results_df['hit_rate'].mean()),
                'std_hit_rate': float(results_df['hit_rate'].std()),
            }

        elif model_name == "xgboost":
            print("XGBoost implementation coming soon...")
            continue

        elif model_name == "ridge":
            print("Ridge implementation coming soon...")
            continue

        else:
            print(f"Unknown model: {model_name}")
            continue

    # Save summary metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'data': args.data,
            'timestamp': datetime.now().isoformat(),
            'models': all_results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Results saved to {results_dir}/")
    print(f"  - Model files: *.txt")
    print(f"  - CV results: *_cv_results.csv")
    print(f"  - Summary: metrics.json")

    # Print summary table
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Mean IC':<12} {'Mean Sharpe':<12} {'Mean MaxDD':<12} {'Hit Rate':<12}")
    print("-"*80)
    for model_name, metrics in all_results.items():
        print(f"{model_name:<15} "
              f"{metrics['mean_ic']:.4f}±{metrics['std_ic']:.4f}  "
              f"{metrics['mean_sharpe']:.4f}±{metrics['std_sharpe']:.4f}  "
              f"{metrics['mean_max_dd']:.4f}±{metrics['std_max_dd']:.4f}  "
              f"{metrics['mean_hit_rate']:.2%}±{metrics['std_hit_rate']:.2%}")


if __name__ == "__main__":
    main()
