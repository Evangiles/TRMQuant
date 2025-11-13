"""
Validate Denoising Performance with ML Models

Uses Purged/Embargo Cross-Validation to prevent data leakage
in time series financial data.

Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not installed, skipping")

import warnings
warnings.filterwarnings('ignore')


def get_feature_target_columns(df):
    """Extract feature and target columns."""
    # Exclude date_id and target columns
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Target: forward_returns
    target_col = 'forward_returns'

    return feature_cols, target_col


def purged_embargo_cv(df, n_splits=5, embargo_pct=0.01, purge_pct=0.01):
    """
    Purged/Embargo Cross-Validation for time series.

    Args:
        df: DataFrame with time series data
        n_splits: Number of CV folds
        embargo_pct: Embargo period as fraction of total data (default: 1%)
        purge_pct: Purging period as fraction of total data (default: 1%)

    Yields:
        (train_idx, test_idx) tuples
    """
    n_samples = len(df)
    embargo_size = int(n_samples * embargo_pct)
    purge_size = int(n_samples * purge_pct)

    test_size = n_samples // n_splits

    for i in range(n_splits):
        # Test set
        test_start = i * test_size
        test_end = test_start + test_size
        test_idx = np.arange(test_start, test_end)

        # Purge: remove samples close to test set boundaries
        purge_start = max(0, test_start - purge_size)
        purge_end = min(n_samples, test_end + purge_size)

        # Embargo: remove additional samples after test set
        embargo_end = min(n_samples, test_end + embargo_size)

        # Train set: everything except test, purge, and embargo regions
        train_idx = np.concatenate([
            np.arange(0, purge_start),
            np.arange(embargo_end, n_samples)
        ])

        # Ensure no overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        yield train_idx, test_idx


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a single model."""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Information Coefficient (IC) - correlation
    ic = np.corrcoef(y_test, y_pred)[0, 1]

    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'ic': ic
    }


def run_cv_experiment(df, feature_cols, target_col, n_splits=5):
    """Run cross-validation experiment with multiple models."""

    # Prepare data
    X = df[feature_cols].fillna(0).values
    y = df[target_col].fillna(0).values

    # Models
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
    }

    if HAS_CATBOOST:
        models['CatBoost'] = cb.CatBoostRegressor(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )

    # Results storage
    results = []

    print(f"\nRunning {n_splits}-fold Purged/Embargo CV...")
    print("─" * 80)

    for fold, (train_idx, test_idx) in enumerate(purged_embargo_cv(df, n_splits=n_splits)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Evaluate each model
        for model_name, model in models.items():
            result = evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
            result['fold'] = fold + 1
            results.append(result)

            print(f"  {model_name:15s} - RMSE: {result['rmse']:.6f}, IC: {result['ic']:.4f}, R²: {result['r2']:.4f}")

    return pd.DataFrame(results)


def summarize_results(results_df):
    """Compute average metrics across folds."""
    summary = results_df.groupby('model').agg({
        'mse': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'ic': ['mean', 'std']
    }).round(6)

    return summary


def compare_datasets(original_csv, denoised_csv, n_splits=5):
    """Compare original vs denoised datasets."""

    print("=" * 80)
    print("DENOISING VALIDATION WITH ML MODELS")
    print("=" * 80)

    # Load data
    print("\nLoading datasets...")
    df_original = pd.read_csv(original_csv)
    df_denoised = pd.read_csv(denoised_csv)

    print(f"Original: {len(df_original)} rows")
    print(f"Denoised: {len(df_denoised)} rows")

    # Get columns
    feature_cols, target_col = get_feature_target_columns(df_original)
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")

    # Experiment 1: Original data
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ORIGINAL DATA")
    print("=" * 80)
    results_original = run_cv_experiment(df_original, feature_cols, target_col, n_splits)

    # Experiment 2: Denoised data
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: DENOISED DATA")
    print("=" * 80)
    results_denoised = run_cv_experiment(df_denoised, feature_cols, target_col, n_splits)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: AVERAGE METRICS ACROSS FOLDS")
    print("=" * 80)

    print("\n[Original Data]:")
    print(summarize_results(results_original))

    print("\n[Denoised Data]:")
    print(summarize_results(results_denoised))

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: DENOISED vs ORIGINAL")
    print("=" * 80)

    summary_orig = results_original.groupby('model')['ic'].mean()
    summary_denoise = results_denoised.groupby('model')['ic'].mean()

    comparison = pd.DataFrame({
        'Original_IC': summary_orig,
        'Denoised_IC': summary_denoise,
        'Improvement': summary_denoise - summary_orig,
        'Improvement_%': ((summary_denoise - summary_orig) / np.abs(summary_orig) * 100).fillna(0)
    }).round(6)

    print("\n[Information Coefficient (IC) Comparison]:")
    print(comparison)

    # Save results
    output_dir = Path("TinyRecursiveModels/evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_original.to_csv(output_dir / "original_cv_results.csv", index=False)
    results_denoised.to_csv(output_dir / "denoised_cv_results.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv")

    print(f"\n[SUCCESS] Results saved to {output_dir}/")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    avg_improvement = comparison['Improvement_%'].mean()
    if avg_improvement > 5:
        print(f"[EFFECTIVE] DENOISING EFFECTIVE: {avg_improvement:.2f}% average IC improvement")
    elif avg_improvement > 0:
        print(f"[MARGINAL] DENOISING MARGINAL: {avg_improvement:.2f}% average IC improvement")
    else:
        print(f"[INEFFECTIVE] DENOISING INEFFECTIVE: {avg_improvement:.2f}% average IC change")
        print("   Consider retraining with more epochs or different hyperparameters")


def main():
    parser = argparse.ArgumentParser(description="Validate denoising with ML models")
    parser.add_argument("--original", type=str, required=True, help="Path to original CSV")
    parser.add_argument("--denoised", type=str, required=True, help="Path to denoised CSV")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--embargo_pct", type=float, default=0.01, help="Embargo period (fraction)")
    parser.add_argument("--purge_pct", type=float, default=0.01, help="Purge period (fraction)")

    args = parser.parse_args()

    compare_datasets(
        args.original,
        args.denoised,
        n_splits=args.n_splits
    )


if __name__ == "__main__":
    main()
