"""
Validate Denoising with Trading Signal Performance

Compares original vs denoised data using actual trading signals and portfolio metrics:
- Sharpe Ratio
- Cumulative Returns
- Maximum Drawdown (MDD)
- Win Rate
- Profit Factor

Uses Purged/Embargo Cross-Validation for proper time series validation.
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
    print("[WARNING] CatBoost not installed, skipping")

import warnings
warnings.filterwarnings('ignore')


def get_feature_target_columns(df):
    """Extract feature and target columns."""
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target_col = 'forward_returns'
    return feature_cols, target_col


def purged_embargo_cv(df, n_splits=5, embargo_pct=0.01, purge_pct=0.01):
    """Purged/Embargo Cross-Validation for time series."""
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

        assert len(np.intersect1d(train_idx, test_idx)) == 0
        yield train_idx, test_idx


def generate_trading_signals(predictions, returns, long_pct=0.2, short_pct=0.2):
    """
    Generate long/short signals based on model predictions.

    Args:
        predictions: Model predicted returns
        returns: Actual forward returns
        long_pct: Top percentile to go long (default: 20%)
        short_pct: Bottom percentile to go short (default: 20%)

    Returns:
        dict with trading metrics
    """
    # Rank predictions
    pred_rank = pd.Series(predictions).rank(pct=True)

    # Generate signals: 1 (long), -1 (short), 0 (neutral)
    signals = np.zeros_like(predictions)
    signals[pred_rank >= (1 - long_pct)] = 1  # Top 20%
    signals[pred_rank <= short_pct] = -1      # Bottom 20%

    # Calculate returns for each position
    position_returns = signals * returns

    # Only consider non-zero positions
    active_returns = position_returns[signals != 0]

    if len(active_returns) == 0:
        return None

    # Portfolio metrics
    mean_return = np.mean(active_returns)
    std_return = np.std(active_returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    # Cumulative returns
    cumulative_return = np.sum(active_returns)

    # Maximum Drawdown
    cumsum = np.cumsum(active_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    max_drawdown = np.min(drawdown)

    # Win rate
    wins = active_returns > 0
    win_rate = np.mean(wins)

    # Profit factor
    gross_profit = np.sum(active_returns[active_returns > 0])
    gross_loss = np.abs(np.sum(active_returns[active_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        'sharpe_ratio': sharpe_ratio,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'mean_return': mean_return,
        'n_trades': len(active_returns)
    }


def evaluate_trading_model(model_name, model, X_train, y_train, X_test, y_test):
    """Train model and evaluate trading performance."""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Generate trading signals and calculate metrics
    trading_metrics = generate_trading_signals(y_pred, y_test)

    if trading_metrics is None:
        return None

    # Add model info
    trading_metrics['model'] = model_name

    # Also include IC for reference
    ic = np.corrcoef(y_test, y_pred)[0, 1]
    trading_metrics['ic'] = ic

    return trading_metrics


def run_trading_cv_experiment(df, feature_cols, target_col, n_splits=5):
    """Run cross-validation with trading signal evaluation."""

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

    print(f"\nRunning {n_splits}-fold Purged/Embargo CV with Trading Signals...")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(purged_embargo_cv(df, n_splits=n_splits)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Evaluate each model
        for model_name, model in models.items():
            result = evaluate_trading_model(model_name, model, X_train, y_train, X_test, y_test)

            if result is not None:
                result['fold'] = fold + 1
                results.append(result)

                print(f"  {model_name:15s} - Sharpe: {result['sharpe_ratio']:6.3f}, "
                      f"Cum.Ret: {result['cumulative_return']:7.4f}, "
                      f"MDD: {result['max_drawdown']:7.4f}, "
                      f"WinRate: {result['win_rate']:.2%}")

    return pd.DataFrame(results)


def summarize_trading_results(results_df):
    """Compute average trading metrics across folds."""
    summary = results_df.groupby('model').agg({
        'sharpe_ratio': ['mean', 'std'],
        'cumulative_return': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
        'win_rate': ['mean', 'std'],
        'profit_factor': ['mean', 'std'],
        'ic': ['mean', 'std']
    }).round(4)

    return summary


def compare_trading_performance(original_csv, denoised_csv, n_splits=5):
    """Compare trading performance: original vs denoised datasets."""

    print("=" * 80)
    print("TRADING SIGNAL VALIDATION")
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
    print(f"Signal strategy: Long top 20%, Short bottom 20%")

    # Experiment 1: Original data
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ORIGINAL DATA")
    print("=" * 80)
    results_original = run_trading_cv_experiment(df_original, feature_cols, target_col, n_splits)

    # Experiment 2: Denoised data
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: DENOISED DATA")
    print("=" * 80)
    results_denoised = run_trading_cv_experiment(df_denoised, feature_cols, target_col, n_splits)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: AVERAGE TRADING METRICS ACROSS FOLDS")
    print("=" * 80)

    print("\n[Original Data]:")
    print(summarize_trading_results(results_original))

    print("\n[Denoised Data]:")
    print(summarize_trading_results(results_denoised))

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: DENOISED vs ORIGINAL")
    print("=" * 80)

    summary_orig = results_original.groupby('model')['sharpe_ratio'].mean()
    summary_denoise = results_denoised.groupby('model')['sharpe_ratio'].mean()

    comparison = pd.DataFrame({
        'Original_Sharpe': summary_orig,
        'Denoised_Sharpe': summary_denoise,
        'Improvement': summary_denoise - summary_orig,
        'Improvement_%': ((summary_denoise - summary_orig) / np.abs(summary_orig) * 100).fillna(0)
    }).round(4)

    print("\n[Sharpe Ratio Comparison]:")
    print(comparison)

    # Cumulative Return comparison
    summary_orig_ret = results_original.groupby('model')['cumulative_return'].mean()
    summary_denoise_ret = results_denoised.groupby('model')['cumulative_return'].mean()

    comparison_ret = pd.DataFrame({
        'Original_Return': summary_orig_ret,
        'Denoised_Return': summary_denoise_ret,
        'Improvement': summary_denoise_ret - summary_orig_ret
    }).round(4)

    print("\n[Cumulative Return Comparison]:")
    print(comparison_ret)

    # Save results
    output_dir = Path("TinyRecursiveModels/evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_original.to_csv(output_dir / "original_trading_results.csv", index=False)
    results_denoised.to_csv(output_dir / "denoised_trading_results.csv", index=False)
    comparison.to_csv(output_dir / "trading_comparison.csv")
    comparison_ret.to_csv(output_dir / "trading_returns_comparison.csv")

    print(f"\n[SUCCESS] Results saved to {output_dir}/")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    avg_sharpe_improvement = comparison['Improvement_%'].mean()
    avg_return_improvement = comparison_ret['Improvement'].mean()

    print(f"\nAverage Sharpe Ratio Improvement: {avg_sharpe_improvement:.2f}%")
    print(f"Average Cumulative Return Improvement: {avg_return_improvement:.4f}")

    if avg_sharpe_improvement > 20 and avg_return_improvement > 0:
        print("\n[EFFECTIVE] DENOISING HIGHLY EFFECTIVE FOR TRADING")
        print("   Significant improvement in both risk-adjusted returns and absolute returns")
    elif avg_sharpe_improvement > 0 and avg_return_improvement > 0:
        print("\n[MARGINAL] DENOISING SHOWS POSITIVE IMPACT")
        print("   Modest improvement in trading performance")
    else:
        print("\n[INEFFECTIVE] DENOISING NOT EFFECTIVE FOR TRADING")
        print("   Consider retraining or different denoising parameters")


def main():
    parser = argparse.ArgumentParser(description="Validate denoising with trading signals")
    parser.add_argument("--original", type=str, required=True, help="Path to original CSV")
    parser.add_argument("--denoised", type=str, required=True, help="Path to denoised CSV")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    compare_trading_performance(
        args.original,
        args.denoised,
        n_splits=args.n_splits
    )


if __name__ == "__main__":
    main()
