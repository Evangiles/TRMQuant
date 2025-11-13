"""
Validate Denoising with Trading Signal Performance (80/20 Holdout)

Compares original vs denoised data using actual trading signals and portfolio metrics:
- Adjusted Sharpe Ratio (Competition Metric)
- Sharpe Ratio
- Cumulative Returns
- Maximum Drawdown (MDD)
- Win Rate
- Volatility Penalty
- Return Penalty

Uses competition's position-based strategy (0-2 range) with:
- Geometric mean for annualized returns
- Volatility penalty when exceeding 1.2x market volatility
- Return penalty for underperforming market
- Adjusted Sharpe = Sharpe / (vol_penalty * return_penalty)

Uses simple 80/20 holdout validation (train_only.csv → val_only.csv).
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


def generate_trading_signals(predictions, returns, risk_free_rates, train_min=None, train_max=None):
    """
    Generate position-based strategy following competition metric.

    Competition formula:
    - position = scaled prediction (0 to 2 range)
    - strategy_returns = rf * (1 - position) + position * forward_returns
    - Sharpe with geometric mean and volatility/return penalties

    Args:
        predictions: Model predicted returns
        returns: Actual forward returns
        risk_free_rates: Risk-free rates
        train_min: Min value from training predictions (for leak-free scaling)
        train_max: Max value from training predictions (for leak-free scaling)

    Returns:
        dict with competition metrics
    """
    MIN_INVESTMENT = 0
    MAX_INVESTMENT = 2
    TRADING_DAYS_PER_YR = 252

    # Convert predictions to positions (0~2 range)
    # Use train set statistics to prevent test set leakage
    if train_min is not None and train_max is not None:
        pred_min = train_min
        pred_max = train_max
    else:
        # Fallback: use test set statistics (NOT RECOMMENDED for production)
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)

    if pred_max - pred_min > 0:
        positions = (predictions - pred_min) / (pred_max - pred_min) * MAX_INVESTMENT
    else:
        positions = np.ones_like(predictions)  # Default to 100% invested

    # Clip to valid range
    positions = np.clip(positions, MIN_INVESTMENT, MAX_INVESTMENT)

    # Calculate strategy returns (competition formula)
    strategy_returns = risk_free_rates * (1 - positions) + positions * returns

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = strategy_returns - risk_free_rates

    # Geometric mean (competition uses this!)
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    if strategy_excess_cumulative <= 0:
        return None

    strategy_mean_excess_return = strategy_excess_cumulative ** (1 / len(strategy_returns)) - 1
    strategy_std = strategy_returns.std()

    if strategy_std == 0:
        return None

    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(TRADING_DAYS_PER_YR)
    strategy_volatility = strategy_std * np.sqrt(TRADING_DAYS_PER_YR) * 100

    # Calculate market return and volatility
    market_excess_returns = returns - risk_free_rates
    market_excess_cumulative = (1 + market_excess_returns).prod()

    if market_excess_cumulative <= 0:
        return None

    market_mean_excess_return = market_excess_cumulative ** (1 / len(returns)) - 1
    market_std = returns.std()
    market_volatility = market_std * np.sqrt(TRADING_DAYS_PER_YR) * 100

    # Calculate volatility penalty (competition formula)
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate return penalty (competition formula)
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * TRADING_DAYS_PER_YR)
    return_penalty = 1 + (return_gap ** 2) / 100

    # Adjusted Sharpe ratio (competition metric!)
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)

    # Additional metrics for analysis
    cumulative_return = (1 + strategy_returns).prod() - 1

    # Maximum Drawdown
    cumsum = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    max_drawdown = np.min(drawdown)

    # Win rate
    wins = strategy_returns > risk_free_rates
    win_rate = np.mean(wins)

    return {
        'adjusted_sharpe': min(adjusted_sharpe, 1_000_000),  # Cap at 1M like competition
        'sharpe_ratio': sharpe,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'mean_return': strategy_mean_excess_return,
        'volatility': strategy_volatility,
        'vol_penalty': vol_penalty,
        'return_penalty': return_penalty,
        'n_trades': len(strategy_returns)
    }


def evaluate_trading_model(model_name, model, X_train, y_train, X_test, y_test, rf_test):
    """Train model and evaluate trading performance (LEAK-FREE)."""
    # Train
    model.fit(X_train, y_train)

    # Predict on BOTH train and test (train for scaling statistics)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # CRITICAL: Use TRAIN predictions for min/max scaling (prevents test set leakage)
    train_min = np.min(y_pred_train)
    train_max = np.max(y_pred_train)

    # Generate trading signals and calculate metrics
    trading_metrics = generate_trading_signals(y_pred_test, y_test, rf_test,
                                                train_min=train_min, train_max=train_max)

    if trading_metrics is None:
        return None

    # Add model info
    trading_metrics['model'] = model_name

    # Also include IC for reference
    ic = np.corrcoef(y_test, y_pred_test)[0, 1]
    trading_metrics['ic'] = ic

    return trading_metrics


def run_holdout_evaluation(df_train, df_val, feature_cols, target_col):
    """Run simple 80/20 holdout evaluation."""

    # Prepare train data
    X_train = df_train[feature_cols].fillna(0).values
    y_train = df_train[target_col].fillna(0).values

    # Prepare val data
    X_val = df_val[feature_cols].fillna(0).values
    y_val = df_val[target_col].fillna(0).values
    rf_val = df_val['risk_free_rate'].fillna(0).values

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

    print(f"\nRunning Holdout Evaluation (Train: {len(X_train)}, Val: {len(X_val)})...")
    print("=" * 80)

    # Evaluate each model
    for model_name, model in models.items():
        result = evaluate_trading_model(model_name, model, X_train, y_train, X_val, y_val, rf_val)

        if result is not None:
            results.append(result)

            print(f"  {model_name:15s} - AdjSharpe: {result['adjusted_sharpe']:6.3f}, "
                  f"Sharpe: {result['sharpe_ratio']:6.3f}, "
                  f"Cum.Ret: {result['cumulative_return']:7.4f}, "
                  f"MDD: {result['max_drawdown']:7.4f}")

    return pd.DataFrame(results)


def summarize_trading_results(results_df):
    """Compute trading metrics summary."""
    summary = results_df[['model', 'adjusted_sharpe', 'sharpe_ratio', 'cumulative_return',
                           'max_drawdown', 'win_rate', 'vol_penalty', 'return_penalty', 'ic']].set_index('model')
    return summary


def compare_trading_performance(train_original_csv, train_denoised_csv, val_original_csv, val_denoised_csv):
    """Compare trading performance: original vs denoised datasets (80/20 holdout)."""

    print("=" * 80)
    print("TRADING SIGNAL VALIDATION (80/20 HOLDOUT)")
    print("=" * 80)

    # Load data
    print("\nLoading datasets...")
    df_train_original = pd.read_csv(train_original_csv)
    df_train_denoised = pd.read_csv(train_denoised_csv)
    df_val_original = pd.read_csv(val_original_csv)
    df_val_denoised = pd.read_csv(val_denoised_csv)

    print(f"Train Original: {len(df_train_original)} rows")
    print(f"Train Denoised: {len(df_train_denoised)} rows")
    print(f"Val Original: {len(df_val_original)} rows")
    print(f"Val Denoised: {len(df_val_denoised)} rows")

    # Get columns
    feature_cols, target_col = get_feature_target_columns(df_train_original)
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")
    print(f"Strategy: Position-based (0-2 range) with risk-free rate consideration")
    print(f"Metric: Adjusted Sharpe with geometric mean + penalties")

    # Experiment 1: Original data
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ORIGINAL DATA")
    print("=" * 80)
    results_original = run_holdout_evaluation(df_train_original, df_val_original, feature_cols, target_col)

    # Experiment 2: Denoised data
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: DENOISED DATA")
    print("=" * 80)
    results_denoised = run_holdout_evaluation(df_train_denoised, df_val_denoised, feature_cols, target_col)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: TRADING METRICS")
    print("=" * 80)

    print("\n[Original Data]:")
    print(summarize_trading_results(results_original))

    print("\n[Denoised Data]:")
    print(summarize_trading_results(results_denoised))

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: DENOISED vs ORIGINAL")
    print("=" * 80)

    # Adjusted Sharpe comparison (PRIMARY METRIC)
    summary_orig_adj = results_original.set_index('model')['adjusted_sharpe']
    summary_denoise_adj = results_denoised.set_index('model')['adjusted_sharpe']

    comparison_adj = pd.DataFrame({
        'Original_AdjSharpe': summary_orig_adj,
        'Denoised_AdjSharpe': summary_denoise_adj,
        'Improvement': summary_denoise_adj - summary_orig_adj,
        'Improvement_%': ((summary_denoise_adj - summary_orig_adj) / np.abs(summary_orig_adj) * 100).fillna(0)
    }).round(4)

    print("\n[Adjusted Sharpe Ratio Comparison (COMPETITION METRIC)]:")
    print(comparison_adj)

    # Regular Sharpe comparison (for reference)
    summary_orig = results_original.set_index('model')['sharpe_ratio']
    summary_denoise = results_denoised.set_index('model')['sharpe_ratio']

    comparison = pd.DataFrame({
        'Original_Sharpe': summary_orig,
        'Denoised_Sharpe': summary_denoise,
        'Improvement': summary_denoise - summary_orig,
        'Improvement_%': ((summary_denoise - summary_orig) / np.abs(summary_orig) * 100).fillna(0)
    }).round(4)

    print("\n[Sharpe Ratio Comparison (Reference)]:")
    print(comparison)

    # Cumulative Return comparison
    summary_orig_ret = results_original.set_index('model')['cumulative_return']
    summary_denoise_ret = results_denoised.set_index('model')['cumulative_return']

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
    comparison_adj.to_csv(output_dir / "trading_adjusted_sharpe_comparison.csv")
    comparison.to_csv(output_dir / "trading_sharpe_comparison.csv")
    comparison_ret.to_csv(output_dir / "trading_returns_comparison.csv")

    print(f"\n[SUCCESS] Results saved to {output_dir}/")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    avg_adj_sharpe_improvement = comparison_adj['Improvement_%'].mean()
    avg_sharpe_improvement = comparison['Improvement_%'].mean()
    avg_return_improvement = comparison_ret['Improvement'].mean()

    print(f"\nAverage Adjusted Sharpe Ratio Improvement (COMPETITION): {avg_adj_sharpe_improvement:.2f}%")
    print(f"Average Sharpe Ratio Improvement (Reference): {avg_sharpe_improvement:.2f}%")
    print(f"Average Cumulative Return Improvement: {avg_return_improvement:.4f}")

    if avg_adj_sharpe_improvement > 20 and avg_return_improvement > 0:
        print("\n[EFFECTIVE] DENOISING HIGHLY EFFECTIVE FOR TRADING")
        print("   Significant improvement in both risk-adjusted returns and absolute returns")
    elif avg_adj_sharpe_improvement > 0 and avg_return_improvement > 0:
        print("\n[MARGINAL] DENOISING SHOWS POSITIVE IMPACT")
        print("   Modest improvement in trading performance")
    else:
        print("\n[INEFFECTIVE] DENOISING NOT EFFECTIVE FOR TRADING")
        print("   Consider retraining or different denoising parameters")


def main():
    parser = argparse.ArgumentParser(description="Validate denoising with trading signals (80/20 holdout)")
    parser.add_argument("--train_original", type=str, required=True, help="Path to train original CSV")
    parser.add_argument("--train_denoised", type=str, required=True, help="Path to train denoised CSV")
    parser.add_argument("--val_original", type=str, required=True, help="Path to val original CSV")
    parser.add_argument("--val_denoised", type=str, required=True, help="Path to val denoised CSV")

    args = parser.parse_args()

    compare_trading_performance(
        args.train_original,
        args.train_denoised,
        args.val_original,
        args.val_denoised
    )


if __name__ == "__main__":
    main()
