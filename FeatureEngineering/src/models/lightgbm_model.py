"""
LightGBM model for baseline and feature engineering comparison.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict
import yaml


def load_config(config_path: str = "configs/models.yaml") -> dict:
    """Load model configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free_rate: np.ndarray,
    allocations: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics using competition's adjusted Sharpe formula.

    Args:
        y_true: Actual forward returns
        y_pred: Predicted forward returns
        risk_free_rate: Risk-free rate per period
        allocations: Optional allocation weights [0, 2]

    Returns:
        Dictionary of metrics
    """
    from scipy.stats import spearmanr

    # Information Coefficient (Spearman rank correlation)
    ic, ic_pvalue = spearmanr(y_pred, y_true)

    # If allocations provided, calculate trading metrics
    if allocations is not None:
        # Competition formula: strategy_returns = rf * (1 - position) + position * forward_returns
        strategy_returns = risk_free_rate * (1 - allocations) + allocations * y_true

        # Strategy metrics (geometric mean)
        strategy_excess_returns = strategy_returns - risk_free_rate
        strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
        strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(strategy_returns)) - 1
        strategy_std = strategy_returns.std()

        trading_days_per_yr = 252
        if strategy_std == 0 or np.isnan(strategy_std):
            strategy_std = 1e-8

        sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
        strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

        # Market metrics
        market_excess_returns = y_true - risk_free_rate
        market_excess_cumulative = (1 + market_excess_returns).prod()
        market_mean_excess_return = (market_excess_cumulative) ** (1 / len(y_true)) - 1
        market_std = y_true.std()
        market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

        if market_volatility == 0 or np.isnan(market_volatility):
            market_volatility = 1e-8

        # Penalties
        excess_vol = max(0.0, strategy_volatility / market_volatility - 1.2)
        vol_penalty = 1.0 + excess_vol

        return_gap = max(0.0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
        return_penalty = 1.0 + (return_gap ** 2) / 100.0

        adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
        adjusted_sharpe = float(min(adjusted_sharpe, 1_000_000))

        # Max Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        max_dd = drawdown.min()

        # Hit Rate (direction accuracy)
        hit_rate = (np.sign(y_pred) == np.sign(y_true)).mean()

        # Turnover
        turnover = np.abs(np.diff(allocations)).sum() / len(allocations) if len(allocations) > 1 else 0.0

        return {
            'ic': ic,
            'ic_pvalue': ic_pvalue,
            'sharpe': sharpe,  # Raw Sharpe
            'adjusted_sharpe': adjusted_sharpe,  # Competition metric
            'vol_penalty': vol_penalty,
            'return_penalty': return_penalty,
            'max_drawdown': max_dd,
            'hit_rate': hit_rate,
            'turnover': turnover,
            'mean_return': strategy_mean_excess_return,
            'std_return': strategy_std,
            'strategy_volatility': strategy_volatility,
            'market_volatility': market_volatility,
        }
    else:
        # Simple allocation: sign(prediction) + 1 → [0, 2]
        simple_alloc = np.clip(np.sign(y_pred) + 1.0, 0.0, 2.0)
        return calculate_metrics(y_true, y_pred, risk_free_rate, simple_alloc)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    rf_train: pd.Series,
    rf_val: pd.Series,
    config_path: str = "configs/models.yaml"
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """
    Train LightGBM model with early stopping.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        rf_train: Training risk-free rate
        rf_val: Validation risk-free rate
        config_path: Path to model config

    Returns:
        (trained_model, val_metrics)
    """
    config = load_config(config_path)
    lgb_params = config['lightgbm']

    # Extract early stopping parameter
    early_stopping_rounds = lgb_params.pop('early_stopping_rounds', 50)

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train
    print("Training LightGBM...")
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )

    # Predictions
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

    # Metrics
    metrics = calculate_metrics(y_val.values, y_pred_val, rf_val.values)

    print(f"\nValidation Metrics:")
    print(f"  IC: {metrics['ic']:.4f} (p={metrics['ic_pvalue']:.4f})")
    print(f"  Raw Sharpe: {metrics['sharpe']:.4f}")
    print(f"  Adjusted Sharpe: {metrics['adjusted_sharpe']:.4f}")
    print(f"  Vol Penalty: {metrics['vol_penalty']:.4f}")
    print(f"  Return Penalty: {metrics['return_penalty']:.4f}")
    print(f"  Max DD: {metrics['max_drawdown']:.4f}")
    print(f"  Hit Rate: {metrics['hit_rate']:.2%}")

    return model, metrics


def cross_validate_lightgbm(
    df: pd.DataFrame,
    target_col: str = 'forward_returns',
    n_splits: int = 5,
    embargo: int = 21,
    config_path: str = "configs/models.yaml"
) -> Tuple[lgb.Booster, pd.DataFrame]:
    """
    Time series cross-validation with embargo.

    Args:
        df: Input dataframe with features and target
        target_col: Name of target column
        n_splits: Number of CV splits
        embargo: Number of days embargo between train/test
        config_path: Path to model config

    Returns:
        (final_model, results_df)
    """
    # Exclude non-feature columns
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]
    rf = df['risk_free_rate']

    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    print(f"\nTime Series Cross-Validation ({n_splits} splits, embargo={embargo})")
    print("="*60)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")

        # Apply embargo: remove last 'embargo' days from train
        train_idx = train_idx[:-embargo] if len(train_idx) > embargo else train_idx
        # Apply embargo: remove first 'embargo' days from test
        test_idx = test_idx[embargo:] if len(test_idx) > embargo else test_idx

        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  Skipping fold {fold} (insufficient data after embargo)")
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        rf_train, rf_test = rf.iloc[train_idx], rf.iloc[test_idx]

        # Train
        model, metrics = train_lightgbm(
            X_train, y_train, X_test, y_test, rf_train, rf_test, config_path
        )

        # Store results
        metrics['fold'] = fold
        metrics['train_samples'] = len(train_idx)
        metrics['test_samples'] = len(test_idx)
        results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    print(f"Mean IC: {results_df['ic'].mean():.4f} ± {results_df['ic'].std():.4f}")
    print(f"Mean Raw Sharpe: {results_df['sharpe'].mean():.4f} ± {results_df['sharpe'].std():.4f}")
    print(f"Mean Adjusted Sharpe: {results_df['adjusted_sharpe'].mean():.4f} ± {results_df['adjusted_sharpe'].std():.4f}")
    print(f"Mean Vol Penalty: {results_df['vol_penalty'].mean():.4f} ± {results_df['vol_penalty'].std():.4f}")
    print(f"Mean Return Penalty: {results_df['return_penalty'].mean():.4f} ± {results_df['return_penalty'].std():.4f}")
    print(f"Mean Max DD: {results_df['max_drawdown'].mean():.4f} ± {results_df['max_drawdown'].std():.4f}")
    print(f"Mean Hit Rate: {results_df['hit_rate'].mean():.2%} ± {results_df['hit_rate'].std():.2%}")

    # Train final model on all data
    print("\nTraining final model on full dataset...")
    final_model, _ = train_lightgbm(X, y, X, y, rf, rf, config_path)

    return final_model, results_df


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/processed/train_baseline.csv")
    model, results = cross_validate_lightgbm(df)
