import numpy as np
import pandas as pd


MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0


class ParticipantVisibleError(Exception):
    pass


def adjusted_sharpe(solution: pd.DataFrame, prediction: np.ndarray) -> float:
    """
    Kaggle metric implementation (volatility- and underperformance-penalized Sharpe).
    Inputs:
      - solution: DataFrame with columns ['forward_returns','risk_free_rate'] at least
      - prediction: np.ndarray of allocations in [0,2]
    Returns:
      - float adjusted sharpe (capped at 1_000_000)
    """
    if not np.issubdtype(prediction.dtype, np.number):
        raise ParticipantVisibleError('Predictions must be numeric')

    df = solution.copy()
    pos = np.asarray(prediction, dtype=np.float64)
    if pos.max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {pos.max()} exceeds maximum of {MAX_INVESTMENT}')
    if pos.min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {pos.min()} below minimum of {MIN_INVESTMENT}')

    df['position'] = pos
    df['strategy_returns'] = df['risk_free_rate'] * (1 - df['position']) + df['position'] * df['forward_returns']

    # Strategy Sharpe
    strategy_excess_returns = df['strategy_returns'] - df['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(df)) - 1
    strategy_std = df['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0 or np.isnan(strategy_std):
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Market
    market_excess_returns = df['forward_returns'] - df['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(df)) - 1
    market_std = df['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)
    if market_volatility == 0 or np.isnan(market_volatility):
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Penalties
    excess_vol = max(0.0, strategy_volatility / market_volatility - 1.2)
    vol_penalty = 1.0 + excess_vol

    return_gap = max(0.0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adjusted = sharpe / (vol_penalty * return_penalty)
    return float(min(adjusted, 1_000_000))


def cumulative_return(weights: np.ndarray, market_excess: np.ndarray) -> float:
    return float(np.sum(weights * market_excess))



