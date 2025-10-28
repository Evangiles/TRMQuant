"""
Preprocessing pipeline with leak-safe rolling normalization.

Key principles:
1. Use only past data (no future leakage)
2. Rolling statistics for stationarity
3. Proper handling of missing values
4. Train-only fit for normalization
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from typing import Tuple, List, Optional
import yaml


def load_config(config_path: str = "configs/preprocessing.yaml") -> dict:
    """Load preprocessing configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def handle_missing_values(df: pd.DataFrame, add_nan_flag: bool = True) -> pd.DataFrame:
    """
    Handle missing values with forward/backward fill.

    Args:
        df: Input dataframe
        add_nan_flag: Whether to create _nan indicator columns

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    # Get feature columns (exclude targets and date_id)
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Create NaN flags before filling
    if add_nan_flag:
        for col in feature_cols:
            df[f'{col}_nan'] = df[col].isna().astype(int)

    # Forward fill then backward fill
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Fill any remaining NaNs with column median
    for col in feature_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


def rolling_mad_winsorize(
    df: pd.DataFrame,
    window: int = 252,
    k: float = 4.0,
    min_periods: int = 21
) -> pd.DataFrame:
    """
    Rolling MAD (Median Absolute Deviation) winsorization.

    Clips outliers based on rolling median ± k * MAD.

    Args:
        df: Input dataframe
        window: Rolling window size
        k: MAD multiplier (4 ≈ 4 sigma)
        min_periods: Minimum periods for rolling

    Returns:
        DataFrame with outliers clipped
    """
    df = df.copy()

    # Get feature columns (exclude targets, date_id, and _nan flags)
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [c for c in df.columns
                   if c not in exclude_cols and not c.endswith('_nan')]

    for col in feature_cols:
        # Rolling median
        rolling_median = df[col].rolling(window, min_periods=min_periods).median()

        # Rolling MAD (using expanding window for efficiency)
        def compute_mad(x):
            return median_abs_deviation(x, nan_policy='omit')

        rolling_mad = df[col].rolling(window, min_periods=min_periods).apply(
            compute_mad, raw=False
        )

        # Clip to [median - k*MAD, median + k*MAD]
        lower = rolling_median - k * rolling_mad
        upper = rolling_median + k * rolling_mad
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def rolling_zscore_normalize(
    df: pd.DataFrame,
    window: int = 252,
    min_periods: int = 21,
    epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Rolling z-score normalization: (x - rolling_mean) / rolling_std.

    Args:
        df: Input dataframe
        window: Rolling window size
        min_periods: Minimum periods for rolling
        epsilon: Small constant to avoid division by zero

    Returns:
        DataFrame with normalized features
    """
    df = df.copy()

    # Get feature columns (exclude targets, date_id, and _nan flags)
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [c for c in df.columns
                   if c not in exclude_cols and not c.endswith('_nan')]

    for col in feature_cols:
        rolling_mean = df[col].rolling(window, min_periods=min_periods).mean()
        rolling_std = df[col].rolling(window, min_periods=min_periods).std()

        df[f'{col}_z'] = (df[col] - rolling_mean) / (rolling_std + epsilon)

    return df


def add_differencing(
    df: pd.DataFrame,
    level_prefixes: List[str] = ['E', 'I', 'P']
) -> pd.DataFrame:
    """
    Add differencing for level-type features (Economic, Interest, Price).

    Args:
        df: Input dataframe
        level_prefixes: Feature family prefixes to difference

    Returns:
        DataFrame with added difference features
    """
    df = df.copy()

    # Get feature columns matching prefixes
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']

    for prefix in level_prefixes:
        cols = [c for c in df.columns
               if c.startswith(prefix) and c not in exclude_cols and not c.endswith('_nan')]

        for col in cols:
            # Simple difference
            df[f'{col}_diff'] = df[col].diff()

            # Log difference (for positive values)
            if (df[col] > 0).all():
                df[f'{col}_logdiff'] = np.log(df[col]).diff()

    return df


def split_train_val(
    df: pd.DataFrame,
    train_fraction: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/val split (no shuffle for time series).

    Args:
        df: Input dataframe (must be sorted by date_id)
        train_fraction: Fraction of data for training

    Returns:
        (train_df, val_df)
    """
    # Ensure sorted by date_id
    df = df.sort_values('date_id').reset_index(drop=True)

    split_idx = int(len(df) * train_fraction)

    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    return train_df, val_df


def preprocess_pipeline(
    input_path: str,
    output_path: str,
    config_path: str = "configs/preprocessing.yaml"
) -> Tuple[pd.DataFrame, dict]:
    """
    Complete preprocessing pipeline.

    Args:
        input_path: Path to raw CSV
        output_path: Path to save processed CSV
        config_path: Path to preprocessing config

    Returns:
        (processed_df, metadata)
    """
    # Load config
    config = load_config(config_path)

    print("Loading data...")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    # Ensure sorted by date_id
    df = df.sort_values('date_id').reset_index(drop=True)

    # 1. Handle missing values
    print("\n1. Handling missing values...")
    df = handle_missing_values(df, add_nan_flag=config['missing']['add_nan_flag'])

    # 2. Outlier handling (Rolling MAD winsorization)
    print("2. Rolling MAD winsorization...")
    df = rolling_mad_winsorize(
        df,
        window=config['outlier']['window'],
        k=config['outlier']['k'],
        min_periods=config['outlier']['min_periods']
    )

    # 3. Rolling z-score normalization
    print("3. Rolling z-score normalization...")
    df = rolling_zscore_normalize(
        df,
        window=config['normalization']['window'],
        min_periods=config['normalization']['min_periods'],
        epsilon=config['normalization']['epsilon']
    )

    # 4. Add differencing for level features
    print("4. Adding differencing for level features...")
    df = add_differencing(df, level_prefixes=config['differencing']['level_prefixes'])

    # 5. Train/val split
    print("5. Splitting train/val...")
    train_df, val_df = split_train_val(df, train_fraction=config['split']['train_fraction'])

    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")

    # Save processed data
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)

    # Metadata
    metadata = {
        'n_samples': len(df),
        'n_features': len([c for c in df.columns if c not in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']]),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'split_date_id': int(train_df['date_id'].iloc[-1]),
    }

    print(f"\nPreprocessing complete!")
    print(f"  Total features: {metadata['n_features']}")
    print(f"  Train samples: {metadata['train_samples']}")
    print(f"  Val samples: {metadata['val_samples']}")

    return df, metadata


if __name__ == "__main__":
    # Example usage
    preprocess_pipeline(
        input_path="data/raw/train.csv",
        output_path="data/processed/train_baseline.csv"
    )
