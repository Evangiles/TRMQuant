"""
Phase 1: 12 Core Features (Strict leak-safe specification)

Using only given column families: M, E, I*, P*, V*, S*, MOM*, D*
No high/low assumptions. No future leakage.

12 features (fixed names):
1. ret_21, 2. ret_63, 3. RV_21, 4. RV_63, 5. vol_regime
6. z_price_63, 7. dist_high_63, 8. dist_low_63
9. mom_short, 10. vol_level, 11. econ_change, 12. term_spread
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import warnings

EPS = 1e-8
SQRT_252 = np.sqrt(252)


def identify_anchor_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Identify anchor columns for features.

    Returns:
        (P_anchor, M_anchor) - representative price and market columns
    """
    # P* columns (price-like, level data)
    P_cols = [c for c in df.columns if c.startswith('P') and not c.endswith('_nan')]

    # M* columns (market data)
    M_cols = [c for c in df.columns if c.startswith('M') and not c.endswith('_nan')]

    # Select anchor with most valid data
    P_anchor = None
    if P_cols:
        valid_counts = {c: df[c].notna().sum() for c in P_cols}
        P_anchor = max(valid_counts, key=valid_counts.get)

    M_anchor = None
    if M_cols:
        valid_counts = {c: df[c].notna().sum() for c in M_cols}
        M_anchor = max(valid_counts, key=valid_counts.get)

    # Fallback: use M_anchor if P_anchor not available
    if P_anchor is None and M_anchor is not None:
        P_anchor = M_anchor
        warnings.warn(f"No P* columns found, using M_anchor={M_anchor} for price features")

    return P_anchor, M_anchor


def identify_rate_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Identify long and short maturity rate columns.

    Returns:
        (I_long, I_short) or (None, None) if not found
    """
    I_cols = [c for c in df.columns if c.startswith('I') and not c.endswith('_nan')]

    if not I_cols:
        return None, None

    # Whitelist patterns (case-insensitive)
    long_patterns = ['10Y', '10y', '30Y', '30y']
    short_patterns = ['3M', '3m', '6M', '6m']

    I_long = None
    I_short = None

    # Try whitelist first
    for pattern in long_patterns:
        matches = [c for c in I_cols if pattern in c]
        if matches:
            I_long = matches[0]
            break

    for pattern in short_patterns:
        matches = [c for c in I_cols if pattern in c]
        if matches:
            I_short = matches[0]
            break

    # Fallback: use first and last I* columns
    if I_long is None and len(I_cols) >= 2:
        I_long = I_cols[-1]  # Assume last is longest maturity
    if I_short is None and len(I_cols) >= 2:
        I_short = I_cols[0]  # Assume first is shortest maturity

    return I_long, I_short


def get_family_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Get all columns for a given family prefix."""
    return [c for c in df.columns
            if c.startswith(prefix) and not c.endswith('_nan')]


def add_12_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add exactly 12 core features following strict specification.

    Args:
        df: Input dataframe (must be sorted by date_id)

    Returns:
        DataFrame with 12 core features added
    """
    df = df.copy()

    # Ensure sorted
    df = df.sort_values('date_id').reset_index(drop=True)

    # Identify anchors
    P_anchor, M_anchor = identify_anchor_columns(df)
    I_long, I_short = identify_rate_columns(df)

    print("=== Column Selection ===")
    print(f"P_anchor: {P_anchor}")
    print(f"M_anchor: {M_anchor}")
    print(f"I_long: {I_long}, I_short: {I_short}")

    if P_anchor is None:
        raise ValueError("Could not identify P_anchor or M_anchor")

    # Get family columns
    MOM_cols = get_family_columns(df, 'MOM')
    V_cols = get_family_columns(df, 'V')
    E_cols = get_family_columns(df, 'E')

    print(f"MOM_cols: {len(MOM_cols)}")
    print(f"V_cols: {len(V_cols)}")
    print(f"E_cols: {len(E_cols)}")

    # Convert to float
    df[P_anchor] = df[P_anchor].astype(float)

    # ========== Features 1-2: Momentum ==========
    print("\n=== Computing Features ===")
    print("1-2. ret_21, ret_63")
    df['ret_21'] = df[P_anchor].pct_change(21)
    df['ret_63'] = df[P_anchor].pct_change(63)

    # ========== Features 3-5: Realized Volatility ==========
    print("3-5. RV_21, RV_63, vol_regime")
    returns = df[P_anchor].pct_change()

    df['RV_21'] = returns.rolling(21, min_periods=21).std() * SQRT_252
    df['RV_63'] = returns.rolling(63, min_periods=63).std() * SQRT_252
    df['vol_regime'] = df['RV_21'] / (df['RV_63'] + EPS)

    # ========== Features 6-8: Mean-reversion ==========
    print("6-8. z_price_63, dist_high_63, dist_low_63")
    rolling_mean = df[P_anchor].rolling(63, min_periods=63).mean()
    rolling_std = df[P_anchor].rolling(63, min_periods=63).std()
    df['z_price_63'] = (df[P_anchor] - rolling_mean) / (rolling_std + EPS)

    rolling_max = df[P_anchor].rolling(63, min_periods=63).max()
    rolling_min = df[P_anchor].rolling(63, min_periods=63).min()
    df['dist_high_63'] = (df[P_anchor] - rolling_max) / (rolling_max + EPS)
    df['dist_low_63'] = (df[P_anchor] - rolling_min) / (rolling_min + EPS)

    # ========== Feature 9: mom_short ==========
    print("9. mom_short")
    if MOM_cols:
        # Z-score each MOM column with 63-day window
        mom_zscores = []
        for col in MOM_cols:
            df[col] = df[col].astype(float)
            roll_mean = df[col].rolling(63, min_periods=63).mean()
            roll_std = df[col].rolling(63, min_periods=63).std()
            z = (df[col] - roll_mean) / (roll_std + EPS)
            mom_zscores.append(z)

        # Cross-sectional mean at each date
        mom_zscores_df = pd.concat(mom_zscores, axis=1)
        df['mom_short'] = mom_zscores_df.mean(axis=1)
    else:
        print("  Warning: No MOM* columns found, setting mom_short=0")
        df['mom_short'] = 0.0

    # ========== Feature 10: vol_level ==========
    print("10. vol_level")
    if V_cols:
        # Z-score each V column with 63-day window
        vol_zscores = []
        for col in V_cols:
            df[col] = df[col].astype(float)
            roll_mean = df[col].rolling(63, min_periods=63).mean()
            roll_std = df[col].rolling(63, min_periods=63).std()
            z = (df[col] - roll_mean) / (roll_std + EPS)
            vol_zscores.append(z)

        # Cross-sectional mean
        vol_zscores_df = pd.concat(vol_zscores, axis=1)
        df['vol_level'] = vol_zscores_df.mean(axis=1)
    else:
        print("  Warning: No V* columns found, setting vol_level=0")
        df['vol_level'] = 0.0

    # ========== Feature 11: econ_change ==========
    print("11. econ_change")
    if E_cols:
        # Diff(1) then 21-day moving average for each E column
        econ_changes = []
        for col in E_cols:
            df[col] = df[col].astype(float)
            diff1 = df[col].diff(1)
            ma21 = diff1.rolling(21, min_periods=21).mean()
            econ_changes.append(ma21)

        # Cross-sectional mean
        econ_changes_df = pd.concat(econ_changes, axis=1)
        df['econ_change'] = econ_changes_df.mean(axis=1)
    else:
        print("  Warning: No E* columns found, setting econ_change=0")
        df['econ_change'] = 0.0

    # ========== Feature 12: term_spread ==========
    print("12. term_spread")
    if I_long and I_short:
        df[I_long] = df[I_long].astype(float)
        df[I_short] = df[I_short].astype(float)
        df['term_spread'] = df[I_long] - df[I_short]
    else:
        print("  Warning: Could not identify I_long/I_short, setting term_spread=0")
        df['term_spread'] = 0.0

    # ========== Validation ==========
    core_features = [
        'ret_21', 'ret_63', 'RV_21', 'RV_63', 'vol_regime',
        'z_price_63', 'dist_high_63', 'dist_low_63',
        'mom_short', 'vol_level', 'econ_change', 'term_spread'
    ]

    missing = [f for f in core_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Report NaN statistics
    print("\n=== NaN Statistics ===")
    for feat in core_features:
        nan_count = df[feat].isna().sum()
        nan_pct = nan_count / len(df) * 100
        print(f"{feat:20s}: {nan_count:5d} NaNs ({nan_pct:5.1f}%)")

    print(f"\n=== Success: 12 core features added ===")

    return df


def handle_nan_with_flags(df: pd.DataFrame, core_features: List[str]) -> pd.DataFrame:
    """
    Handle NaN values: zero-fill + add _nan flags.

    Args:
        df: DataFrame with core features
        core_features: List of 12 core feature names

    Returns:
        DataFrame with NaN handled
    """
    df = df.copy()

    for feat in core_features:
        # Add _nan flag
        df[f'{feat}_nan'] = df[feat].isna().astype(int)

        # Zero-fill
        df[feat] = df[feat].fillna(0.0)

    return df


def get_core_feature_names() -> List[str]:
    """Return list of 12 core feature names."""
    return [
        'ret_21', 'ret_63', 'RV_21', 'RV_63', 'vol_regime',
        'z_price_63', 'dist_high_63', 'dist_low_63',
        'mom_short', 'vol_level', 'econ_change', 'term_spread'
    ]


if __name__ == "__main__":
    # Test on processed data
    print("=== Testing on RAW data ===")
    df = pd.read_csv("data/processed/train_baseline_raw.csv")
    print(f"Input: {df.shape}")

    df_with_features = add_12_core_features(df)

    # Handle NaN
    core_features = get_core_feature_names()
    df_with_features = handle_nan_with_flags(df_with_features, core_features)

    print(f"\nOutput: {df_with_features.shape}")

    # Count total features
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    n_features = len([c for c in df_with_features.columns if c not in exclude_cols])
    print(f"Total features: {n_features}")

    # Save
    df_with_features.to_csv("data/features/train_phase1_raw.csv", index=False)
    print("\nSaved to data/features/train_phase1_raw.csv")

    # Test on DENOISED data
    print("\n" + "="*60)
    print("=== Testing on DENOISED data ===")
    df = pd.read_csv("data/processed/train_baseline_denoised.csv")
    print(f"Input: {df.shape}")

    df_with_features = add_12_core_features(df)
    df_with_features = handle_nan_with_flags(df_with_features, core_features)

    print(f"\nOutput: {df_with_features.shape}")

    df_with_features.to_csv("data/features/train_phase1_denoised.csv", index=False)
    print("\nSaved to data/features/train_phase1_denoised.csv")
