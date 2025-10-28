"""
Phase 2 & 3: Advanced Feature Engineering

Phase 2: Family-wise PCA (dimension reduction)
Phase 3: Stability filtering with RankIC
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from typing import List, Tuple, Dict
import warnings


def get_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group features by family prefix.

    Returns:
        Dict mapping family name to list of column names
    """
    families = {}
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']

    # Define family prefixes
    prefixes = ['M', 'E', 'I', 'P', 'V', 'S', 'MOM', 'D']

    for prefix in prefixes:
        cols = [c for c in df.columns
                if c.startswith(prefix) and c not in exclude_cols
                and not c.endswith('_nan')]
        if cols:
            families[prefix] = cols

    # Add Phase 1 core features as separate family
    core_features = [
        'ret_21', 'ret_63', 'RV_21', 'RV_63', 'vol_regime',
        'z_price_63', 'dist_high_63', 'dist_low_63',
        'mom_short', 'vol_level', 'econ_change', 'term_spread'
    ]
    core_exists = [f for f in core_features if f in df.columns]
    if core_exists:
        families['CORE'] = core_exists

    return families


def apply_family_pca(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    variance_threshold: float = 0.95,
    min_components: int = 1,
    max_components: int = 10,
    exclude_core: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply PCA to each feature family separately (leak-safe with standardization).

    Args:
        df: Input dataframe (must be sorted by date_id)
        train_fraction: Fraction for train split
        variance_threshold: Cumulative variance to preserve (default 95%)
        min_components: Minimum components per family
        max_components: Maximum components per family
        exclude_core: Whether to exclude CORE features from PCA (default True)

    Returns:
        (transformed_df, pca_info with saved models)
    """
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    df = df.sort_values('date_id').reset_index(drop=True)

    # Split index
    split_idx = int(len(df) * train_fraction)

    # Get feature families
    families = get_feature_families(df)

    # Extract CORE features (don't PCA them)
    core_features = None
    if exclude_core and 'CORE' in families:
        core_features = families.pop('CORE')
        print(f"CORE features excluded from PCA: {len(core_features)}")

    print(f"\n=== Phase 2: Family-wise PCA (with standardization) ===")
    print(f"Train samples: {split_idx}, Total: {len(df)}")
    print(f"Variance threshold: {variance_threshold}")

    pca_info = {'models': {}, 'meta': {}}
    pca_features = []

    for family_name, family_cols in families.items():
        print(f"\nFamily: {family_name} ({len(family_cols)} features)")

        # Skip if too few features
        if len(family_cols) < 2:
            print(f"  Skipping (< 2 features)")
            # Keep original features
            for col in family_cols:
                pca_features.append(df[col])
            continue

        # Extract family features
        X = df[family_cols].values.astype(float)

        # Handle NaN: fill with 0 for PCA input only
        X_filled = np.nan_to_num(X, nan=0.0)

        # 1. Standardize on TRAIN data only
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_filled[:split_idx])

        # 2. Fit PCA on standardized train data
        n_components = min(len(family_cols), max_components)

        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_train_scaled)

        # Find number of components for variance threshold
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_keep = max(min_components, np.searchsorted(cumsum_var, variance_threshold) + 1)
        n_keep = min(n_keep, n_components)

        # Re-fit with selected components
        pca_final = PCA(n_components=n_keep, random_state=42)
        pca_final.fit(X_train_scaled)

        # 3. Transform full dataset (standardize then PCA)
        X_scaled = scaler.transform(X_filled)
        X_pca = pca_final.transform(X_scaled)

        # Create PCA feature names
        for i in range(n_keep):
            pca_col = pd.Series(X_pca[:, i], index=df.index)
            pca_col.name = f'PCA_{family_name}_{i+1}'
            pca_features.append(pca_col)

        # Store models for inference
        pca_info['models'][family_name] = {
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'pca_components': pca_final.components_,
            'pca_mean': pca_final.mean_,
            'pca_explained_variance_ratio': pca_final.explained_variance_ratio_
        }

        # Store metadata
        pca_info['meta'][family_name] = {
            'n_original': len(family_cols),
            'n_components': n_keep,
            'variance_explained': float(cumsum_var[n_keep-1]),
            'reduction': f"{len(family_cols)} → {n_keep} ({len(family_cols)-n_keep} removed)",
            'original_features': family_cols
        }

        print(f"  {pca_info['meta'][family_name]['reduction']}")
        print(f"  Variance explained: {pca_info['meta'][family_name]['variance_explained']:.2%}")

    # Add CORE features back (no PCA)
    if core_features:
        print(f"\nAdding CORE features (no PCA): {len(core_features)}")
        for col in core_features:
            pca_features.append(df[col])

    # Create new dataframe with PCA features
    pca_df = pd.concat(pca_features, axis=1)

    # Add metadata columns
    meta_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    for col in meta_cols:
        if col in df.columns:
            pca_df.insert(0, col, df[col])

    print(f"\n=== PCA Summary ===")
    total_original = sum(info['n_original'] for info in pca_info['meta'].values())
    total_components = sum(info['n_components'] for info in pca_info['meta'].values())
    if core_features:
        total_components += len(core_features)
    print(f"Total: {total_original} → {total_components} features ({total_original - total_components} removed)")
    if core_features:
        print(f"  + {len(core_features)} CORE features preserved")

    return pca_df, pca_info


def calculate_rolling_spearman_vectorized(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'forward_returns',
    window: int = 252,
    min_periods: int = 126
) -> pd.DataFrame:
    """
    Calculate rolling Spearman correlation (vectorized for speed).

    Note: This is temporal correlation, not cross-sectional RankIC.
    Spearman = Pearson(ranks), so we rank then use rolling corr.

    Args:
        df: Input dataframe
        feature_cols: List of feature column names
        target_col: Target column name
        window: Rolling window size
        min_periods: Minimum periods for rolling

    Returns:
        DataFrame with rolling Spearman correlation for each feature
    """
    print(f"\n=== Calculating Rolling Spearman (vectorized) ===")
    print(f"Window: {window}, Min periods: {min_periods}")
    print(f"Features: {len(feature_cols)}")

    # Select features + target
    cols = feature_cols + [target_col]
    data = df[cols].copy()

    # Rank all columns (method='average' for ties)
    print("  Ranking features...")
    ranked_data = data.rank(method='average')

    # Calculate rolling Pearson on ranked data = Spearman
    print("  Computing rolling correlations...")

    # Use pairwise=False to get correlation for each feature vs target
    spearman_corrs = []

    for i, col in enumerate(feature_cols):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(feature_cols)}")

        # Rolling correlation between ranked feature and ranked target
        corr = ranked_data[[col, target_col]].rolling(
            window=window,
            min_periods=min_periods
        ).corr().unstack()[target_col][col]

        corr.name = col
        spearman_corrs.append(corr)

    spearman_df = pd.concat(spearman_corrs, axis=1)
    print(f"  Complete!")

    return spearman_df


def apply_stability_filter(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    ic_threshold: float = 0.005,
    std_threshold: float = 0.2,
    window: int = 252,
    min_periods: int = 126
) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter features based on stability of temporal Spearman correlation (leak-safe).

    Note: Uses temporal correlation, not cross-sectional RankIC.
    Filters for features with consistent predictive relationship to target.

    Args:
        df: Input dataframe (must be sorted by date_id)
        train_fraction: Fraction for train split
        ic_threshold: Minimum mean abs(Spearman) to keep (default 0.005)
        std_threshold: Maximum std(Spearman) to keep (default 0.2)
        window: Rolling window for correlation calculation
        min_periods: Minimum periods for rolling

    Returns:
        (filtered_df, filter_info)
    """
    df = df.copy()
    df = df.sort_values('date_id').reset_index(drop=True)

    # Split index
    split_idx = int(len(df) * train_fraction)

    # Get feature columns
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\n=== Phase 3: Stability Filter (temporal Spearman) ===")
    print(f"Features before filtering: {len(feature_cols)}")
    print(f"IC threshold: {ic_threshold}, Std threshold: {std_threshold}")
    print(f"Note: Adjustable thresholds - consider grid search on validation set")

    # Calculate rolling Spearman correlation (vectorized)
    spearman_df = calculate_rolling_spearman_vectorized(
        df,
        feature_cols,
        target_col='forward_returns',
        window=window,
        min_periods=min_periods
    )

    # Calculate stability metrics on TRAIN data only
    train_spearman = spearman_df.iloc[:split_idx]

    feature_stats = []
    for col in feature_cols:
        corr_values = train_spearman[col].dropna()

        if len(corr_values) < 10:
            # Not enough data
            mean_corr = 0.0
            std_corr = 999.0
        else:
            mean_corr = corr_values.mean()
            std_corr = corr_values.std()

        feature_stats.append({
            'feature': col,
            'mean_abs_corr': abs(mean_corr),
            'std_corr': std_corr,
            'mean_corr': mean_corr
        })

    stats_df = pd.DataFrame(feature_stats)

    # Filter based on thresholds
    keep_mask = (stats_df['mean_abs_corr'] >= ic_threshold) & (stats_df['std_corr'] <= std_threshold)
    keep_features = stats_df[keep_mask]['feature'].tolist()

    print(f"\n=== Filter Results ===")
    print(f"Features kept: {len(keep_features)}/{len(feature_cols)}")
    print(f"Features removed: {len(feature_cols) - len(keep_features)}")

    # Create filtered dataframe
    filtered_df = df[exclude_cols + keep_features].copy()

    # Filter info
    filter_info = {
        'n_original': len(feature_cols),
        'n_kept': len(keep_features),
        'n_removed': len(feature_cols) - len(keep_features),
        'ic_threshold': ic_threshold,
        'std_threshold': std_threshold,
        'kept_features': keep_features,
        'removed_features': [c for c in feature_cols if c not in keep_features],
        'feature_stats': stats_df  # Include for grid search
    }

    # Show top kept features by correlation
    top_kept = stats_df[keep_mask].nlargest(10, 'mean_abs_corr')
    print(f"\nTop 10 kept features (by mean abs Spearman):")
    for _, row in top_kept.iterrows():
        print(f"  {row['feature']:30s}: Corr={row['mean_corr']:+.4f}, Std={row['std_corr']:.4f}")

    # Show some removed features for comparison
    if len(feature_cols) > len(keep_features):
        removed = stats_df[~keep_mask].head(5)
        print(f"\nSample removed features:")
        for _, row in removed.iterrows():
            print(f"  {row['feature']:30s}: Corr={row['mean_corr']:+.4f}, Std={row['std_corr']:.4f}")

    return filtered_df, filter_info


if __name__ == "__main__":
    # Test on Phase 1 denoised data
    print("=== Testing Phase 2 & 3 on denoised+denoised_FE ===")
    df = pd.read_csv("data/features/train_phase1_denoised.csv")
    print(f"Input: {df.shape}")

    # Phase 2: PCA
    df_pca, pca_info = apply_family_pca(
        df,
        train_fraction=0.8,
        variance_threshold=0.95,
        max_components=10
    )
    print(f"\nAfter PCA: {df_pca.shape}")

    # Save Phase 2
    df_pca.to_csv("data/features/train_phase2_denoised.csv", index=False)
    print("Saved: data/features/train_phase2_denoised.csv")

    # Phase 3: Stability filter
    df_filtered, filter_info = apply_stability_filter(
        df_pca,
        train_fraction=0.8,
        ic_threshold=0.01,
        std_threshold=0.2,
        window=252
    )
    print(f"\nAfter stability filter: {df_filtered.shape}")

    # Save Phase 3
    df_filtered.to_csv("data/features/train_phase3_denoised.csv", index=False)
    print("Saved: data/features/train_phase3_denoised.csv")

    print("\n=== Complete ===")
    print(f"Phase 1: {df.shape}")
    print(f"Phase 2 (PCA): {df_pca.shape}")
    print(f"Phase 3 (Filter): {df_filtered.shape}")
