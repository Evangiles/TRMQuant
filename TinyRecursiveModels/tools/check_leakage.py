import os
import argparse
import numpy as np
import pandas as pd


def split_index(n: int, frac: float = 0.8) -> int:
    return int(n * frac)


def drop_columns(df: pd.DataFrame, feature_cols: list[str], threshold: float) -> list[str]:
    miss = df[feature_cols].isna().mean()
    keep = [c for c in feature_cols if miss[c] < threshold]
    return keep


def impute_global(df_feat: pd.DataFrame, mode: str) -> pd.DataFrame:
    X = df_feat.copy()
    if mode == "ffill":
        X = X.ffill().bfill()
        med = X.median(numeric_only=True)
        X = X.fillna(med)
    elif mode == "zero_flag":
        for c in list(X.columns):
            if pd.api.types.is_numeric_dtype(X[c]):
                X[f"{c}_nan"] = X[c].isna().astype(float)
        X = X.fillna(0.0)
    else:
        raise ValueError("Unknown impute mode")
    return X


def impute_split_safe(df_feat: pd.DataFrame, split_idx: int, mode: str) -> pd.DataFrame:
    # Apply imputation separately on train and val segments to avoid using future info across the boundary.
    # For ffill mode, do NOT apply bfill on val (would leak from future within val horizon).
    X = df_feat.copy()
    X_train = X.iloc[:split_idx].copy()
    X_val = X.iloc[split_idx:].copy()

    if mode == "ffill":
        X_train = X_train.ffill().bfill()
        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)

        # val: forward-only fill, then train med for remaining NaNs
        X_val = X_val.ffill()
        X_val = X_val.fillna(med)
    elif mode == "zero_flag":
        for part in (X_train, X_val):
            for c in list(df_feat.columns):
                if pd.api.types.is_numeric_dtype(part[c]):
                    part[f"{c}_nan"] = part[c].isna().astype(float)
            part.fillna(0.0, inplace=True)
    else:
        raise ValueError("Unknown impute mode")

    return pd.concat([X_train, X_val], axis=0)


def standardize_split(X: np.ndarray, split_idx: int) -> np.ndarray:
    train = X[:split_idx]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def run_check(path: str, drop: int, thr: float, impute: str):
    df = pd.read_csv(path)
    cols = list(df.columns)
    date_idx = cols.index("date_id")
    target_cols = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    first_target = min(cols.index(c) for c in target_cols)
    feat_cols = cols[date_idx + 1:first_target]

    # record split index
    split_idx = split_index(len(df))

    if drop:
        keep = drop_columns(df, feat_cols, thr)
        dropped = [c for c in feat_cols if c not in keep]
        feat_cols = keep
    else:
        dropped = []

    # Build feature subframe (after any column drops)
    df_feat = df[feat_cols].copy()
    # original NaN mask for val (aligned to current feature set)
    orig_nan_mask_val = df_feat.iloc[split_idx:].isna()

    # Global pipeline (potentially leaky)
    Xg_df = impute_global(df_feat, impute)
    Xg = Xg_df.to_numpy(dtype=np.float32)
    Xg = standardize_split(Xg, split_idx)

    # Split-safe pipeline (no cross-boundary leakage; no bfill on val)
    Xs_df = impute_split_safe(df_feat, split_idx, impute)
    Xs = Xs_df.to_numpy(dtype=np.float32)
    Xs = standardize_split(Xs, split_idx)

    # Compare val slices; any difference implies global preprocessing used info not available in safe mode
    Xg_val = Xg[split_idx:]
    Xs_val = Xs[split_idx:]
    diff = np.isnan(Xg_val) != np.isnan(Xs_val)
    diff |= np.isfinite(Xg_val) & np.isfinite(Xs_val) & (np.abs(Xg_val - Xs_val) > 1e-6)
    num_diff = int(diff.sum())
    total = int(Xg_val.size)

    # Specific leak indicator for ffill mode: bfill on global caused previously-NaN val entries to become non-NaN
    bfill_leak_count = 0
    if impute == "ffill":
        filled_global = Xg_df.iloc[split_idx:][df_feat.columns].notna()
        # safe val did not apply bfill, so any cell that was NaN originally but non-NaN in global and still NaN in safe implies bfill leak
        safe_val_nonan = Xs_df.iloc[split_idx:][df_feat.columns].notna()
        bfill_leak_mask = (orig_nan_mask_val & filled_global & (~safe_val_nonan)).fillna(False)
        bfill_leak_count = int(bfill_leak_mask.to_numpy(dtype=bool).sum())

    print(f"Config: drop={drop} thr={thr} impute={impute}")
    print(f" - dropped_cols: {len(dropped)}")
    print(f" - val_diff_count: {num_diff}/{total} ({num_diff/total:.4%})")
    if impute == "ffill":
        print(f" - potential_bfill_leak_cells: {bfill_leak_count}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check potential data leakage for preprocessing variants.")
    ap.add_argument("--path", type=str, default=os.path.join("TinyRecursiveModels", "train.csv"))
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    for drop in (1, 0):
        for impute in ("ffill", "zero_flag"):
            run_check(args.path, drop=drop, thr=args.threshold, impute=impute)


