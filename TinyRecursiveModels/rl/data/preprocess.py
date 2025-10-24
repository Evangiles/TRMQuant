from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import csv
import math
import numpy as np
import pandas as pd


TARGET_COLS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]


def _safe_float(x: str) -> float:
    if x is None or x == "":
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def load_market_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load CSV with pandas. Returns (date_id, features[T,F], forward_returns[T], risk_free_rate[T], market_excess_returns[T], feature_names).
    Features are all columns between date_id and the first of the 3 target columns.
    """
    df = pd.read_csv(path)
    cols = list(df.columns)
    date_idx = cols.index("date_id")
    target_indices = [cols.index(c) for c in TARGET_COLS]
    first_target = min(target_indices)

    feat_names = cols[date_idx + 1:first_target]

    date_np = df["date_id"].astype(np.int32).to_numpy()
    feats_np = df[feat_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    fwd_np = pd.to_numeric(df["forward_returns"], errors="coerce").to_numpy(dtype=np.float32)
    rf_np = pd.to_numeric(df["risk_free_rate"], errors="coerce").to_numpy(dtype=np.float32)
    mkt_np = pd.to_numeric(df["market_forward_excess_returns"], errors="coerce").to_numpy(dtype=np.float32)
    return date_np, feats_np, fwd_np, rf_np, mkt_np, feat_names


def impute_forward_back_fill(X: np.ndarray) -> np.ndarray:
    """
    Per-column ffill then bfill; remaining NaN -> column median (pandas ops).
    """
    df = pd.DataFrame(X)
    df = df.ffill().bfill()
    med = df.median(numeric_only=True)
    df = df.fillna(med)
    return df.to_numpy(dtype=np.float32)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / (self.std + self.eps)


def fit_standardizer(X: np.ndarray, clip_quantiles: Tuple[float, float] = (0.001, 0.999)) -> Standardizer:
    Xc = X.copy()
    low = np.quantile(Xc, clip_quantiles[0], axis=0)
    high = np.quantile(Xc, clip_quantiles[1], axis=0)
    Xc = np.clip(Xc, low, high)
    mean = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


