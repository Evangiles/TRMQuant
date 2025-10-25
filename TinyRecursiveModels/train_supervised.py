import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from rl.data.preprocess import fit_standardizer, impute_series
from rl.metrics import adjusted_sharpe
from rl.utils.ema import EMAHelper
from models.recursive_reasoning.trm_supervised_recursive import TRMSupervisedRecursive, TRMSupRecConfig


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, window: int):
        self.X = X
        self.window = window
        self.indices = np.arange(window, len(X))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        return self.X[t - self.window:t]


def build_targets_zscore(mkt_excess: np.ndarray, window: int, k: float = 0.5) -> np.ndarray:
    # rolling z-score w.r.t recent window (excluding current)
    xs = []
    for t in range(window, len(mkt_excess)):
        hist = mkt_excess[t - window:t]
        mu = float(np.mean(hist))
        sd = float(np.std(hist) + 1e-8)
        z = (mkt_excess[t] - mu) / sd
        a = np.clip(1.0 + k * z, 0.0, 2.0)
        xs.append(a)
    return np.array(xs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Supervised allocation training with ablations")
    parser.add_argument("--path", type=str, default=os.path.join("TinyRecursiveModels", "train.csv"))
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--drop_high_missing", type=int, default=0, help="1 to drop columns with missing ratio >= threshold")
    parser.add_argument("--drop_threshold", type=float, default=0.6)
    parser.add_argument("--impute", type=str, default="ffill", choices=["ffill", "zero_flag"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # load full dataframe
    df = pd.read_csv(args.path)
    cols = list(df.columns)
    date_idx = cols.index("date_id")
    target_cols = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    first_target = min(cols.index(c) for c in target_cols)
    feature_cols = cols[date_idx + 1:first_target]

    # missing ratio
    miss_ratio = df[feature_cols].isna().mean()
    drop_set = set()
    if args.drop_high_missing:
        drop_set = set(miss_ratio[miss_ratio >= args.drop_threshold].index.tolist())
        feature_cols = [c for c in feature_cols if c not in drop_set]

    # imputation
    X_df = df[feature_cols].copy()
    flag_cols = []
    if args.impute == "ffill":
        X_df = X_df.ffill().bfill()
        med = X_df.median(numeric_only=True)
        X_df = X_df.fillna(med)
    else:  # zero_flag
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(X_df[c]):
                flag = X_df[c].isna().astype(float)
                flag_name = f"{c}_nan"
                X_df[flag_name] = flag
                flag_cols.append(flag_name)
        X_df = X_df.fillna(0.0)

    # targets
    fwd = impute_series(pd.to_numeric(df["forward_returns"], errors="coerce").to_numpy())
    rf = impute_series(pd.to_numeric(df["risk_free_rate"], errors="coerce").to_numpy())
    mkt_excess = impute_series(pd.to_numeric(df["market_forward_excess_returns"], errors="coerce").to_numpy())

    X = X_df.to_numpy(dtype=np.float32)

    # standardize features on train split only (exclude flag columns)
    split_idx = int(len(X) * 0.8)
    if args.impute == "zero_flag" and len(flag_cols):
        base_cols = [i for i, c in enumerate(X_df.columns) if c not in flag_cols]
        flag_idx = [i for i, c in enumerate(X_df.columns) if c in flag_cols]
        X_base = X[:, base_cols]
        X_flag = X[:, flag_idx] if len(flag_idx) else np.empty((len(X), 0), dtype=np.float32)
        stdzr = fit_standardizer(X_base[:split_idx])
        X_base = stdzr.transform(X_base)
        X = np.concatenate([X_base, X_flag], axis=1)
    else:
        stdzr = fit_standardizer(X[:split_idx])
        X = stdzr.transform(X)

    window = args.window
    # build windows and z-score targets
    targets = build_targets_zscore(mkt_excess, window=window, k=0.5)
    ds = WindowDataset(X, window)
    # align dataset to targets length
    assert len(ds) == len(targets)

    # split train/val by time
    n = len(ds)
    n_train = int(n * 0.8)
    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n)

    class IndexDataset(Dataset):
        def __init__(self, base: WindowDataset, indices: np.ndarray, targets: np.ndarray):
            self.base = base
            self.indices = indices
            self.targets = targets
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            j = self.indices[i]
            return self.base[j], self.targets[j]

    train_set = IndexDataset(ds, idx_train, targets)
    val_set = IndexDataset(ds, idx_val, targets)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRMSupervisedRecursive(
        TRMSupRecConfig(
            window_size=window,
            num_features=X.shape[1],
            hidden_size=256,
            num_heads=4,
            expansion=4.0,
            L_layers=2,
            H_cycles=3,
            L_cycles=4,
        )
    ).to(device)

    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=3e-5)
    loss_fn = nn.HuberLoss(delta=0.5)

    USE_EMA = bool(int(os.environ.get("USE_EMA", "1")))
    EMA_MU = float(os.environ.get("EMA_MU", "0.999"))
    ema = EMAHelper(mu=EMA_MU) if USE_EMA else None
    if ema is not None:
        ema.register(model)

    best = -1e9
    best_path = os.path.join("TinyRecursiveModels", "checkpoints_supervised.pt")
    EPOCHS = args.epochs

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS} Train", leave=False)
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            running += float(loss.detach().cpu())
            pbar.update(1)
        pbar.close()
        scheduler.step()

        # validation adjusted sharpe
        model_eval = ema.ema_copy(model) if ema is not None else model
        model_eval.eval()
        with torch.no_grad():
            # stream over val windows to predict allocations
            allocs = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                a = model_eval(xb)
                allocs.append(a.detach().cpu().numpy())
            allocs = np.concatenate(allocs, axis=0)

        # build solution df aligned to val horizon
        start = window + n_train
        fwd_val = fwd[start: start + len(allocs)].astype(np.float64)
        rf_val = rf[start: start + len(allocs)].astype(np.float64)
        df = pd.DataFrame({
            'forward_returns': fwd_val,
            'risk_free_rate': rf_val,
        })
        score = adjusted_sharpe(df, allocs.astype(np.float64))
        print(f"Epoch {epoch+1}: train_loss={running/len(train_loader):.4f} val_adj_sharpe={score:.6f}")
        if score > best:
            best = score
            torch.save({'model': model_eval.state_dict(), 'score': best}, best_path)
            print(f"Saved best supervised checkpoint to {best_path} (score={best:.6f})")


if __name__ == "__main__":
    main()


