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
        return self.X[t - self.window:t], int(t)


def standardize_split(X: np.ndarray, split_idx: int) -> np.ndarray:
    train = X[:split_idx]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def main():
    ap = argparse.ArgumentParser(description="Finetune with custom Sharpe-aligned loss")
    ap.add_argument("--path", type=str, default=os.path.join("TinyRecursiveModels", "train.csv"))
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--drop_high_missing", type=int, default=0)
    ap.add_argument("--drop_threshold", type=float, default=0.6)
    ap.add_argument("--impute", type=str, default="zero_flag", choices=["ffill", "zero_flag"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lambda_vol", type=float, default=1.0)
    ap.add_argument("--lambda_gap", type=float, default=0.1)
    ap.add_argument("--lambda_turn", type=float, default=0.0)
    ap.add_argument("--warm_start", type=str, default=os.path.join("TinyRecursiveModels", "checkpoints_supervised.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    df = pd.read_csv(args.path)
    cols = list(df.columns)
    date_idx = cols.index("date_id")
    target_cols = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    first_target = min(cols.index(c) for c in target_cols)
    feat_cols = cols[date_idx + 1:first_target]

    # Drop high-missing if requested
    if args.drop_high_missing:
        miss = df[feat_cols].isna().mean()
        feat_cols = [c for c in feat_cols if miss[c] < args.drop_threshold]

    # Impute features
    X_df = df[feat_cols].copy()
    flag_cols = []
    if args.impute == "ffill":
        # split-safe imputation: train ffill+bfill+median, val ffill + train-median
        split_idx = int(len(X_df) * 0.8)
        X_tr = X_df.iloc[:split_idx].copy().ffill().bfill()
        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med)
        X_val = X_df.iloc[split_idx:].copy().ffill().fillna(med)
        X_df = pd.concat([X_tr, X_val], axis=0)
    else:
        for c in feat_cols:
            if pd.api.types.is_numeric_dtype(X_df[c]):
                X_df[f"{c}_nan"] = X_df[c].isna().astype(float)
                flag_cols.append(f"{c}_nan")
        X_df = X_df.fillna(0.0)

    # Targets arrays
    fwd = impute_series(pd.to_numeric(df["forward_returns"], errors="coerce").to_numpy())
    rf = impute_series(pd.to_numeric(df["risk_free_rate"], errors="coerce").to_numpy())
    mkt_ex = impute_series(pd.to_numeric(df["market_forward_excess_returns"], errors="coerce").to_numpy())

    # To numpy and standardize using train stats
    X = X_df.to_numpy(dtype=np.float32)
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
        X = standardize_split(X, split_idx)

    # Datasets/loaders with indices so we can fetch returns for each sample
    window = args.window
    base_ds = WindowDataset(X, window)
    # split time-wise
    n = len(base_ds)
    n_train = int(n * 0.8)
    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n)

    class IndexDataset(Dataset):
        def __init__(self, base: WindowDataset, indices: np.ndarray):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            j = self.indices[i]
            xb, t = self.base[j]
            return xb, t

    train_set = IndexDataset(base_ds, idx_train)
    val_set = IndexDataset(base_ds, idx_val)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRMSupervisedRecursive(
        TRMSupRecConfig(
            window_size=window,
            num_features=X.shape[1],
            hidden_size=256,
            num_heads=4,
            expansion=4.0,
            L_layers=1,
            H_cycles=4,
            L_cycles=6,
        )
    ).to(device)

    # Warm start
    if os.path.exists(args.warm_start):
        ckpt = torch.load(args.warm_start, map_location=device)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(10, args.epochs), eta_min=args.lr * 0.1)

    USE_EMA = bool(int(os.environ.get("USE_EMA", "1")))
    EMA_MU = float(os.environ.get("EMA_MU", "0.999"))
    ema = EMAHelper(mu=EMA_MU) if USE_EMA else None
    if ema is not None:
        ema.register(model)

    # prepare torch constants/targets on device
    fwd_t = torch.from_numpy(fwd).to(device=device, dtype=torch.float32)
    rf_t = torch.from_numpy(rf).to(device=device, dtype=torch.float32)
    mkt_ex_t = torch.from_numpy(mkt_ex).to(device=device, dtype=torch.float32)
    sqrt252_t = torch.tensor(np.sqrt(252.0), device=device, dtype=torch.float32)
    best = -1e9
    best_path = os.path.join("TinyRecursiveModels", "checkpoints_finetune.pt")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f"Finetune {epoch+1}/{args.epochs}", leave=False)
        for xb, t_idx in train_loader:
            xb = xb.to(device)
            t_idx = t_idx.to(device=device, dtype=torch.long)
            a = model(xb)  # [B] in [0,2]

            # Fetch returns for these end-times (torch)
            f = fwd_t[t_idx]  # [B]
            r = rf_t[t_idx]
            m = mkt_ex_t[t_idx]

            # Strategy and excess (torch)
            r_str = r * (1.0 - a) + a * f             # [B]
            ex = r_str - r                             # [B]
            mean_ex = ex.mean()
            std_str = r_str.std(unbiased=False) + 1e-8
            sharpe = mean_ex / std_str * sqrt252_t
            L_base = -sharpe

            vol = std_str * sqrt252_t
            mkt_vol = f.std(unbiased=False) * sqrt252_t
            L_vol = args.lambda_vol * torch.clamp(vol - 1.2 * mkt_vol, min=0.0)

            return_gap = torch.clamp((m.mean() - mean_ex) * 100.0 * 252.0, min=0.0)
            L_gap = args.lambda_gap * (return_gap ** 2) / 100.0

            # Turnover (approx within batch order)
            if a.shape[0] > 1:
                turnover = (a[1:] - a[:-1]).abs().mean()
            else:
                turnover = torch.tensor(0.0, device=device)
            L_turn = args.lambda_turn * turnover

            loss = (L_base + L_vol + L_gap + L_turn)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            pbar.update(1)
        pbar.close()
        scheduler.step()

        # Validation Adjusted Sharpe
        model_eval = ema.ema_copy(model) if ema is not None else model
        model_eval.eval()
        with torch.no_grad():
            allocs = []
            t_collect = []
            for xb, t_idx in val_loader:
                xb = xb.to(device)
                a = model_eval(xb)
                allocs.append(a.detach().cpu().numpy())
                t_collect.append(t_idx.numpy())
            allocs = np.concatenate(allocs, axis=0)
            t_collect = np.concatenate(t_collect, axis=0)

        start = 0  # t_collect already holds aligned end-times for val
        fwd_val = fwd[t_collect].astype(np.float64)
        rf_val = rf[t_collect].astype(np.float64)
        df_sol = pd.DataFrame({'forward_returns': fwd_val, 'risk_free_rate': rf_val})
        score = adjusted_sharpe(df_sol, allocs.astype(np.float64))
        print(f"Epoch {epoch+1}: val_adj_sharpe={score:.6f}")
        if score > best:
            best = score
            torch.save({'model': model_eval.state_dict(), 'score': best}, best_path)
            print(f"Saved best finetuned checkpoint to {best_path} (score={best:.6f})")


if __name__ == "__main__":
    main()


