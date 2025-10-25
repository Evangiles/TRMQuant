import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from rl.data.preprocess import load_market_csv, impute_forward_back_fill, fit_standardizer, impute_series
from rl.metrics import adjusted_sharpe
from rl.utils.ema import EMAHelper
from models.recursive_reasoning.trm_supervised import TRMSupervised, TRMSupervisedConfig


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
    csv_path = os.path.join("TinyRecursiveModels", "train.csv")
    date_id, X, fwd, rf, mkt_excess, feat_names = load_market_csv(csv_path)

    # impute features and targets
    X = impute_forward_back_fill(X)
    fwd = impute_series(fwd)
    rf = impute_series(rf)
    mkt_excess = impute_series(mkt_excess)

    # standardize features on train split only
    split_idx = int(len(X) * 0.8)
    stdzr = fit_standardizer(X[:split_idx])
    X = stdzr.transform(X)

    window = 60
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
    model = TRMSupervised(
        TRMSupervisedConfig(
            window_size=window,
            num_features=X.shape[1],
            hidden_size=256,
            num_heads=4,
            expansion=4.0,
            L_layers=2,
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
    EPOCHS = 50

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
        import pandas as pd
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


