"""Train diffusion denoiser on financial time series."""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from models.diffusion import FinancialDenoiser, VPSDE


class FeatureWindowDataset(Dataset):
    """Dataset for rolling windows of financial features."""

    def __init__(
        self,
        features: np.ndarray,
        window_size: int = 60,
        stride: int = 20,
        split: str = "train",
        train_fraction: float = 0.8,
    ):
        """
        Args:
            features: [T, F] array of features
            window_size: Length of each window
            stride: Rolling window stride
            split: 'train' or 'val'
            train_fraction: Fraction for train split
        """
        self.features = features
        self.window_size = window_size
        self.stride = stride

        # Generate all valid window start indices
        T, F = features.shape
        all_indices = np.arange(0, T - window_size + 1, stride)

        # Split train/val
        split_idx = int(len(all_indices) * train_fraction)
        if split == "train":
            self.indices = all_indices[:split_idx]
        else:
            self.indices = all_indices[split_idx:]

        self.num_features = F

    def __len__(self):
        return len(self.indices) * self.num_features

    def __getitem__(self, idx):
        # Map idx to (window_idx, feature_idx)
        window_idx = idx // self.num_features
        feature_idx = idx % self.num_features

        # Get window start
        t_start = self.indices[window_idx]
        t_end = t_start + self.window_size

        # Extract feature window
        window = self.features[t_start:t_end, feature_idx]

        # Per-window z-score normalization
        mean = window.mean()
        std = window.std() + 1e-8
        window_normalized = (window - mean) / std

        return torch.from_numpy(window_normalized).float()


def train_step(
    model: nn.Module,
    sde: VPSDE,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    p_uncond: float = 0.1,
) -> float:
    """Single training step with classifier-free guidance.

    Args:
        model: Denoiser model
        sde: VP-SDE instance
        batch: [B, L] batch of windows
        optimizer: Optimizer
        device: Device
        p_uncond: Probability of unconditional training

    Returns:
        loss: Scalar loss value
    """
    batch = batch.to(device)
    B, L = batch.shape

    # Sample random timesteps
    t_idx = torch.randint(0, sde.num_timesteps, (B,), device=device)

    # Forward diffusion
    noise = torch.randn_like(batch)
    x_t, _ = sde.forward_diffusion(batch, t_idx, noise)

    # Classifier-free guidance: randomly drop condition
    mask = torch.rand(B, device=device) > p_uncond
    cond = batch * mask.unsqueeze(-1)

    # Predict score
    score_pred = model(x_t, t_idx.float(), cond)

    # Compute target score: ∇log p(x_t|x_0) = -ε / √(1-α_t)
    alpha_t = sde.get_alpha(t_idx).unsqueeze(-1)
    score_target = -noise / torch.sqrt(1.0 - alpha_t)

    # Weighted MSE loss (reduction='none' for proper per-sample weighting)
    loss_per_sample = nn.functional.mse_loss(score_pred, score_target, reduction='none')  # [B, L]
    loss = ((1.0 - alpha_t) * loss_per_sample.mean(dim=-1)).mean()  # scalar

    # Backward
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(loss.detach().cpu())


@torch.no_grad()
def validate(
    model: nn.Module,
    sde: VPSDE,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate model.

    Args:
        model: Denoiser model
        sde: VP-SDE instance
        val_loader: Validation dataloader
        device: Device

    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        batch = batch.to(device)
        B, L = batch.shape

        # Sample random timesteps
        t_idx = torch.randint(0, sde.num_timesteps, (B,), device=device)

        # Forward diffusion
        noise = torch.randn_like(batch)
        x_t, _ = sde.forward_diffusion(batch, t_idx, noise)

        # Predict score (with condition)
        score_pred = model(x_t, t_idx.float(), batch)

        # Target score
        alpha_t = sde.get_alpha(t_idx).unsqueeze(-1)
        score_target = -noise / torch.sqrt(1.0 - alpha_t)

        # Loss (proper weighting)
        loss_per_sample = nn.functional.mse_loss(score_pred, score_target, reduction='none')
        loss = ((1.0 - alpha_t) * loss_per_sample.mean(dim=-1)).mean()

        total_loss += float(loss.cpu()) * B
        count += B

    return total_loss / count


def main():
    parser = argparse.ArgumentParser(description="Train diffusion denoiser")
    parser.add_argument("--path", type=str, default="train.csv")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading {args.path}...")
    df = pd.read_csv(args.path)

    # Extract feature columns (everything except date_id and targets)
    cols = list(df.columns)
    date_idx = cols.index("date_id")
    target_cols = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    first_target = min(cols.index(c) for c in target_cols)
    feature_cols = cols[date_idx + 1:first_target]

    print(f"Features: {len(feature_cols)} columns")

    # Extract features as numpy array
    X_df = df[feature_cols].copy()

    # Simple imputation: forward fill + backward fill + median
    X_df = X_df.ffill().bfill()
    median = X_df.median(numeric_only=True)
    X_df = X_df.fillna(median)

    X = X_df.to_numpy(dtype=np.float32)
    print(f"Data shape: {X.shape}")

    # Create datasets
    train_dataset = FeatureWindowDataset(
        X, window_size=args.window, stride=args.stride, split="train"
    )
    val_dataset = FeatureWindowDataset(
        X, window_size=args.window, stride=args.stride, split="val"
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = FinancialDenoiser(
        window_size=args.window,
        d_model=256,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create SDE
    sde = VPSDE(beta_min=0.0001, beta_max=0.02, num_timesteps=1000)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Training loop
    best_val_loss = float('inf')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "denoiser.pt")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        train_losses = []

        for batch in pbar:
            loss = train_step(model, sde, batch, optimizer, device, p_uncond=0.1)
            train_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_train_loss = np.mean(train_losses)

        # Validation
        val_loss = validate(model, sde, val_loader, device)

        scheduler.step()

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'sde_config': {
                    'beta_min': sde.beta_min,
                    'beta_max': sde.beta_max,
                    'num_timesteps': sde.num_timesteps,
                },
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (val_loss={val_loss:.4f})")

    print("Training complete!")


if __name__ == "__main__":
    main()
