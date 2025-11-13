"""
Train Group-Specific Mamba Denoisers

Trains separate denoiser models for each feature cluster
with cluster-specific guidance strategies.

Usage:
    python train/train_group_denoiser.py --cluster_id 0 --epochs 100
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_mamba import (
    MultivariateMambaDenoiser,
    VPSDE,
    DenoisingLoss
)


class TimeSeriesWindowDataset(Dataset):
    """
    Dataset for windowed multivariate time series.

    Creates sliding windows from time series data.

    IMPORTANT: Normalization statistics must be pre-computed externally
    to prevent validation leakage when creating separate train/val datasets.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list,
        window_size: int = 60,
        stride: int = 1,
        normalization_mean: np.ndarray = None,
        normalization_std: np.ndarray = None
    ):
        """
        Args:
            data: DataFrame with time series
            feature_cols: List of feature column names
            window_size: Window size
            stride: Stride for sliding window
            normalization_mean: Pre-computed mean for normalization (required)
            normalization_std: Pre-computed std for normalization (required)
        """
        self.window_size = window_size
        self.feature_cols = feature_cols

        # Extract feature data
        feature_data = data[feature_cols].values  # [T, F]

        # Handle NaN and inf
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Use pre-computed normalization statistics
        if normalization_mean is None or normalization_std is None:
            raise ValueError(
                "normalization_mean and normalization_std must be provided to prevent data leakage. "
                "Compute statistics externally before creating dataset."
            )

        feature_data = (feature_data - normalization_mean) / normalization_std

        # Create windows
        self.windows = []
        for i in range(0, len(feature_data) - window_size + 1, stride):
            window = feature_data[i:i + window_size]
            self.windows.append(window)

        self.windows = np.array(self.windows)  # [N, W, F]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])  # [W, F]
        return window


def load_cluster_config(config_path: Path):
    """Load cluster assignments from JSON."""
    with open(config_path, 'r') as f:
        cluster_config = json.load(f)
    return cluster_config


def get_cluster_features(cluster_config: dict, cluster_id: int):
    """Extract feature names for a specific cluster."""
    cluster_key = f"cluster_{cluster_id}"
    if cluster_key not in cluster_config['clusters']:
        raise ValueError(f"Cluster {cluster_id} not found")

    cluster_info = cluster_config['clusters'][cluster_key]
    return cluster_info['features'], cluster_info.get('type', 'random_walk')


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    sde: VPSDE,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    accumulation_steps: int = 1
):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        x0 = batch.to(device)  # [B, W, F]

        # Sample random timesteps
        t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)

        # Forward diffusion: add noise
        x_t, noise = sde.sample(x0, t)

        # Predict noise
        predicted_noise = model(x_t, t)

        # Compute predicted x0 for guidance loss
        alpha_bar = sde.alphas_cumprod[t]
        while alpha_bar.dim() < x_t.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        predicted_x0 = (x_t - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / (torch.sqrt(alpha_bar) + 1e-8)
        predicted_x0 = torch.clamp(predicted_x0, -10, 10)  # Prevent extreme values

        # Compute loss
        loss = loss_fn(predicted_noise, noise, predicted_x0)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  NaN/Inf detected in loss! Skipping batch...")
            continue

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward
        loss.backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Strong gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", type=int, required=True, help="Cluster ID to train")
    parser.add_argument("--data_path", type=str, default="TinyRecursiveModels/CSVs/train_only.csv")
    parser.add_argument("--cluster_config", type=str, default="FinancialDenoising/clustering_results/cluster_assignments.json")
    parser.add_argument("--output_dir", type=str, default="FinancialDenoising/trained_models")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)  # Empirically validated
    parser.add_argument("--guidance_weight", type=float, default=0.1)
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("="*80)
    print("GROUP-SPECIFIC DENOISER TRAINING")
    print("="*80)
    print(f"\nCluster ID: {args.cluster_id}")
    print(f"Device: {args.device}")

    # Load cluster configuration
    cluster_config = load_cluster_config(Path(args.cluster_config))
    feature_names, cluster_type = get_cluster_features(cluster_config, args.cluster_id)

    print(f"Cluster type: {cluster_type}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {', '.join(feature_names[:10])}")
    if len(feature_names) > 10:
        print(f"          ... and {len(feature_names)-10} more")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} rows")

    # Validate that we're using train-only data
    if 'train_only' not in str(args.data_path):
        print(f"\n⚠️  WARNING: Training on {args.data_path}")
        print(f"    Expected 'train_only.csv' to prevent validation leakage!")
        print(f"    Proceeding anyway, but ensure this is intentional.\n")

    # Pre-compute normalization statistics from training data
    # This must be done BEFORE creating dataset to prevent leakage
    print(f"\nComputing normalization statistics...")
    feature_data = df[feature_names].values
    feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

    normalization_mean = np.mean(feature_data, axis=0, keepdims=True)
    normalization_std = np.std(feature_data, axis=0, keepdims=True) + 1e-8

    print(f"  Mean range: [{normalization_mean.min():.6f}, {normalization_mean.max():.6f}]")
    print(f"  Std range:  [{normalization_std.min():.6f}, {normalization_std.max():.6f}]")

    # Create dataset with pre-computed statistics
    dataset = TimeSeriesWindowDataset(
        df,
        feature_names,
        window_size=args.window_size,
        stride=1,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std
    )
    print(f"Created {len(dataset)} windows")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False
    )

    # Initialize model
    print(f"\nInitializing model...")
    model = MultivariateMambaDenoiser(
        n_features=len(feature_names),
        window_size=args.window_size,
        d_model=args.d_model,
        n_layers=args.n_layers
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Initialize SDE
    sde = VPSDE(device=args.device)

    # Initialize loss
    loss_fn = DenoisingLoss(
        cluster_type=cluster_type,
        guidance_weight=args.guidance_weight
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler with warmup
    def get_lr_scale(epoch, warmup_epochs=5):
        """Linear warmup then constant"""
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Apply learning rate warmup
        lr_scale = get_lr_scale(epoch, warmup_epochs=5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * lr_scale

        avg_loss = train_epoch(model, dataloader, sde, loss_fn, optimizer, args.device, args.accumulation_steps)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}, LR: {args.lr * lr_scale:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_dir / f"cluster_{args.cluster_id}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'cluster_id': args.cluster_id,
                'cluster_type': cluster_type,
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'window_size': args.window_size,
                'd_model': args.d_model,
                'n_layers': args.n_layers,
                # CRITICAL: Save normalization statistics for leak-free inference
                'normalization_mean': normalization_mean.tolist(),
                'normalization_std': normalization_std.tolist(),
            }, checkpoint_path)

            print(f"  → Saved checkpoint to {checkpoint_path}")

    print(f"\nTraining completed! Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
