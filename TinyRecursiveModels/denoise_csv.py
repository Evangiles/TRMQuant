"""Denoise train.csv using trained diffusion model."""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from models.diffusion import FinancialDenoiser, VPSDE
from models.diffusion.losses import tv_gradient, fourier_gradient


@torch.no_grad()
def denoise_windows_batch(
    model: FinancialDenoiser,
    sde: VPSDE,
    x0_batch: torch.Tensor,
    T_prime: int = 500,
    n_seeds: int = 5,
    corrector_steps: int = 1,
    eta_tv: float = 0.01,
    eta_fourier: float = 0.01,
    omega: float = 1.0,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Denoise batch of windows using noising-denoising procedure.

    Args:
        model: Trained denoiser
        sde: VP-SDE instance
        x0_batch: Input windows [B, L]
        T_prime: Noising level (< num_timesteps)
        n_seeds: Number of random seeds for averaging
        corrector_steps: Langevin MCMC steps
        eta_tv: TV loss step size
        eta_fourier: Fourier loss step size
        omega: Classifier-free guidance scale
        device: Device

    Returns:
        x_denoised: Denoised windows [B, L]
    """
    model.eval()
    x0_batch = x0_batch.to(device)
    B, L = x0_batch.shape

    denoised_list = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)

        # Step 1: Forward noising to T_prime
        t_idx = torch.full((B,), T_prime, device=device, dtype=torch.long)
        x_t, _ = sde.forward_diffusion(x0_batch, t_idx)  # [B, L]

        # Step 2: Reverse denoising from T_prime to 0
        for i in range(T_prime, 0, -1):
            t = torch.full((B,), float(i), device=device)
            t_idx = torch.full((B,), i, device=device, dtype=torch.long)

            # Predictor step (with CFG)
            # Conditional score
            score_cond = model(x_t, t, x0_batch)  # [B, L]

            # Unconditional score
            score_uncond = model(x_t, t, torch.zeros_like(x0_batch))  # [B, L]

            # CFG
            score = omega * score_cond + (1.0 - omega) * score_uncond

            # VP-SDE predictor
            x_t = sde.reverse_step(x_t, i, score)  # [B, L]

            # Corrector steps
            for _ in range(corrector_steps):
                score = model(x_t, t, x0_batch)  # [B, L]
                x_t = sde.corrector_step(x_t, score, step_size=2e-5)  # [B, L]

            # TV loss guidance
            if eta_tv > 0:
                grad_tv = tv_gradient(x_t)  # [B, L]
                x_t = x_t - eta_tv * grad_tv

            # Fourier loss guidance
            if eta_fourier > 0:
                grad_f = fourier_gradient(x_t, x0_batch, threshold=0.1)  # [B, L]
                x_t = x_t - eta_fourier * grad_f

        denoised_list.append(x_t)  # Keep on GPU

    # Average over seeds on GPU
    x_denoised = torch.stack(denoised_list).mean(dim=0)  # [B, L]

    return x_denoised


def denoise_feature_column(
    model: FinancialDenoiser,
    sde: VPSDE,
    feature_series: np.ndarray,
    window_size: int = 60,
    stride: int = 1,
    batch_size: int = 32,
    **denoise_kwargs,
) -> np.ndarray:
    """Denoise entire feature column using CAUSAL rolling windows.

    IMPORTANT: Uses only past data [t-window_size:t] to denoise timestep t.
    This ensures no data leakage - at time t, only historical data is available.

    Args:
        model: Trained denoiser
        sde: VP-SDE instance
        feature_series: [T] feature time series
        window_size: Window size (lookback period)
        stride: Stride for rolling windows (timesteps to skip between denoising)
        batch_size: Number of windows to process in parallel
        **denoise_kwargs: Arguments for denoise_windows_batch

    Returns:
        denoised_series: [T] denoised feature series
        - First window_size points: copied from original (insufficient history)
        - Remaining points: denoised using causal windows
    """
    device = denoise_kwargs.get('device', torch.device('cpu'))
    T = len(feature_series)

    # Initialize with original data (first window_size points won't be denoised)
    denoised = feature_series.copy()

    # CAUSAL WINDOWS: For timestep t, use [t-window_size:t]
    # Start from window_size (first valid timestep with full history)
    window_timesteps = list(range(window_size, T, stride))

    windows = []
    window_means = []
    window_stds = []
    target_timesteps = []

    for t in window_timesteps:
        # Causal window: look back from t
        window = feature_series[t - window_size:t]
        window_mean = window.mean()
        window_std = window.std() + 1e-8
        window_normalized = (window - window_mean) / window_std

        windows.append(window_normalized)
        window_means.append(window_mean)
        window_stds.append(window_std)
        target_timesteps.append(t)

    # Process in batches
    num_windows = len(windows)
    for batch_start in range(0, num_windows, batch_size):
        batch_end = min(batch_start + batch_size, num_windows)

        # Create batch tensor (stays on GPU)
        batch_windows = torch.tensor(
            np.array(windows[batch_start:batch_end]),
            dtype=torch.float32,
            device=device
        )

        # Denoise batch
        batch_denoised = denoise_windows_batch(model, sde, batch_windows, **denoise_kwargs)

        # Move to CPU only once per batch
        batch_denoised_cpu = batch_denoised.cpu().numpy()

        # De-normalize and update ONLY the target timestep (last point of window)
        for i, global_idx in enumerate(range(batch_start, batch_end)):
            t = target_timesteps[global_idx]

            # Denormalize the entire window
            window_denoised = batch_denoised_cpu[i] * window_stds[global_idx] + window_means[global_idx]

            # Update ONLY the last point (timestep t)
            # This is the denoised value for time t using past [t-60:t]
            denoised[t] = window_denoised[-1]

    return denoised


def main():
    parser = argparse.ArgumentParser(description="Denoise train.csv")
    parser.add_argument("--input", type=str, default="train.csv")
    parser.add_argument("--output", type=str, default="train_denoised.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser.pt")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10, help="Stride for rolling windows (smaller = more overlap)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for parallel window processing")
    parser.add_argument("--T_prime", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds for averaging")
    parser.add_argument("--corrector_steps", type=int, default=1)
    parser.add_argument("--eta_tv", type=float, default=0.01)
    parser.add_argument("--eta_fourier", type=float, default=0.01)
    parser.add_argument("--omega", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model
    model = FinancialDenoiser(
        window_size=args.window,
        d_model=256,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create SDE
    sde_config = checkpoint['sde_config']
    sde = VPSDE(
        beta_min=sde_config['beta_min'],
        beta_max=sde_config['beta_max'],
        num_timesteps=sde_config['num_timesteps'],
    )

    print(f"Model loaded (val_loss={checkpoint['val_loss']:.4f})")

    # Load CSV
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    # Extract feature columns
    cols = list(df.columns)
    date_idx = cols.index("date_id")
    target_cols = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    first_target = min(cols.index(c) for c in target_cols)
    feature_cols = cols[date_idx + 1:first_target]

    print(f"Features to denoise: {len(feature_cols)} columns")

    # Extract features - data must be clean!
    X_df = df[feature_cols].copy()

    # Check for NaN - data must be clean!
    nan_count = X_df.isna().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"Found {nan_count} NaN values in input data! "
            f"Please use clean dataset (e.g., train_clean.csv) without NaN."
        )

    # Denoise each feature
    denoised_df = X_df.copy()
    denoise_kwargs = {
        'T_prime': args.T_prime,
        'n_seeds': args.n_seeds,
        'corrector_steps': args.corrector_steps,
        'eta_tv': args.eta_tv,
        'eta_fourier': args.eta_fourier,
        'omega': args.omega,
        'device': device,
    }

    for col in tqdm(feature_cols, desc="Denoising features"):
        feature_series = X_df[col].to_numpy(dtype=np.float32)

        denoised_series = denoise_feature_column(
            model,
            sde,
            feature_series,
            window_size=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            **denoise_kwargs,
        )

        denoised_df[col] = denoised_series

    # Reconstruct full dataframe
    df_out = df.copy()
    df_out[feature_cols] = denoised_df[feature_cols]

    # Save
    df_out.to_csv(args.output, index=False)
    print(f"Denoised CSV saved to {args.output}")


if __name__ == "__main__":
    main()
