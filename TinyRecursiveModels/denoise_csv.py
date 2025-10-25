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
def denoise_window(
    model: FinancialDenoiser,
    sde: VPSDE,
    x0: torch.Tensor,
    T_prime: int = 500,
    n_seeds: int = 5,
    corrector_steps: int = 1,
    eta_tv: float = 0.01,
    eta_fourier: float = 0.01,
    omega: float = 1.0,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Denoise a single window using noising-denoising procedure.

    Args:
        model: Trained denoiser
        sde: VP-SDE instance
        x0: Input window [L]
        T_prime: Noising level (< num_timesteps)
        n_seeds: Number of random seeds for averaging
        corrector_steps: Langevin MCMC steps
        eta_tv: TV loss step size
        eta_fourier: Fourier loss step size
        omega: Classifier-free guidance scale
        device: Device

    Returns:
        x_denoised: Denoised window [L]
    """
    model.eval()
    x0 = x0.to(device)
    L = x0.shape[0]

    denoised_list = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)

        # Step 1: Forward noising to T_prime
        t_idx = torch.tensor([T_prime], device=device)
        x_t, _ = sde.forward_diffusion(x0.unsqueeze(0), t_idx)
        x_t = x_t.squeeze(0)  # [L]

        # Step 2: Reverse denoising from T_prime to 0
        for i in range(T_prime, 0, -1):
            t = torch.tensor([float(i)], device=device)
            t_idx = torch.tensor([i], device=device)

            # Predictor step (with CFG)
            x_t_batch = x_t.unsqueeze(0)  # [1, L]
            c_batch = x0.unsqueeze(0)     # [1, L]

            # Conditional score
            score_cond = model(x_t_batch, t, c_batch).squeeze(0)

            # Unconditional score
            score_uncond = model(x_t_batch, t, torch.zeros_like(c_batch)).squeeze(0)

            # CFG
            score = omega * score_cond + (1.0 - omega) * score_uncond

            # VP-SDE predictor
            x_t = sde.reverse_step(x_t.unsqueeze(0), i, score.unsqueeze(0)).squeeze(0)

            # Corrector steps
            for _ in range(corrector_steps):
                score = model(x_t.unsqueeze(0), t, c_batch).squeeze(0)
                x_t = sde.corrector_step(x_t.unsqueeze(0), score.unsqueeze(0), step_size=2e-5).squeeze(0)

            # TV loss guidance
            if eta_tv > 0:
                grad_tv = tv_gradient(x_t.unsqueeze(0)).squeeze(0)
                x_t = x_t - eta_tv * grad_tv

            # Fourier loss guidance
            if eta_fourier > 0:
                grad_f = fourier_gradient(x_t.unsqueeze(0), c_batch, threshold=0.1).squeeze(0)
                x_t = x_t - eta_fourier * grad_f

        denoised_list.append(x_t.cpu())

    # Average over seeds
    x_denoised = torch.stack(denoised_list).mean(dim=0)

    return x_denoised


def denoise_feature_column(
    model: FinancialDenoiser,
    sde: VPSDE,
    feature_series: np.ndarray,
    window_size: int = 60,
    stride: int = 1,
    **denoise_kwargs,
) -> np.ndarray:
    """Denoise entire feature column using rolling windows.

    Args:
        model: Trained denoiser
        sde: VP-SDE instance
        feature_series: [T] feature time series
        window_size: Window size
        stride: Stride for rolling windows
        **denoise_kwargs: Arguments for denoise_window

    Returns:
        denoised_series: [T] denoised feature series
    """
    T = len(feature_series)
    denoised = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.int32)

    # Compute per-series statistics for normalization
    series_mean = np.mean(feature_series)
    series_std = np.std(feature_series) + 1e-8

    # Rolling windows
    for t_start in range(0, T - window_size + 1, stride):
        t_end = t_start + window_size

        # Extract and normalize window
        window = feature_series[t_start:t_end]
        window_mean = window.mean()
        window_std = window.std() + 1e-8
        window_normalized = (window - window_mean) / window_std

        # Denoise
        window_tensor = torch.from_numpy(window_normalized).float()
        window_denoised = denoise_window(model, sde, window_tensor, **denoise_kwargs)
        window_denoised = window_denoised.numpy()

        # De-normalize
        window_denoised = window_denoised * window_std + window_mean

        # Accumulate
        denoised[t_start:t_end] += window_denoised
        counts[t_start:t_end] += 1

    # Average overlapping windows
    denoised = denoised / np.maximum(counts, 1)

    return denoised


def main():
    parser = argparse.ArgumentParser(description="Denoise train.csv")
    parser.add_argument("--input", type=str, default="train.csv")
    parser.add_argument("--output", type=str, default="train_denoised.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser.pt")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10, help="Stride for rolling windows (smaller = more overlap)")
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

    # Simple imputation (same as training)
    X_df = df[feature_cols].copy()
    X_df = X_df.ffill().bfill()
    median = X_df.median(numeric_only=True)
    X_df = X_df.fillna(median)

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
