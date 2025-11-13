"""
Denoise Complete Dataset with Trained Group-Specific Models (NO TEMPORAL LEAKAGE)

Applies trained denoiser models using non-overlapping windows:
- stride=window_size (default 60) → No overlap between windows
- Window [0:60] processes rows 0-59 independently
- Window [60:120] processes rows 60-119 independently
- No overlap → No temporal leakage → Suitable for real-time deployment

Usage:
    python inference/denoise_dataset.py \
        --input_csv train_only.csv \
        --output_csv train_denoised_no_leakage.csv \
        --models_dir trained_models \
        --stride 60
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_mamba import MultivariateMambaDenoiser, VPSDE


def load_cluster_config(config_path: Path):
    """Load cluster assignments."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_model_checkpoint(checkpoint_path: Path, device: str):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MultivariateMambaDenoiser(
        n_features=checkpoint['n_features'],
        window_size=checkpoint['window_size'],
        d_model=checkpoint['d_model'],
        n_layers=checkpoint['n_layers']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def denoise_windows(
    model: MultivariateMambaDenoiser,
    windows: np.ndarray,
    sde: VPSDE,
    device: str,
    num_steps: int = 50,
    batch_size: int = 64
) -> np.ndarray:
    """
    Denoise multiple windows using the trained model.

    Args:
        model: Trained denoiser
        windows: [N, window_size, n_features]
        sde: VP-SDE for reverse diffusion
        device: Device
        num_steps: Number of denoising steps
        batch_size: Batch size for inference

    Returns:
        Denoised windows [N, window_size, n_features]
    """
    n_windows = len(windows)
    denoised_windows = []

    # Process in batches
    for i in tqdm(range(0, n_windows, batch_size), desc="Denoising"):
        batch = windows[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)

        # Start from noisy data (we use original data as "noisy")
        # In practice, we do single-step denoising from t=T/2
        t = torch.full(
            (batch_tensor.shape[0],),
            sde.num_timesteps // 2,
            device=device,
            dtype=torch.long
        )

        # Model predicts NOISE (not clean signal!)
        predicted_noise = model(batch_tensor, t)

        # Recover clean signal using VP-SDE formula: x0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
        alpha_bar = sde.alphas_cumprod[t]
        while alpha_bar.dim() < batch_tensor.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        denoised = (batch_tensor - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / (torch.sqrt(alpha_bar) + 1e-8)
        denoised = torch.clamp(denoised, -10, 10)  # Stability

        denoised_windows.append(denoised.cpu().numpy())

    return np.concatenate(denoised_windows, axis=0)


def create_windows_no_overlap(
    data: np.ndarray,
    window_size: int,
    stride: int
) -> tuple:
    """
    Create non-overlapping windows.

    With stride=window_size (e.g., 60), windows are completely independent:
    - Window [0:60] processes rows 0-59
    - Window [60:120] processes rows 60-119
    - No overlap → No temporal leakage

    Args:
        data: [T, F] time series data
        window_size: Window size
        stride: Stride for sliding window (use stride=window_size for no overlap)

    Returns:
        (windows, positions) where:
        - windows: [N, window_size, F]
        - positions: [N] start positions
    """
    windows = []
    positions = []

    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
        positions.append(i)

    return np.array(windows), np.array(positions)


def reconstruct_no_overlap(
    denoised_windows: np.ndarray,
    positions: np.ndarray,
    total_length: int,
    window_size: int,
    original_data: np.ndarray
) -> np.ndarray:
    """
    Reconstruct full time series from non-overlapping windows.

    With stride=window_size, each window is independent:
    - Window [0:60] updates reconstructed[0:60]
    - Window [60:120] updates reconstructed[60:120]
    - No overlap → No temporal leakage

    Remaining points at the end use original data.

    Args:
        denoised_windows: [N, window_size, F]
        positions: [N] start positions
        total_length: Original time series length
        window_size: Window size
        original_data: Original data for remaining points

    Returns:
        Reconstructed time series [total_length, F]
    """
    # Initialize with original data
    reconstructed = original_data.copy()

    # Update each window's full range (no overlap)
    for window, pos in zip(denoised_windows, positions):
        reconstructed[pos:pos + window_size] = window

    return reconstructed


def denormalize_features(data: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Denormalize features using training statistics."""
    return data * std + mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--cluster_config", type=str, default="FinancialDenoising/clustering_results/cluster_assignments.json")
    parser.add_argument("--models_dir", type=str, default="FinancialDenoising/trained_models")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=60)  # Changed from 30 to 60 for causal (no overlap)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("="*80)
    print("DATASET DENOISING")
    print("="*80)
    print(f"\nInput: {args.input_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Device: {args.device}")
    print(f"Window size: {args.window_size}, Stride: {args.stride}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")

    # Load cluster configuration
    cluster_config = load_cluster_config(Path(args.cluster_config))
    n_clusters = cluster_config['metadata']['n_clusters']
    print(f"Number of clusters: {n_clusters}")

    # Initialize SDE
    sde = VPSDE(device=args.device)

    # Initialize output dataframe
    df_denoised = df.copy()

    # Process each cluster
    for cluster_id in range(n_clusters):
        print(f"\n{'─'*80}")
        print(f"Processing Cluster {cluster_id}")
        print('─'*80)

        # Get cluster features
        cluster_key = f"cluster_{cluster_id}"
        cluster_info = cluster_config['clusters'][cluster_key]
        feature_names = cluster_info['features']
        cluster_type = cluster_info.get('type', 'random_walk')

        print(f"Type: {cluster_type}")
        print(f"Features: {len(feature_names)}")

        # Load model
        model_path = Path(args.models_dir) / f"cluster_{cluster_id}_best.pt"
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            print(f"    Skipping cluster {cluster_id}")
            continue

        model, checkpoint = load_model_checkpoint(model_path, args.device)
        print(f"✓ Loaded model from {model_path}")
        print(f"  Training loss: {checkpoint.get('loss', 'N/A'):.6f}")

        # CRITICAL: Load training normalization statistics
        # This prevents validation leakage by using ONLY training data statistics
        if 'normalization_mean' not in checkpoint or 'normalization_std' not in checkpoint:
            raise ValueError(
                f"Checkpoint {model_path} missing normalization statistics!\n"
                "This checkpoint was created with old code. Please retrain the model."
            )

        train_mean = np.array(checkpoint['normalization_mean'])
        train_std = np.array(checkpoint['normalization_std'])
        print(f"  Loaded training statistics:")
        print(f"    Mean range: [{train_mean.min():.6f}, {train_mean.max():.6f}]")
        print(f"    Std range:  [{train_std.min():.6f}, {train_std.max():.6f}]")

        # Extract feature data
        feature_data = df[feature_names].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize using TRAINING statistics (not inference data statistics!)
        normalized_data = (feature_data - train_mean) / train_std
        mean, std = train_mean, train_std

        # Create non-overlapping windows (stride=window_size eliminates temporal leakage)
        windows, positions = create_windows_no_overlap(
            normalized_data,
            args.window_size,
            args.stride
        )
        print(f"Created {len(windows)} non-overlapping windows (stride={args.stride}, no temporal leakage)")

        # Denoise
        denoised_windows = denoise_windows(
            model, windows, sde, args.device,
            num_steps=args.num_steps,
            batch_size=args.batch_size
        )

        # Reconstruct from non-overlapping windows (no temporal leakage)
        denoised_data = reconstruct_no_overlap(
            denoised_windows,
            positions,
            len(feature_data),
            args.window_size,
            normalized_data  # Original data for remaining points at end
        )

        # Denormalize
        denoised_data = denormalize_features(denoised_data, mean, std)

        # Update dataframe
        for i, col in enumerate(feature_names):
            df_denoised[col] = denoised_data[:, i]

        print(f"✓ Cluster {cluster_id} completed")

    # Save
    print(f"\n{'='*80}")
    print("Saving denoised dataset...")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_denoised.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("DENOISING COMPLETED")
    print('='*80)
    print(f"Original data: {args.input_csv}")
    print(f"Denoised data: {args.output_csv}")
    print(f"Total features processed: {sum(len(cluster_config['clusters'][f'cluster_{i}']['features']) for i in range(n_clusters))}")


if __name__ == "__main__":
    main()
