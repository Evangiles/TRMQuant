"""
Causal Denoising Dataset (LEAK-FREE for Real-World Deployment)

Applies trained denoiser models using causal sliding windows (stride=1):
- Row t is denoised using ONLY window [t-59:t] (past 60 rows)
- Model output: [60, F] → Use ONLY last row (row t)
- No future information leakage → Suitable for production deployment

Key Differences from denoise_dataset.py:
- stride=1 (every row) vs stride=60 (every 60 rows)
- Uses last row only vs overwrites entire window
- Causal (past only) vs Non-causal (future info included)

Usage:
    python inference/denoise_causal.py \
        --input_csv train.csv \
        --output_csv train_denoised_causal.csv \
        --models_dir trained_models \
        --device cuda
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
def denoise_single_window(
    model: MultivariateMambaDenoiser,
    window: np.ndarray,
    sde: VPSDE,
    device: str,
    num_steps: int = 50
) -> np.ndarray:
    """
    Denoise a single window and return ONLY the last row.

    Args:
        model: Trained denoiser
        window: [window_size, n_features] normalized
        sde: VP-SDE for reverse diffusion
        device: Device
        num_steps: Number of denoising steps

    Returns:
        Last row of denoised window [n_features]
    """
    # Convert to tensor
    window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # [1, W, F]

    # Denoise at middle timestep
    t = torch.full((1,), sde.num_timesteps // 2, device=device, dtype=torch.long)

    # Model predicts NOISE (not clean signal!)
    predicted_noise = model(window_tensor, t)  # [1, W, F]

    # Recover clean signal using VP-SDE formula: x0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
    alpha_bar = sde.alphas_cumprod[t]
    while alpha_bar.dim() < window_tensor.dim():
        alpha_bar = alpha_bar.unsqueeze(-1)

    denoised_window = (window_tensor - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / (torch.sqrt(alpha_bar) + 1e-8)
    denoised_window = torch.clamp(denoised_window, -10, 10)  # Stability

    # Return ONLY the last row (causal!)
    return denoised_window[0, -1, :].cpu().numpy()  # [F]


def denoise_causal(
    data: pd.DataFrame,
    feature_names: list,
    model: MultivariateMambaDenoiser,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    sde: VPSDE,
    device: str,
    window_size: int = 60,
    num_steps: int = 50
) -> np.ndarray:
    """
    Causal denoising: Each row uses ONLY past data.

    Args:
        data: Input dataframe
        feature_names: List of feature names for this cluster
        model: Trained denoiser model
        train_mean: Training normalization mean
        train_std: Training normalization std
        sde: VP-SDE
        device: Device
        window_size: Window size (default 60)
        num_steps: Denoising steps

    Returns:
        Denoised features [T, n_features]
    """
    # Extract feature data
    feature_data = data[feature_names].values
    feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

    T = len(feature_data)
    n_features = len(feature_names)
    denoised = np.zeros((T, n_features), dtype=np.float32)

    print(f"  Causal denoising {T} rows (stride=1)...")

    for t in tqdm(range(T), desc="  Processing", leave=False):
        if t < window_size - 1:
            # Insufficient history → use original (or partial window)
            denoised[t] = feature_data[t]
            continue

        # Window [t-59:t] (past 60 rows including current)
        window = feature_data[t - window_size + 1:t + 1]  # [60, n_features]

        # Normalize using TRAINING statistics
        window_norm = (window - train_mean) / train_std

        # Denoise and extract ONLY last row
        denoised[t] = denoise_single_window(model, window_norm, sde, device, num_steps)

        # Denormalize
        denoised[t] = denoised[t] * train_std + train_mean

    return denoised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--cluster_config", type=str, default="FinancialDenoising/clustering_results/cluster_assignments.json")
    parser.add_argument("--models_dir", type=str, default="FinancialDenoising/trained_models")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("=" * 80)
    print("CAUSAL DENOISING (LEAK-FREE)")
    print("=" * 80)
    print(f"\nInput: {args.input_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Device: {args.device}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: 1 (causal - every row uses past {args.window_size} rows only)")

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
        print(f"\n{'─' * 80}")
        print(f"Processing Cluster {cluster_id}")
        print('─' * 80)

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

        # CRITICAL: Load training normalization statistics
        if 'normalization_mean' not in checkpoint or 'normalization_std' not in checkpoint:
            raise ValueError(
                f"Checkpoint {model_path} missing normalization statistics!\n"
                "Please retrain the model with updated training code."
            )

        train_mean = np.array(checkpoint['normalization_mean'])
        train_std = np.array(checkpoint['normalization_std'])
        print(f"  Loaded training statistics (mean: [{train_mean.min():.4f}, {train_mean.max():.4f}], "
              f"std: [{train_std.min():.4f}, {train_std.max():.4f}])")

        # Causal denoising
        denoised_features = denoise_causal(
            df,
            feature_names,
            model,
            train_mean,
            train_std,
            sde,
            args.device,
            args.window_size,
            args.num_steps
        )

        # Update dataframe
        for i, col in enumerate(feature_names):
            df_denoised[col] = denoised_features[:, i]

        print(f"✓ Cluster {cluster_id} completed")

    # Save
    print(f"\n{'=' * 80}")
    print("Saving causal denoised dataset...")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_denoised.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("CAUSAL DENOISING COMPLETED")
    print('=' * 80)
    print(f"Original data: {args.input_csv}")
    print(f"Denoised data: {args.output_csv}")
    print(f"Method: Causal sliding window (stride=1)")
    print(f"Each row uses ONLY past {args.window_size} rows → NO FUTURE LEAKAGE")
    print(f"Total features processed: {sum(len(cluster_config['clusters'][f'cluster_{i}']['features']) for i in range(n_clusters))}")


if __name__ == "__main__":
    main()
