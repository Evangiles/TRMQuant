# Financial Denoising

Diffusion-based denoising framework for financial time series data using Mamba state-space models.

## Overview

This module provides a complete pipeline for denoising financial time series features:
- **Cluster-based denoising**: Groups features by characteristics (mean-reverting, trending, random walk)
- **Diffusion models**: VP-SDE (Variance Preserving SDE) with Mamba architecture
- **Causal inference**: Leak-free denoising for production deployment

## Directory Structure

```
FinancialDenoising/
├── models/
│   ├── diffusion_mamba/      # VP-SDE diffusion models with Mamba
│   └── clustering/            # Feature clustering algorithms
├── training/
│   ├── train_denoiser.py      # Train cluster-specific denoisers
│   └── cluster_features.py    # Feature clustering pipeline
├── inference/
│   ├── denoise_causal.py      # Causal denoising (stride=1, no leakage)
│   ├── denoise_dataset.py     # Non-causal denoising (stride=60)
│   └── visualize_denoising.py # Visualization tools
├── analysis/
│   └── analyze_denoised_data.py
├── evaluation/
│   └── validate_denoising.py
├── clustering_results/        # Saved cluster configurations
├── trained_models/            # Saved model checkpoints
└── utils/                     # Shared utilities
```

## Quick Start

### 1. Feature Clustering

```bash
python FinancialDenoising/training/cluster_features.py \
    --data_path TinyRecursiveModels/CSVs/train_only.csv \
    --output_dir FinancialDenoising/clustering_results \
    --n_clusters 7
```

### 2. Train Denoisers

```bash
# Train for each cluster (0-6)
for i in {0..6}; do
    python FinancialDenoising/training/train_denoiser.py \
        --cluster_id $i \
        --data_path TinyRecursiveModels/CSVs/train_only.csv \
        --epochs 100 \
        --device cuda
done
```

### 3. Causal Denoising (Production)

```bash
python FinancialDenoising/inference/denoise_causal.py \
    --input_csv TinyRecursiveModels/CSVs/train_only.csv \
    --output_csv train_denoised_causal.csv \
    --device cuda
```

**Key**: Uses stride=1 with only past 60 rows → no future leakage

### 4. Non-Causal Denoising (Evaluation)

```bash
python FinancialDenoising/inference/denoise_dataset.py \
    --input_csv TinyRecursiveModels/CSVs/train_only.csv \
    --output_csv train_denoised.csv \
    --stride 60 \
    --device cuda
```

**Note**: Uses stride=60 with full window → faster but not for production

## Model Architecture

- **Base**: Bidirectional Mamba state-space model
- **Diffusion**: VP-SDE with linear β schedule (β_min=0.0001, β_max=0.02)
- **Guidance**: Cluster-specific losses (TV loss for mean-reverting, Fourier loss for trending)
- **Training**: Noise prediction with geometric mean guidance

## Validation

Use `Common/evaluation/validate_trading_signals.py` to compare original vs denoised data:

```bash
python Common/evaluation/validate_trading_signals.py \
    --train_original TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised train_denoised_causal.csv \
    --val_original TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised val_denoised_causal.csv
```

## Key Files

- `models/diffusion_mamba/vp_sde.py`: VP-SDE implementation
- `models/diffusion_mamba/denoiser.py`: Mamba denoiser architecture
- `models/diffusion_mamba/losses.py`: Guidance losses (TV, Fourier)
- `inference/denoise_causal.py`: **Production-ready causal denoising**

## Dependencies

- PyTorch
- NumPy, Pandas
- scikit-learn (clustering)
- tqdm

## Citation

If you use this denoising framework, please cite:
```
@misc{financialdenoising2025,
  title={Diffusion-based Financial Time Series Denoising with Mamba},
  author={Your Name},
  year={2025}
}
```
