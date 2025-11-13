# TinyRecursiveModels for Financial Time Series

Application of Tiny Recursive Models (TRM) to financial time series prediction and trading.

## Overview

This module applies recursive reasoning from the original TRM paper to financial markets:
- **Recursive prediction**: Multi-step refinement of forward returns prediction
- **RL training**: PPO-based training for trading strategy optimization
- **Market environment**: Custom gym environment for backtesting

## Original TRM Paper

Based on: ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)

**Key insight**: Small models with recursive reasoning can achieve strong performance without massive parameter counts.

## Directory Structure

```
TinyRecursiveModels/
├── models/
│   ├── recursive_reasoning/   # TRM model adaptations
│   ├── common.py              # Shared layers
│   └── layers.py              # Custom layers
├── rl/
│   ├── envs/                  # Trading environments
│   ├── ppo.py                 # PPO implementation
│   ├── infer.py               # Model inference
│   └── metrics.py             # Trading metrics
├── training/
│   ├── train_ppo.py           # RL training scripts
│   ├── train_supervised.py    # Supervised training
│   └── train_finetune_custom.py
├── config/                    # Model configurations
├── tools/                     # Analysis utilities
├── utils/                     # Helper functions
└── CSVs/                      # Training data (shared)
```

## Quick Start

### 1. Data Preparation

Data is stored in `TinyRecursiveModels/CSVs/`:
- `train_only.csv`: Training set (80%)
- `val_only.csv`: Validation set (20%)

**Optional**: Apply denoising using `FinancialDenoising` module:
```bash
# See FinancialDenoising/README.md for denoising pipeline
python FinancialDenoising/inference/denoise_causal.py \
    --input_csv TinyRecursiveModels/CSVs/train_only.csv \
    --output_csv train_denoised.csv
```

### 2. Supervised Pre-training

```bash
python TinyRecursiveModels/train_supervised.py \
    --data_path TinyRecursiveModels/CSVs/train_only.csv \
    --window_size 60 \
    --epochs 100
```

### 3. RL Fine-tuning

```bash
python TinyRecursiveModels/train_ppo.py \
    --data_path TinyRecursiveModels/CSVs/train_only.csv \
    --pretrained_model checkpoints/supervised_best.pt \
    --total_timesteps 1000000
```

### 4. Evaluation

```bash
# Using Common validation script
python Common/evaluation/validate_trading_signals.py \
    --train_original TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised TRM_predictions_train.csv \
    --val_original TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised TRM_predictions_val.csv
```

## Model Architecture

### Recursive Reasoning

```
Input features (x) → Embedding
Initial prediction (y₀)
For k in [1, K]:
    z = RecursiveReasoning(x, y_{k-1}, z)  # Update latent
    y_k = UpdateAnswer(z, y_{k-1})          # Refine prediction
Return y_K
```

### Key Adaptations

- **Input**: 60-day window of financial features (94 features)
- **Output**: Forward returns prediction
- **Cycles**: 3-4 recursive reasoning cycles
- **Training**: PPO with Sharpe ratio reward

## Market Environment

`rl/envs/market_env.py` implements:
- **State**: 60-day feature window + portfolio state
- **Action**: Position allocation [0, 2]
- **Reward**: Sharpe ratio, adjusted returns
- **Causal**: Only past data used (no future leakage)

## Metrics

Competition metrics (see `Common/evaluation/validate_trading_signals.py`):
- Adjusted Sharpe Ratio (primary)
- Sharpe Ratio
- Cumulative Returns
- Maximum Drawdown
- Win Rate
- Information Coefficient (IC)

## Integration with Denoising

TRM can use denoised features from `FinancialDenoising`:

```bash
# 1. Denoise features
python FinancialDenoising/inference/denoise_causal.py \
    --input_csv TinyRecursiveModels/CSVs/train_only.csv \
    --output_csv train_denoised.csv

# 2. Train TRM on denoised data
python TinyRecursiveModels/train_supervised.py \
    --data_path train_denoised.csv

# 3. Compare performance
python Common/evaluation/validate_trading_signals.py \
    --train_original TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised train_denoised.csv \
    --val_original TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised val_denoised.csv
```

## Dependencies

- PyTorch
- Gymnasium (RL environment)
- NumPy, Pandas
- Stable-Baselines3 (PPO)

## Citation

Original TRM paper:
```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks},
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871},
}
```

## Related Modules

- **FinancialDenoising**: Feature denoising pipeline (separate module)
- **Common**: Shared evaluation utilities
