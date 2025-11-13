# Common Utilities

Shared evaluation and utility functions used across multiple modules (FinancialDenoising, TinyRecursiveModels).

## Overview

This directory contains common utilities that are used by both the denoising pipeline and TRM models.

## Directory Structure

```
Common/
└── evaluation/
    └── validate_trading_signals.py
```

## validate_trading_signals.py

Compares trading performance between original and denoised (or processed) datasets using competition metrics.

### Usage

```bash
python Common/evaluation/validate_trading_signals.py \
    --train_original TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised TinyRecursiveModels/CSVs/train_denoised.csv \
    --val_original TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised TinyRecursiveModels/CSVs/val_denoised.csv
```

### Metrics

**Primary Metric (Competition)**:
- **Adjusted Sharpe Ratio**: `Sharpe / (vol_penalty * return_penalty)`
  - Vol penalty: Applied when strategy volatility > 1.2x market volatility
  - Return penalty: Applied when strategy underperforms market

**Reference Metrics**:
- Sharpe Ratio (geometric mean)
- Cumulative Returns
- Maximum Drawdown
- Win Rate
- Information Coefficient (IC)

### Strategy

- **Position-based**: Predictions scaled to [0, 2] range
- **Formula**: `strategy_returns = rf * (1 - position) + position * forward_returns`
- **Leak-free scaling**: Uses training set min/max for position scaling

### Output

Saves comparison results to `TinyRecursiveModels/evaluation_results/`:
- `original_trading_results.csv`
- `denoised_trading_results.csv`
- `trading_adjusted_sharpe_comparison.csv`
- `trading_sharpe_comparison.csv`
- `trading_returns_comparison.csv`

## Usage from Other Modules

### From FinancialDenoising

```bash
python Common/evaluation/validate_trading_signals.py \
    --train_original TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised train_denoised_causal.csv \
    --val_original TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised val_denoised_causal.csv
```

### From TinyRecursiveModels

```bash
python Common/evaluation/validate_trading_signals.py \
    --train_original TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised TRM_predictions_train.csv \
    --val_original TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised TRM_predictions_val.csv
```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost (optional)

## File Requirements

All CSV files must contain:
- `date_id`: Date identifier
- `forward_returns`: Target variable
- `risk_free_rate`: Risk-free rate for trading strategy
- `market_forward_excess_returns`: Market excess returns (excluded from features)
- Feature columns (e.g., F1, F2, ..., F94)
