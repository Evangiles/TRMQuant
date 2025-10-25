<!-- 7f20a1fc-f99b-452b-a919-993198b55e4d 08a209b7-bb15-460b-abb2-0c097711afe8 -->
# Supervised Allocation Baseline

#### Open question (please pick one target label scheme)

1. Z-score mapping (default): a*_t = clip(1 + k·zscore(mkt_excess_t), 0, 2), k≈0.5.
2. Sign-based: a*_t ∈ {0,2} by sign(mkt_excess_t), |z|<ε → 1.
3. Percentile buckets: a*_t ∈ {0,0.5,1,1.5,2} by quantiles.

#### Files to add

- `models/recursive_reasoning/trm_supervised.py`: TRM temporal encoder + regression head → allocation ∈ [0,2]
- `train_supervised.py`: end-to-end training (MSE/Huber to target a*), validation via adjusted_sharpe, early stopping, EMA toggle (same env vars as PPO)

#### Implementation details

- Data/Windows: reuse `load_market_csv`, `impute_forward_back_fill/impute_series`, standardize; window=60.
- Model:
- Reuse encoder pattern from `trm_ppo.py` (no carry), head: `CastedLinear(hidden, 1)` + tanh-squash → [0,2].
- Loss: Huber (δ=0.5) to reduce outlier sensitivity; L2 weight decay=1e-4.
- Training:
- Train/val split 80/20, batch size 256 (pure supervised, PyTorch DataLoader with pinned memory), epochs 50, cosine LR, best checkpoint on val adjusted_sharpe.
- EMA optional via `USE_EMA`/`EMA_MU` env vars.
- Metrics:
- During training: print loss; After each epoch: compute `adjusted_sharpe` on val by streaming windows and model predictions.

#### Steps

- Create `trm_supervised.py` with encoder and allocation head (no carry).
- Add `train_supervised.py` that:
- loads CSV → imputes/standardizes → builds sliding windows and target a* (chosen scheme)
- trains the model with Huber loss and cosine LR; optional EMA
- evaluates adjusted_sharpe each epoch; saves best checkpoint
- Keep code isolated; no changes to PPO pipeline.

#### Output

- Checkpoint: `TinyRecursiveModels/checkpoints_supervised.pt`
- Console logs: epoch loss + AdjustedSharpe(val)

### To-dos

- [ ] Add TRM temporal encoder + regression head to predict allocation
- [ ] Implement train_supervised.py with windows, targets, training loop, EMA, metrics