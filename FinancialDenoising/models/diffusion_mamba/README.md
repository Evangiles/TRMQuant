# Group-Specific Multivariate Mamba Denoiser

Statistically-driven diffusion denoising system for heterogeneous financial features.

## Architecture

### 1. Feature Clustering (Phase 1 ✅)
- **Input**: 97 features → **Output**: 7 statistically homogeneous groups
- **Clustering**: K-means on 11 statistical characteristics
  - Hurst exponent, ADF p-value, autocorrelation, skewness, kurtosis, volatility
- **Result**: 7 groups (39%, 17%, 21%, 12%, 9%, 1%, 1%)

### 2. Mamba Architecture (Phase 2 ✅)
- **Selective SSM**: O(L) complexity state space model
- **BiMamba**: Bidirectional temporal scanning
- **Multivariate**: Processes all features in group jointly

### 3. Diffusion Framework
- **VP-SDE**: Variance Preserving SDE with linear β schedule [0.0001, 0.02]
- **Guidance Losses**:
  - **TV Loss**: Smoothness for mean-reverting clusters
  - **Fourier Loss**: High-frequency suppression for trending clusters

## Quick Start (Kaggle)

### 1. Clone Repository
```bash
!git clone https://github.com/Evangiles/TRMQuant.git
%cd TRMQuant
```

### 2. Install Dependencies
```bash
!pip install torch pandas numpy scikit-learn statsmodels matplotlib tqdm
```

### 3. Run Feature Clustering
```bash
!python TinyRecursiveModels/train/cluster_features.py \
    --data_path TinyRecursiveModels/train.csv \
    --visualize
```

### 4. Train Denoiser for Cluster 0
```bash
!python TinyRecursiveModels/train/train_group_denoiser.py \
    --cluster_id 0 \
    --epochs 50 \
    --batch_size 64 \
    --device cuda
```

### 5. Train All Clusters (Parallel)
```python
import os

for cluster_id in range(7):
    cmd = f"python TinyRecursiveModels/train/train_group_denoiser.py \
        --cluster_id {cluster_id} \
        --epochs 50 \
        --batch_size 64 \
        --device cuda"

    print(f"Training cluster {cluster_id}...")
    os.system(cmd)
```

## Model Components

### SelectiveSSM
```python
from models.diffusion_mamba import SelectiveSSM

ssm = SelectiveSSM(d_model=128, d_state=16)
output = ssm(x)  # [B, L, d_model]
```

### BiMambaBlock
```python
from models.diffusion_mamba import BiMambaBlock

bi_mamba = BiMambaBlock(d_model=128)
output = bi_mamba(x)  # Bidirectional scan
```

### MultivariateMambaDenoiser
```python
from models.diffusion_mamba import MultivariateMambaDenoiser

model = MultivariateMambaDenoiser(
    n_features=20,       # Features in cluster
    window_size=60,
    d_model=128,
    n_layers=4
)

# Forward pass
x_noisy = torch.randn(4, 60, 20)  # [B, L, F]
t = torch.randint(0, 1000, (4,))  # Timesteps
output = model(x_noisy, t)
```

### VP-SDE
```python
from models.diffusion_mamba import VPSDE

sde = VPSDE(num_diffusion_timesteps=1000)

# Forward process (add noise)
x_t, noise = sde.sample(x_clean, t)

# Reverse process (denoise)
x_prev = sde.denoise_step(x_t, t, predicted_noise)
```

### Losses
```python
from models.diffusion_mamba import DenoisingLoss

loss_fn = DenoisingLoss(
    cluster_type="mean_reverting",  # or "trending", "random_walk"
    guidance_weight=0.1,
    tv_weight=1.0,
    fourier_weight=1.0
)

loss = loss_fn(predicted_noise, true_noise, denoised_output)
```

## Cluster Characteristics

| Cluster | Features | Type | Strategy |
|---------|----------|------|----------|
| 0 | 38 (39%) | Mean-reverting | TV loss (2x) |
| 1 | 16 (17%) | Mean-reverting | TV loss (2x) |
| 2 | 20 (21%) | Mean-reverting | TV loss (2x) |
| 3 | 12 (12%) | Mean-reverting | TV loss (2x) |
| 4 | 9 (9%) | Trending | Fourier loss (2x) |
| 5 | 1 (1%) | Random walk | Balanced |
| 6 | 1 (1%) | Mean-reverting | TV loss (2x) |

## Training Configuration

### Recommended Hyperparameters
```python
{
    "window_size": 60,
    "d_model": 128,
    "d_state": 16,
    "n_layers": 4,
    "expand_factor": 2,
    "dropout": 0.1,
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 100,
    "guidance_weight": 0.1
}
```

### Expected Training Time (A100 GPU)
- Cluster 0 (38 features): ~30 min/epoch
- Cluster 1 (16 features): ~15 min/epoch
- Cluster 2 (20 features): ~18 min/epoch
- Cluster 3 (12 features): ~12 min/epoch
- Cluster 4 (9 features): ~10 min/epoch
- Clusters 5-6 (1 feature each): ~2 min/epoch

**Total**: ~87 min/epoch × 100 epochs = ~145 hours for all clusters

### Optimization Tips
1. **Reduce epochs**: Start with 20-50 epochs for testing
2. **Batch size**: Increase to 128 if memory allows
3. **Mixed precision**: Use `torch.cuda.amp` for 2x speedup
4. **Checkpoint frequency**: Save every 10 epochs

## Next Steps

### Phase 3: Training Pipeline ⏳
- Batch training script for all clusters
- Distributed training support
- Checkpoint management

### Phase 4: Inference Pipeline ⏳
- Denoising inference script
- Generate denoised datasets
- Window aggregation strategies

### Phase 5: TRM Integration ⏳
- Replace naive denoising with group-specific approach
- Benchmark against baseline
- End-to-end validation

## Complexity Analysis

### Computational Savings
- **Naive approach**: 94 models × O(L²) = O(94L²) per sample
- **Our approach**: 7 models × O(L) = O(7L) per sample
- **Speedup**: ~13.4× theoretical speedup + reduced L² → L

### Memory Footprint
- **Peak memory**: max(38 features) vs. 94 features → **61% reduction**
- **Model files**: 7 checkpoints vs. 94 → **93% reduction**

## References

1. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
2. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
3. [Financial Time Series Denoising via Diffusion Models](https://arxiv.org/abs/2409.02138)
