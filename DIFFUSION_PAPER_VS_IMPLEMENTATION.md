# Diffusion Model for Financial Denoising: Paper vs Implementation

**Document Version**: 1.0
**Created**: 2025-11-12
**Paper**: [arXiv:2409.02138v1](https://arxiv.org/abs/2409.02138) - "A Financial Time Series Denoiser Based on Diffusion Model"
**Implementation**: TRMQuant GitHub Repository

---

## Executive Summary

This document provides a comprehensive comparison between the original paper's diffusion-based denoising approach and the current TRMQuant implementation. While both share the core VP-SDE framework, they differ significantly in **architecture**, **data scope**, **training methodology**, and **downstream application**.

### Key Finding
- **Paper**: Single-feature (close price) denoising for direct trading signal generation
- **Implementation**: 94-feature unified denoising as preprocessing for TRM model

---

## 1. Paper Overview (arXiv:2409.02138v1)

### 1.1 Problem Statement
- **Issue**: Financial time series exhibit low signal-to-noise ratio (SNR)
- **Impact**: Degrades ML model performance and trading profitability
- **Solution**: Use diffusion model as a denoiser to improve data quality

### 1.2 Methodology

#### Data
| Dataset | Timeframe | Frequency | Stocks | Samples |
|---------|-----------|-----------|--------|---------|
| 1day | 2014-2023 | Daily | S&P 500 | 47,807 |
| 1hour | 2023.06-2024.04 | Hourly | S&P 500 | 25,985 |
| 5min | 2024.02-2024.04 | 5-minute | S&P 500 | 51,794 |

**Features**:
- **Input**: Close prices only (1D univariate)
- **Window**: Length 60, stride 20
- **Normalization**: Per-window z-score

#### Model Architecture

**Network**: CSDI-style Conditional Transformer
```
Input: x ∈ R^60 (noisy close prices)
Condition: c ∈ R^60 (original x_0)
Time: t ∈ [0, T] (sinusoidal embedding)

Architecture:
├── Input Embedding
├── Positional Encoding
├── Transformer Encoder (4-6 layers, 8 heads, d_model=512)
│   ├── Multi-Head Self-Attention
│   ├── Cross-Attention (condition c)
│   ├── Feed-Forward Network (d_ff=2048)
│   └── Layer Normalization + Residual
└── Output Projection → score ∈ R^60
```

**Parameters**:
- `d_model=512`, `n_heads=8`, `n_layers=6`
- `d_ff=2048`, `dropout=0.1`
- Time embedding: `dim=128`

#### Diffusion Process

**SDE Framework**: VP-SDE (Variance Preserving)

**Forward Process**:
```
dx = -β(t)/2 * x dt + √β(t) dw

Discrete: x_i = √(1-β_i) * x_{i-1} + √β_i * z_{i-1}
```

**Reverse Process**:
```
dx = [-β(t)/2 * x - β(t) * s_θ(x,t,c)] dt + √β(t) dw̄

Discrete: x_{i-1} = 1/√(1-β_i) * [x_i + β_i * s_θ(x_i,i)] + √β_i * z_i
```

**Hyperparameters**:
- Beta schedule: Linear
- `β_min=0.0001`, `β_max=0.02`
- Timesteps: `T=1000`

#### Training Algorithm

**Objective**: Denoising Score Matching
```python
L(θ) = Σ (1 - α_i) * E[||s_θ(x̃,i,c) - ∇log p_αi(x̃|x)||²]
```

**Classifier-Free Guidance**:
- Unconditional probability: `p_uncond=0.1`
- Guidance scale: `ω=1.0`

**Optimizer**:
- AdamW: `lr=1e-4`, `weight_decay=1e-4`
- Scheduler: CosineAnnealing
- Gradient clipping: `max_norm=1.0`

**Training Config**:
- Epochs: 100
- Batch size: 128
- EMA decay: 0.9999

#### Inference Algorithm

**Noising-Denoising Procedure**:
1. **Forward noising**: `x_0 → x_T'` where `T'=500` (50% of max)
2. **Reverse denoising**: `x_T' → x̂_0`
   - Predictor: VP-SDE reverse step
   - Corrector: Langevin MCMC (`M=1` step, `ε=2e-5`)
   - TV guidance: `η_TV=0.01`
   - Fourier guidance: `η_F=0.01` (threshold=0.1)
3. **Multi-seed averaging**: `n_seeds=5`

**Auxiliary Losses**:

**Total Variation (TV) Loss**:
```python
L_TV(x) = Σ |x[i+1] - x[i]|
∇L_TV = sign(x[1:] - x[:-1])
```

**Fourier Loss**:
```python
L_F(x_t, x_0) = ||FFT(x_t) - Filter(FFT(x_0), f=0.1)||²
# Filters frequencies with amplitude < 0.1
```

### 1.3 Experiments & Results

#### Downstream Classification Task

**Setup**:
- Task: Binary classification (future log return sign)
- Horizons: 1, 5, 10 timesteps
- Model: LightGBM (ensemble tree)
- Labels: Calculated from denoised data
- Metrics: F1 Score, MCC (Matthews Correlation Coefficient)

**Results**:

| Method | 1day F1/MCC | 1hour F1/MCC | 5min F1/MCC |
|--------|-------------|--------------|-------------|
| Original | 0.604/0.014 | 0.452/-0.003 | 0.435/0.007 |
| EMA | 0.656/0.120 | 0.558/0.085 | 0.494/0.021 |
| DAE | 0.676/0.183 | 0.563/0.065 | 0.151/0.029 |
| VE-SDE | 0.720/0.332 | 0.774/0.291 | 0.668/0.222 |
| **VP-SDE** | **0.719/0.323** | **0.806/0.329** | **0.798/0.313** |

**Key Insight**: VP-SDE outperforms baselines, especially on higher-frequency data (1hour, 5min)

#### Trading Performance

**Strategies Tested**:
1. **MACD**: Moving Average Convergence Divergence
2. **Bollinger Bands**: Statistical volatility bands
3. **Following-trend**: Trade in predicted direction
4. **Countering-trend**: Trade opposite to prediction

**Metrics**:
- **LOR** (Long-Only Return): Cumulative return from long positions
- **LSR** (Long-Short Return): Cumulative return from both directions
- **NoT** (Number of Trades): Transaction frequency (lower = less cost)

**Results** (MACD Strategy, 1day):

| Method | LOR | LSR | NoT |
|--------|-----|-----|-----|
| Original | -0.6 | -0.6 | 160 |
| EMA | 0.1 | 0.1 | 120 |
| DAE | 0.5 | 0.5 | 80 |
| **VP-SDE** | **1.2** | **1.2** | **40** |

**Key Insight**: Denoised signals generate **3x fewer trades** with **significantly higher returns**

#### Directional Change Events

**Metric**: Number of trend reversals at various thresholds

| Dataset | Threshold | Ori | EMA | DAE | VE-SDE | VP-SDE |
|---------|-----------|-----|-----|-----|--------|--------|
| 1day | 0.01 | 18.15 | 8.96 | 5.72 | 5.98 | 6.50 |
| 1hour | 0.02 | 3.08 | 2.29 | 1.82 | 2.51 | 2.43 |
| 5min | 0.001 | 13.45 | 6.63 | 7.24 | 4.46 | 5.00 |

**Key Insight**: Denoising reduces noise-driven reversals while preserving meaningful trends

---

## 2. Implementation Overview (TRMQuant GitHub)

### 2.1 Context & Purpose

**Objective**: Denoise 94 financial features as preprocessing step for **TRM (Tiny Recursive Model)** training

**Pipeline**:
```
Raw CSV (94 features) → Diffusion Denoiser → Denoised CSV → TRM Training → Trading
```

**Key Difference**: Denoising is **not** the final model, but a **data preprocessing stage**

### 2.2 Data Structure

#### Input Dataset
- **File**: `train.csv` (8,991 rows)
- **Features**: 94 columns (D1-D9, E1-E20, I1-I9, M1-M18, P1-P13, S1-S12, V1-V13)
- **Targets**: 3 columns (forward_returns, risk_free_rate, market_forward_excess_returns)
- **Total**: 97 columns + date_id

#### Feature Processing
```python
# Load features
feature_cols = cols[date_idx+1:first_target]  # 94 features
X = df[feature_cols].to_numpy()  # [8991, 94]

# For each feature (univariate)
for feature_idx in range(94):
    feature_series = X[:, feature_idx]  # [8991]
    # Process as 1D time series
```

### 2.3 Model Architecture

**Network**: `FinancialDenoiser` (CNN-Transformer Hybrid)

```
Input: x ∈ R^60 (univariate feature window)
Condition: c ∈ R^60 (original x_0)
Time: t ∈ [0, 1000] (sinusoidal embedding)

Architecture:
├── Time Embedding
│   └── Sinusoidal(128) → Linear(128→256) → SiLU → Linear(256→256)
│
├── 1D CNN Encoder (local pattern extraction)
│   ├── Conv1d(1→64, k=5) + GroupNorm + SiLU      [B, 64, 60]
│   ├── Conv1d(64→128, k=3) + GroupNorm + SiLU    [B, 128, 60]
│   └── Conv1d(128→256, k=3) + GroupNorm + SiLU   [B, 256, 60]
│
├── Condition Projection
│   └── Conv1d(1→256, k=1) + GroupNorm + SiLU
│
├── Positional Encoding
│   └── Learnable [1, 60, 256]
│
├── Transformer (global dependencies)
│   ├── Layer 1: MultiHeadAttention(d=256, h=4) + FFN(256→1024→256)
│   └── Layer 2: MultiHeadAttention(d=256, h=4) + FFN(256→1024→256)
│
└── 1D CNN Decoder (signal reconstruction)
    ├── ConvTranspose1d(256→128, k=3) + GroupNorm + SiLU  [B, 128, 60]
    ├── ConvTranspose1d(128→64, k=3) + GroupNorm + SiLU   [B, 64, 60]
    └── ConvTranspose1d(64→1, k=5)                        [B, 1, 60]

Output: score ∈ R^60
```

**Key Parameters**:
- `d_model=256` (vs 512 in paper)
- `n_heads=4` (vs 8 in paper)
- `n_layers=2` (vs 6 in paper)
- `d_ff=1024` (vs 2048 in paper)
- Total params: **~1-2M** (vs ~10M+ in paper)

**Design Rationale**:
- **CNN layers**: Capture local temporal patterns (inductive bias for finance)
- **Transformer**: Model long-range dependencies
- **Lightweight**: Faster training/inference for 94 features

### 2.4 Training Methodology

#### Dataset Construction

**Key Innovation**: Unified training across all features

```python
class FeatureWindowDataset:
    def __len__(self):
        return len(self.indices) * self.num_features
        # (num_windows) × 94

    def __getitem__(self, idx):
        window_idx = idx // 94  # Which window
        feature_idx = idx % 94  # Which feature

        # Extract single feature window
        window = features[t_start:t_end, feature_idx]  # [60]

        # Per-window normalization
        normalized = (window - mean) / (std + 1e-8)
        return normalized
```

**Example**:
- Input: `[8991, 94]` features
- Windows: `(8991-60+1) / 20 = 447` windows per feature
- Total samples: `447 × 94 = 42,018` training samples

**Training Loop**:
```python
for epoch in range(50):
    for batch in dataloader:  # batch_size=512
        # Batch contains mixed windows from different features
        # e.g., [D1[0:60], E5[100:160], V13[200:260], ...]

        # Sample timesteps
        t = torch.randint(0, 1000, (batch_size,))

        # Forward diffusion
        x_t = sde.forward_diffusion(x_0, t)

        # Classifier-free guidance (10% unconditional)
        mask = torch.rand(batch_size) > 0.1
        cond = x_0 * mask.unsqueeze(-1)

        # Predict score
        score_pred = model(x_t, t, cond)

        # Loss
        loss = (1 - alpha_t) * ||score_pred - score_target||²

        # Optimize
        optimizer.step()
```

**Key Difference from Paper**:
- **Paper**: Likely trains separate models per feature/dataset
- **Implementation**: **Single shared model** trained on all 94 features

#### Training Configuration

```yaml
data:
  train: train/clean_train.csv
  val: val/clean_val.csv
  window_size: 60
  stride: 20  # Overlapping windows

model:
  d_model: 256
  n_heads: 4
  n_layers: 2
  dropout: 0.1

diffusion:
  beta_min: 0.0001
  beta_max: 0.02
  num_timesteps: 1000

training:
  epochs: 50
  batch_size: 512
  lr: 1e-4
  weight_decay: 1e-4
  p_uncond: 0.1

validation:
  eval_interval: 1000
  save_best: True
```

### 2.5 Inference Pipeline

#### Feature-wise Denoising

```python
def denoise_feature_column(model, sde, feature_series):
    """
    Args:
        feature_series: [8991] 1D time series for one feature

    Returns:
        denoised_series: [8991] denoised version
    """
    T = len(feature_series)  # 8991
    denoised = feature_series.copy()

    # Generate rolling windows (stride=10 for dense coverage)
    for t_start in range(0, T - 60 + 1, 10):
        t_end = t_start + 60
        window = feature_series[t_start:t_end]  # [60]

        # Normalize window
        mean, std = window.mean(), window.std() + 1e-8
        window_norm = (window - mean) / std

        # Denoise with multiple seeds
        denoised_windows = []
        for seed in range(3):  # n_seeds=3
            torch.manual_seed(seed)

            # Forward noising to T'=500
            x_t = sde.forward_diffusion(window_norm, t=500)

            # Reverse denoising (500 → 0)
            for i in range(500, 0, -1):
                # Classifier-free guidance
                score_cond = model(x_t, i, window_norm)
                score_uncond = model(x_t, i, zeros)
                score = 1.0 * score_cond + 0.0 * score_uncond

                # VP-SDE predictor
                x_t = sde.reverse_step(x_t, i, score)

                # Langevin corrector (1 step)
                score = model(x_t, i, window_norm)
                x_t = sde.corrector_step(x_t, score, eps=2e-5)

                # TV guidance
                x_t -= 0.01 * tv_gradient(x_t)

                # Fourier guidance
                x_t -= 0.01 * fourier_gradient(x_t, window_norm)

            denoised_windows.append(x_t)

        # Average over seeds
        x_denoised = mean(denoised_windows)

        # Denormalize
        x_denoised = x_denoised * std + mean

        # Overwrite (later windows overwrite earlier at overlaps)
        denoised[t_start:t_end] = x_denoised

    return denoised
```

#### Full CSV Processing

```python
# Load train.csv
df = pd.read_csv("train.csv")
feature_cols = [col for col in df.columns if col not in
                ['date_id', 'forward_returns', 'risk_free_rate',
                 'market_forward_excess_returns']]

# Denoise each feature
for col in tqdm(feature_cols):  # 94 iterations
    feature_series = df[col].to_numpy()  # [8991]

    denoised_series = denoise_feature_column(
        model, sde, feature_series,
        T_prime=500, n_seeds=3, stride=10
    )

    df[col] = denoised_series

# Save
df.to_csv("train_denoised.csv", index=False)
```

**Inference Config**:
```yaml
inference:
  T_prime: 500        # Noising level (50% of max)
  n_seeds: 3          # Random seeds for averaging
  corrector_steps: 1  # Langevin MCMC steps
  stride: 10          # Dense window coverage
  batch_size: 32      # Parallel window processing

guidance:
  omega: 1.0          # CFG scale
  eta_tv: 0.01        # TV loss step size
  eta_fourier: 0.01   # Fourier loss step size
```

### 2.6 Downstream Application

**Purpose**: Preprocessing for TRM model

```
train_denoised.csv → TRM Supervised Learning → TRM PPO Training → Trading
```

**TRM Pipeline**:
1. Load denoised features
2. Train supervised TRM (return prediction)
3. Fine-tune with custom Sharpe-aligned loss
4. PPO reinforcement learning for trading policy
5. Deploy for live trading

**Key Difference from Paper**:
- **Paper**: Denoised data → LightGBM classification → Trading signals
- **Implementation**: Denoised data → TRM (recursive reasoning) → PPO → Trading

---

## 3. Detailed Comparison

### 3.1 Architecture Differences

| Component | Paper | Implementation | Impact |
|-----------|-------|----------------|--------|
| **Encoder** | Transformer only | **CNN + Transformer** | Better local pattern capture |
| **d_model** | 512 | **256** | 2x faster, less memory |
| **n_heads** | 8 | **4** | Lighter attention |
| **n_layers** | 6 | **2** | 3x faster training |
| **d_ff** | 2048 | **1024** | Reduced capacity |
| **Normalization** | LayerNorm | **GroupNorm (CNN)** + LayerNorm | Better stability |
| **Activation** | GELU | **SiLU** | Faster computation |
| **Parameters** | ~10M+ | **~1-2M** | 5-10x smaller |

**Rationale**:
- Implementation optimizes for **speed** and **memory** to process 94 features
- CNN layers provide **inductive bias** for financial time series
- Lightweight design enables **rapid iteration** during research

### 3.2 Data & Training Differences

| Aspect | Paper | Implementation |
|--------|-------|----------------|
| **Input Features** | 1 (close price) | **94 (all features)** |
| **Data Sources** | 3 datasets (1day, 1hour, 5min) | **1 unified CSV** |
| **Total Timesteps** | Variable per dataset | **8,991** |
| **Window Size** | 60 | **60** ✅ |
| **Training Stride** | 20 | **20** ✅ |
| **Inference Stride** | Not specified | **10** (denser coverage) |
| **Training Samples** | ~47K (1day) | **~42K (447 windows × 94)** |
| **Model Per Feature** | Likely 1 per dataset | **1 shared for all 94** |
| **Batch Size** | 128 | **512** |
| **Epochs** | 100 | **50** |
| **Validation Split** | 4:1 | **Separate val.csv** |

**Key Insight**:
- Paper focuses on **single-feature depth** (different frequencies)
- Implementation focuses on **multi-feature breadth** (all features)

### 3.3 Diffusion Process Comparison

| Parameter | Paper | Implementation | Match |
|-----------|-------|----------------|-------|
| **SDE Type** | VP-SDE | VP-SDE | ✅ |
| **Beta Schedule** | Linear | Linear | ✅ |
| **β_min** | 0.0001 | 0.0001 | ✅ |
| **β_max** | 0.02 | 0.02 | ✅ |
| **Timesteps (T)** | 1000 | 1000 | ✅ |
| **Noising Level (T')** | 500 | 500 | ✅ |
| **CFG Scale (ω)** | 1.0 | 1.0 | ✅ |
| **p_uncond** | 0.1 | 0.1 | ✅ |
| **n_seeds** | 5 | **3** | ⚠️ |
| **Corrector Steps** | 1 | 1 | ✅ |
| **TV Guidance** | Yes (η=0.01) | Yes (η=0.01) | ✅ |
| **Fourier Guidance** | Yes (η=0.01) | Yes (η=0.01) | ✅ |

**Difference**:
- Implementation uses **3 seeds** (vs 5 in paper) for faster inference
- Trade-off: Slightly less averaging, but 40% faster

### 3.4 Training Objective Comparison

**Both use identical loss**:
```python
# Denoising Score Matching Loss
L(θ) = (1 - α_t) * ||s_θ(x_t, t, c) - ∇log p(x_t|x_0)||²

# Where:
# s_θ(x_t, t, c) = predicted score
# ∇log p(x_t|x_0) = -ε / √(1-α_t)  (target score)
```

**Optimizer Comparison**:

| Setting | Paper | Implementation |
|---------|-------|----------------|
| Optimizer | AdamW | AdamW ✅ |
| Learning Rate | 1e-4 | 1e-4 ✅ |
| Weight Decay | 1e-4 | 1e-4 ✅ |
| Scheduler | CosineAnnealing | CosineAnnealing ✅ |
| Gradient Clip | 1.0 | 1.0 ✅ |
| EMA | Yes (0.9999) | **Not used** ⚠️ |

**Difference**:
- Implementation does **not use EMA** for model weights
- Simplifies deployment (no need to maintain shadow weights)

### 3.5 Inference Differences

#### Denoising Procedure

**Paper** (Algorithm 2):
```
1. Choose T' ∈ [0, T]
2. FOR seed = 1 to 5:
     Get x_T' by forward diffusion
     FOR i = K-1 to 0:
         xi ← Predictor(x_{i+1}, t_{i+1}, c)
         FOR j = 1 to M:
             xi ← Corrector(xi, t_{i+1}, c)
         xi ← xi - η_TV * ∇L_TV(xi)
         xi ← xi - η_F * ∇L_F(xi, c)
     Add x0 to list
3. x̂ ← Mean(list)
```

**Implementation**:
```python
# Identical structure, but:
# - 3 seeds instead of 5
# - Processes 94 features sequentially
# - Dense overlapping windows (stride=10)
# - Batch processing for speed
```

#### Window Merging Strategy

**Paper**: Not explicitly detailed (likely averaging overlaps)

**Implementation**: **Overwrite strategy**
```python
# Later windows overwrite earlier ones at overlaps
for t_start in range(0, T-60+1, 10):
    denoised_window = denoise(data[t_start:t_start+60])
    result[t_start:t_start+60] = denoised_window  # Overwrite
```

**Implication**:
- Non-causal (uses future data)
- Matches paper's validation methodology
- **Not suitable for real-time deployment**

### 3.6 Evaluation Differences

| Aspect | Paper | Implementation |
|--------|-------|----------------|
| **Evaluation Task** | Return classification + Trading | **Preprocessing for TRM** |
| **Downstream Model** | LightGBM | **TRM (Recursive Reasoning)** |
| **Metrics** | F1, MCC, LOR, LSR, NoT, DC Events | **TRM performance metrics** |
| **Trading Strategies** | MACD, Bollinger, Following, Countering | **PPO-based policy** |
| **Validation** | Direct trading signals | **Indirect (via TRM)** |

**Key Difference**:
- Paper validates **denoising quality directly**
- Implementation validates **end-to-end pipeline**

---

## 4. Critical Differences Summary

### 4.1 Architectural Differences

#### ✅ Advantages of Implementation

1. **CNN-Transformer Hybrid**
   - Better inductive bias for financial time series
   - Captures both local patterns (CNN) and long-range dependencies (Transformer)
   - More parameter-efficient

2. **Lightweight Design**
   - 5-10x fewer parameters (~1-2M vs ~10M+)
   - Faster training and inference
   - Suitable for 94-feature processing

3. **GroupNorm in CNN**
   - More stable than BatchNorm for small batches
   - Better generalization

#### ⚠️ Potential Limitations

1. **Reduced Capacity**
   - Smaller model may underfit complex patterns
   - Less expressive than full Transformer

2. **Fewer Transformer Layers**
   - May miss subtle long-range dependencies
   - 2 layers vs 6 in paper

### 4.2 Training Methodology Differences

#### ✅ Advantages of Implementation

1. **Unified Multi-Feature Training**
   - Single model learns patterns across all 94 features
   - Better generalization via feature diversity
   - More efficient than 94 separate models

2. **Larger Batch Size**
   - Faster convergence (512 vs 128)
   - Better gradient estimates

3. **Shorter Training**
   - 50 epochs vs 100
   - Faster experimentation

#### ⚠️ Potential Limitations

1. **No EMA**
   - May have less stable final model
   - Paper uses EMA with decay=0.9999

2. **Fewer Random Seeds**
   - 3 seeds vs 5 in inference
   - Slightly noisier outputs

3. **Feature Independence Assumption**
   - Treats each feature as independent 1D series
   - Ignores inter-feature correlations

### 4.3 Data Processing Differences

#### Paper Approach
- **Focus**: Single feature (close price) at multiple frequencies
- **Depth**: 3 datasets (1day, 1hour, 5min) for same feature
- **Total Models**: Likely 3 (one per frequency)
- **Philosophy**: Frequency-specific denoising

#### Implementation Approach
- **Focus**: 94 features at single frequency
- **Breadth**: All feature types (D, E, I, M, P, S, V)
- **Total Models**: 1 (shared across features)
- **Philosophy**: Feature-agnostic denoising

#### Trade-offs

| Aspect | Paper | Implementation |
|--------|-------|----------------|
| **Feature Diversity** | Low (1 type) | **High (94 types)** ✅ |
| **Frequency Diversity** | **High (3 levels)** ✅ | Low (1 level) |
| **Model Specialization** | **Per-frequency** ✅ | Per-feature |
| **Training Efficiency** | 3 models needed | **1 model** ✅ |
| **Deployment Complexity** | Higher | **Lower** ✅ |

### 4.4 Application Context Differences

#### Paper's End-to-End Pipeline
```
Close Prices → Diffusion Denoiser → Denoised Prices
                                         ↓
                                   LightGBM Classifier
                                         ↓
                                   Return Prediction
                                         ↓
                             Trading Signals (MACD/Bollinger)
                                         ↓
                                   Execute Trades
```

#### Implementation's End-to-End Pipeline
```
94 Raw Features → Diffusion Denoiser → 94 Denoised Features
                                              ↓
                                      TRM Supervised Training
                                              ↓
                                      TRM Custom Loss Fine-tuning
                                              ↓
                                      TRM PPO Reinforcement Learning
                                              ↓
                                      Trading Policy
                                              ↓
                                      Execute Trades
```

**Key Insight**:
- **Paper**: Denoising is the **main contribution**
- **Implementation**: Denoising is a **preprocessing step** for TRM

---

## 5. Performance Implications

### 5.1 Expected Differences

#### Denoising Quality

**Paper's Advantages**:
- Larger model capacity (10M+ params)
- Frequency-specific optimization
- More seeds (5) for averaging
- EMA for stable weights

**Implementation's Advantages**:
- Multi-feature training (better generalization)
- CNN inductive bias (better local patterns)
- Denser window coverage (stride=10 vs unspecified)

**Expected Outcome**: **Comparable quality** with different strengths
- Paper: Better on single-feature depth
- Implementation: Better on multi-feature breadth

#### Computational Efficiency

| Metric | Paper | Implementation | Speedup |
|--------|-------|----------------|---------|
| **Model Size** | ~10M params | ~1-2M params | 5-10x |
| **Training Time** | 100 epochs × 128 batch | 50 epochs × 512 batch | ~2x |
| **Inference/Window** | 5 seeds × 500 steps | 3 seeds × 500 steps | 1.67x |
| **Total Inference** | 1 feature | 94 features | 94x slower |
| **Memory Usage** | Higher | Lower | 2-4x |

**Trade-off Analysis**:
- Implementation is **faster per model**
- But needs to process **94x more features**
- Net result: ~50-60x slower overall inference

#### Deployment Considerations

**Paper**:
- ✅ Simpler (1 feature)
- ✅ Direct trading signals
- ❌ Limited to close prices
- ❌ No feature engineering

**Implementation**:
- ✅ Rich feature set (94 features)
- ✅ Integration with TRM
- ❌ Complex pipeline
- ❌ Higher latency

### 5.2 Validation Challenges

**Paper's Validation**:
- Direct metrics: F1, MCC on return classification
- Trading metrics: LOR, LSR, NoT
- Clear attribution to denoising

**Implementation's Validation**:
- Indirect: TRM performance (Sharpe, returns)
- Hard to isolate denoising contribution
- Confounded by TRM architecture

**Recommendation**: Run ablation study
```
TRM on raw features vs TRM on denoised features
→ Measure Sharpe improvement
```

---

## 6. Recommendations

### 6.1 For Understanding the Code

1. **Start with Paper Architecture**
   - Read paper sections 3-4 for diffusion theory
   - Understand VP-SDE forward/reverse processes
   - Study classifier-free guidance

2. **Map to Implementation**
   - `models/diffusion/denoiser.py` → Core model
   - `models/diffusion/vp_sde.py` → SDE framework
   - `train_diffusion_denoiser.py` → Training loop
   - `denoise_csv.py` → Inference pipeline

3. **Key Files to Review**
   ```
   TinyRecursiveModels/
   ├── models/diffusion/
   │   ├── __init__.py         # Exports
   │   ├── denoiser.py         # FinancialDenoiser model
   │   ├── vp_sde.py           # VP-SDE implementation
   │   └── losses.py           # TV & Fourier losses
   ├── train_diffusion_denoiser.py  # Training script
   └── denoise_csv.py          # Inference script
   ```

### 6.2 For Reproducing Paper Results

If you want to match paper's approach:

1. **Use Single Feature**
   ```python
   # Modify FeatureWindowDataset to use only close prices
   feature_cols = ['close']  # Instead of all 94
   ```

2. **Increase Model Capacity**
   ```python
   model = FinancialDenoiser(
       d_model=512,    # vs 256
       n_heads=8,      # vs 4
       n_layers=6,     # vs 2
       d_ff=2048,      # vs 1024
   )
   ```

3. **Use Pure Transformer**
   ```python
   # Remove CNN encoder/decoder
   # Use CSDI-style architecture from paper
   ```

4. **Add EMA**
   ```python
   from torch_ema import ExponentialMovingAverage
   ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
   ```

5. **Increase Seeds**
   ```python
   # In denoise_csv.py
   n_seeds = 5  # vs 3
   ```

### 6.3 For Improving Implementation

1. **Enable Multi-Feature Joint Denoising**
   ```python
   # Instead of processing features independently,
   # treat as multivariate time series
   input_dim = 94  # vs 1
   x_input: [B, 60, 94]  # vs [B, 60]
   ```

2. **Add Feature Correlation Modeling**
   ```python
   # Use cross-attention between features
   # Or use graph neural networks for feature relationships
   ```

3. **Implement Causal Denoising**
   ```python
   # For real-time deployment
   # Use only past data: windows [t-60:t] instead of [t:t+60]
   ```

4. **Add Experiment Tracking**
   ```python
   import wandb
   wandb.init(project="diffusion-denoising")
   wandb.log({"val_loss": loss, "epoch": epoch})
   ```

---

## 7. Conclusion

### 7.1 Summary of Differences

| Category | Paper | Implementation | Verdict |
|----------|-------|----------------|---------|
| **Purpose** | End-to-end denoising for trading | Preprocessing for TRM | Different goals |
| **Data Scope** | 1 feature, 3 frequencies | 94 features, 1 frequency | Different coverage |
| **Architecture** | Full Transformer (10M+ params) | CNN-Transformer (1-2M) | Trade-off speed/capacity |
| **Training** | Per-dataset models | Unified multi-feature model | Different strategy |
| **Inference** | 5 seeds, unknown stride | 3 seeds, stride=10 | Minor difference |
| **Validation** | Direct (F1, MCC, trading) | Indirect (via TRM) | Different methodology |

### 7.2 Key Takeaways

1. **Implementation is NOT a direct reproduction of the paper**
   - Different architecture (hybrid vs pure transformer)
   - Different data scope (94 features vs 1 feature)
   - Different purpose (preprocessing vs end-to-end)

2. **Core diffusion framework is preserved**
   - VP-SDE: ✅ Identical
   - Loss function: ✅ Identical
   - Guidance methods: ✅ Identical (CFG, TV, Fourier)

3. **Implementation makes practical trade-offs**
   - Lighter model for 94-feature processing
   - Faster training/inference
   - Unified architecture for all features

4. **Both approaches are valid**
   - Paper: Deep dive into single-feature denoising
   - Implementation: Broad coverage for TRM integration

### 7.3 Future Work

1. **Validation**
   - Run ablation: TRM with/without denoising
   - Measure isolated denoising contribution

2. **Optimization**
   - Experiment with larger models (closer to paper)
   - Try multivariate denoising (joint 94-feature model)
   - Implement causal denoising for real-time use

3. **Extension**
   - Add frequency-specific models (1day, 1hour, 5min)
   - Integrate paper's trading strategies
   - Compare LightGBM vs TRM downstream

---

## 8. References

**Paper**:
- Wang, Zhuohan, and Carmine Ventre. "A Financial Time Series Denoiser Based on Diffusion Model." arXiv preprint arXiv:2409.02138 (2024).

**Implementation**:
- GitHub: https://github.com/Evangiles/TRMQuant
- Files analyzed:
  - `TinyRecursiveModels/models/diffusion/denoiser.py`
  - `TinyRecursiveModels/models/diffusion/vp_sde.py`
  - `TinyRecursiveModels/train_diffusion_denoiser.py`
  - `TinyRecursiveModels/denoise_csv.py`

**Related Work**:
- Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM)
- Song et al. "Score-Based Generative Modeling through SDEs"
- Tashiro et al. "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation"

---

**Document End**
