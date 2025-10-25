# TRM + Diffusion Denoiser: Model Implementation Specification

## 1. Overview

**Objective**: Integrate Diffusion Model-based denoising with Tiny Recursive Model (TRM) for enhanced financial time series prediction and trading.

**Architecture**: Two-stage pipeline
- **Stage 1**: Conditional Diffusion Model Denoiser (preprocessing)
- **Stage 2**: TRM Recursive Reasoning Model (prediction)

**Key Innovation**: Combine noise reduction (Diffusion) with recursive reasoning (TRM) for robust financial forecasting.

---

## 2. Stage 1: Diffusion Model Denoiser

### 2.1 Model Architecture

#### Base Network: Conditional Transformer
```
Input: x ∈ R^L (noisy time series, length L=60)
Condition: c ∈ R^L (original time series x_0)
Time Embedding: t ∈ [0, T] (sinusoidal encoding)

Network: CSDI-style Conditional Transformer
  ├── Input Embedding Layer
  ├── Positional Encoding
  ├── Transformer Encoder Blocks (N=4-6)
  │   ├── Multi-Head Self-Attention (heads=8)
  │   ├── Cross-Attention (condition c)
  │   ├── Feed-Forward Network (dim=512)
  │   └── Layer Normalization + Residual
  └── Output Projection Layer

Output: s_θ(x, t, c) - score estimation ∈ R^L
```

#### Network Specifications
```python
class ConditionalDiffusionDenoiser(nn.Module):
    def __init__(self,
                 input_dim: int = 1,          # Univariate time series
                 seq_len: int = 60,           # Sequence length
                 d_model: int = 512,          # Model dimension
                 n_heads: int = 8,            # Attention heads
                 n_layers: int = 6,           # Transformer layers
                 d_ff: int = 2048,            # FFN hidden dim
                 dropout: float = 0.1,
                 time_embed_dim: int = 128):  # Time embedding
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_len)

        # Time embedding (sinusoidal)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Conditional projection
        self.cond_proj = nn.Linear(input_dim, d_model)

        # Transformer encoder blocks
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, n_layers)

        # Output projection (score estimation)
        self.output_proj = nn.Linear(d_model, input_dim)
```

### 2.2 Training Algorithm

#### SDE Framework Selection
**Choice**: VP-SDE (Variance Preserving SDE)
- Better performance on hourly/5min data (paper results)
- More stable training than VE-SDE

**Forward Process**:
```
dx = -β(t)/2 * x dt + √β(t) dw
where β(t) follows linear schedule: β_min=0.0001 to β_max=0.02
```

**Reverse Process**:
```
dx = [-β(t)/2 * x - β(t) * s_θ(x,t,c)] dt + √β(t) dw̄
```

#### Training Objective
```python
def training_loss(model, x_0, c):
    """
    Args:
        x_0: Original time series [B, L]
        c: Condition (c = x_0 for self-conditioning) [B, L]

    Returns:
        loss: Denoising score matching loss
    """
    batch_size = x_0.shape[0]

    # Sample random timestep t ~ U(0, T)
    t = torch.rand(batch_size, device=x_0.device) * T

    # Sample noise ε ~ N(0, I)
    epsilon = torch.randn_like(x_0)

    # Forward diffusion: x_t = √α_t * x_0 + √(1-α_t) * ε
    alpha_t = compute_alpha_t(t)  # Cumulative product of (1-β_i)
    x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * epsilon

    # Predict score s_θ(x_t, t, c)
    score_pred = model(x_t, t, c)

    # Compute target score: ∇_x log p(x_t|x_0) = -ε / √(1-α_t)
    score_target = -epsilon / torch.sqrt(1 - alpha_t)

    # Loss: weighted L2 loss
    loss = (1 - alpha_t) * F.mse_loss(score_pred, score_target)

    return loss
```

#### Classifier-Free Guidance
```python
def train_step_with_cfg(model, x_0, p_uncond=0.1):
    """
    Classifier-free guidance training

    Args:
        p_uncond: Probability of unconditional training (10%)
    """
    batch_size = x_0.shape[0]

    # Randomly drop condition with probability p_uncond
    mask = torch.rand(batch_size) > p_uncond
    c = x_0 * mask.unsqueeze(-1).to(x_0.device)  # Zero out condition randomly

    loss = training_loss(model, x_0, c)
    return loss
```

#### Training Configuration
```yaml
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 1e-4
  betas: [0.9, 0.999]

scheduler:
  type: CosineAnnealingLR
  T_max: 100000
  eta_min: 1e-6

training:
  epochs: 100
  batch_size: 128
  gradient_clip: 1.0
  ema_decay: 0.9999  # Exponential moving average
  eval_interval: 1000

diffusion:
  timesteps: 1000  # T
  beta_schedule: linear
  beta_min: 0.0001
  beta_max: 0.02

guidance:
  p_uncond: 0.1  # Unconditional probability
  omega: 1.0     # Guidance scale
```

### 2.3 Inference Algorithm

#### Noising-Denoising Procedure
```python
def denoise(model, x_0,
            T_prime: int = 500,      # Noising steps
            omega: float = 1.0,       # CFG scale
            n_seeds: int = 5,         # Random seeds
            M: int = 1):              # Corrector steps
    """
    Inference algorithm for denoising

    Args:
        x_0: Noisy input time series [B, L]
        T_prime: Noising level (< T=1000)
        omega: Classifier-free guidance scale
        n_seeds: Number of random seeds for averaging
        M: Number of corrector steps per predictor step

    Returns:
        x_hat: Denoised time series [B, L]
    """
    c = x_0  # Condition is original data
    denoised_list = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)

        # Step 1: Forward noising to T'
        x_T_prime = forward_diffusion(x_0, T_prime)

        # Step 2: Reverse denoising from T' to 0
        x_t = x_T_prime

        for i in range(T_prime, 0, -1):
            t = torch.tensor([i / T]).to(x_0.device)

            # Predictor step (VP-SDE)
            with torch.no_grad():
                # Get conditional and unconditional scores
                score_cond = model(x_t, t, c)
                score_uncond = model(x_t, t, torch.zeros_like(c))

                # Classifier-free guidance
                score = omega * score_cond + (1 - omega) * score_uncond

                # VP-SDE predictor
                beta_t = get_beta(t)
                x_t = (1 / torch.sqrt(1 - beta_t)) * (
                    x_t + beta_t * score
                ) + torch.sqrt(beta_t) * torch.randn_like(x_t)

            # Corrector steps (Annealed Langevin MCMC)
            for _ in range(M):
                with torch.no_grad():
                    score = model(x_t, t, c)
                    epsilon = 2e-5  # Step size
                    x_t = x_t + epsilon * score + torch.sqrt(2 * epsilon) * torch.randn_like(x_t)

            # Guidance by TV Loss
            x_t = x_t - eta_tv * compute_tv_gradient(x_t)

            # Guidance by Fourier Loss
            x_t = x_t - eta_f * compute_fourier_gradient(x_t, c)

        denoised_list.append(x_t)

    # Average over multiple seeds
    x_hat = torch.stack(denoised_list).mean(dim=0)

    return x_hat
```

#### Auxiliary Loss Functions
```python
def compute_tv_loss(x):
    """Total Variation Loss for smoothness"""
    return torch.abs(x[:, 1:] - x[:, :-1]).sum()

def compute_tv_gradient(x, eta=0.01):
    """Gradient of TV loss"""
    grad = torch.zeros_like(x)
    grad[:, :-1] = torch.sign(x[:, 1:] - x[:, :-1])
    grad[:, 1:] -= torch.sign(x[:, 1:] - x[:, :-1])
    return eta * grad

def compute_fourier_loss(x_t, x_0, threshold=0.1):
    """Fourier Loss for frequency preservation"""
    fft_xt = torch.fft.rfft(x_t, dim=-1)
    fft_x0 = torch.fft.rfft(x_0, dim=-1)

    # Filter low-amplitude frequencies
    mask = torch.abs(fft_x0) > threshold
    fft_x0_filtered = fft_x0 * mask

    loss = F.mse_loss(fft_xt, fft_x0_filtered)
    return loss

def compute_fourier_gradient(x_t, x_0, eta=0.01):
    """Gradient of Fourier loss (simplified)"""
    fft_xt = torch.fft.rfft(x_t, dim=-1)
    fft_x0 = torch.fft.rfft(x_0, dim=-1)
    mask = torch.abs(fft_x0) > 0.1

    grad_fft = (fft_xt - fft_x0 * mask)
    grad = torch.fft.irfft(grad_fft, n=x_t.shape[-1])

    return eta * grad
```

#### Inference Configuration
```yaml
inference:
  T_prime: 500           # Noising level (50% of max)
  omega: 1.0             # CFG scale
  n_seeds: 5             # Random seeds for averaging
  corrector_steps: 1     # Langevin MCMC steps

guidance:
  eta_tv: 0.01          # TV loss step size
  eta_fourier: 0.01     # Fourier loss step size
  fourier_threshold: 0.1  # Frequency filtering
```

---

## 3. Stage 2: TRM Recursive Reasoning Model

### 3.1 Model Architecture

#### Core Components
```
Input: x_denoised ∈ R^L (denoised time series)
Output: y ∈ R (predicted return / action)

TRM Architecture:
  ├── Input Embedding: x → e_x ∈ R^d
  ├── Answer Embedding: y → e_y ∈ R^d
  ├── Latent State: z ∈ R^d
  │
  ├── High-Level Loop (K iterations):
  │   ├── Low-Level Loop (n iterations):
  │   │   └── z' = RecursiveBlock(x, y, z)
  │   │       ├── Attention(Q=z, K=V=[x,y,z])
  │   │       ├── MLP(z')
  │   │       └── Residual + LayerNorm
  │   └── y' = UpdateAnswer(y, z)
  │       ├── CrossAttention(Q=y, K=V=z)
  │       └── MLP(y')
  └── Output Projection: y → prediction
```

#### Network Specifications
```python
class TRMRecursiveReasoning(nn.Module):
    def __init__(self,
                 input_dim: int = 60,        # Sequence length
                 hidden_dim: int = 256,      # Model dimension
                 n_high_cycles: int = 3,     # K (high-level iterations)
                 n_low_cycles: int = 4,      # n (low-level iterations)
                 n_layers: int = 2,          # Transformer layers
                 n_heads: int = 8,           # Attention heads
                 dropout: float = 0.1):
        super().__init__()

        self.n_high_cycles = n_high_cycles
        self.n_low_cycles = n_low_cycles

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Initial answer embedding
        self.answer_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # Initial latent embedding
        self.latent_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # Recursive reasoning blocks (low-level)
        self.recursive_blocks = nn.ModuleList([
            RecursiveReasoningBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Answer update block (high-level)
        self.answer_update = AnswerUpdateBlock(hidden_dim, n_heads, dropout)

        # Output projection
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification or regression
        )

class RecursiveReasoningBlock(nn.Module):
    """Low-level recursive reasoning: updates latent z"""
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        """
        Args:
            x: input embedding [B, d]
            y: answer embedding [B, d]
            z: latent embedding [B, d]
        Returns:
            z': updated latent [B, d]
        """
        # Concatenate [x, y, z] as context
        context = torch.stack([x, y, z], dim=1)  # [B, 3, d]

        # Self-attention on z with context
        z_query = z.unsqueeze(1)  # [B, 1, d]
        attn_out, _ = self.attention(z_query, context, context)
        z = self.norm1(z + self.dropout(attn_out.squeeze(1)))

        # Feed-forward
        z = self.norm2(z + self.dropout(self.ffn(z)))

        return z

class AnswerUpdateBlock(nn.Module):
    """High-level answer update: improves answer y given latent z"""
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, z):
        """
        Args:
            y: answer embedding [B, d]
            z: latent embedding [B, d]
        Returns:
            y': updated answer [B, d]
        """
        y_query = y.unsqueeze(1)  # [B, 1, d]
        z_context = z.unsqueeze(1)  # [B, 1, d]

        # Cross-attention: y attends to z
        attn_out, _ = self.cross_attention(y_query, z_context, z_context)
        y = self.norm1(y + self.dropout(attn_out.squeeze(1)))

        # Feed-forward
        y = self.norm2(y + self.dropout(self.ffn(y)))

        return y
```

### 3.2 Forward Pass

```python
def forward(self, x_denoised):
    """
    TRM forward pass with recursive reasoning

    Args:
        x_denoised: Denoised time series [B, L]

    Returns:
        prediction: Output [B, 1]
    """
    batch_size = x_denoised.shape[0]

    # Embed input
    x_embed = self.input_embedding(x_denoised)  # [B, d]

    # Initialize answer and latent
    y = self.answer_embedding.expand(batch_size, -1)  # [B, d]
    z = self.latent_embedding.expand(batch_size, -1)  # [B, d]

    # High-level loop: K iterations
    for k in range(self.n_high_cycles):

        # Low-level loop: n iterations (recursive reasoning)
        for n in range(self.n_low_cycles):
            for block in self.recursive_blocks:
                z = block(x_embed, y, z)

        # Update answer based on refined latent
        y = self.answer_update(y, z)

    # Final prediction
    prediction = self.output_head(y)  # [B, 1]

    return prediction
```

### 3.3 Training Configuration

#### Loss Function
```python
def compute_loss(model, x_denoised, target, task='classification'):
    """
    Args:
        x_denoised: Denoised input [B, L]
        target: Ground truth [B, 1]
        task: 'classification' or 'regression'
    """
    prediction = model(x_denoised)

    if task == 'classification':
        # Binary classification (return sign)
        loss = F.binary_cross_entropy_with_logits(prediction, target)
    else:
        # Regression (return value)
        loss = F.huber_loss(prediction, target, delta=1.0)

    return loss
```

#### Training Setup
```yaml
model:
  hidden_dim: 256
  n_high_cycles: 3     # K iterations
  n_low_cycles: 4      # n iterations per K
  n_layers: 2          # Recursive blocks
  n_heads: 8
  dropout: 0.1

optimizer:
  type: Adam-atan2     # Specialized optimizer
  lr: 1e-4
  weight_decay: 1.0

training:
  epochs: 50000
  batch_size: 256
  eval_interval: 5000
  early_stopping: True
  patience: 10

task:
  type: classification  # or 'regression'
  prediction_horizons: [1, 5, 10]  # timesteps

ema:
  enabled: True
  decay: 0.999
```

---

## 4. Two-Stage Pipeline Integration

### 4.1 Training Pipeline

```python
class TRMDiffusionPipeline:
    def __init__(self, config):
        # Stage 1: Diffusion denoiser
        self.denoiser = ConditionalDiffusionDenoiser(**config.denoiser)

        # Stage 2: TRM predictor
        self.trm = TRMRecursiveReasoning(**config.trm)

        # Optimizers
        self.opt_denoiser = AdamW(self.denoiser.parameters(), **config.opt_denoiser)
        self.opt_trm = AdamAtan2(self.trm.parameters(), **config.opt_trm)

    def train_stage1(self, dataloader, epochs):
        """Train diffusion denoiser"""
        self.denoiser.train()

        for epoch in range(epochs):
            for batch in dataloader:
                x_noisy = batch['close_prices']  # [B, 60]

                # Forward pass
                loss = train_step_with_cfg(self.denoiser, x_noisy)

                # Backward pass
                self.opt_denoiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                self.opt_denoiser.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def train_stage2(self, dataloader, epochs):
        """Train TRM with frozen denoiser"""
        self.denoiser.eval()  # Freeze denoiser
        self.trm.train()

        for epoch in range(epochs):
            for batch in dataloader:
                x_noisy = batch['close_prices']  # [B, 60]
                target = batch['returns']         # [B, 1]

                # Stage 1: Denoise (no grad)
                with torch.no_grad():
                    x_denoised = denoise(self.denoiser, x_noisy)

                # Stage 2: TRM prediction
                prediction = self.trm(x_denoised)
                loss = compute_loss(self.trm, x_denoised, target)

                # Backward pass
                self.opt_trm.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trm.parameters(), 1.0)
                self.opt_trm.step()

            if epoch % 5000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def train_end_to_end(self, dataloader, epochs, freeze_denoiser=False):
        """Joint fine-tuning (optional)"""
        if not freeze_denoiser:
            self.denoiser.train()
        else:
            self.denoiser.eval()

        self.trm.train()

        for epoch in range(epochs):
            for batch in dataloader:
                x_noisy = batch['close_prices']
                target = batch['returns']

                # Stage 1: Denoise
                if freeze_denoiser:
                    with torch.no_grad():
                        x_denoised = denoise(self.denoiser, x_noisy)
                else:
                    x_denoised = denoise(self.denoiser, x_noisy)

                # Stage 2: Predict
                prediction = self.trm(x_denoised)
                loss = compute_loss(self.trm, x_denoised, target)

                # Backward pass (both stages if not frozen)
                self.opt_trm.zero_grad()
                if not freeze_denoiser:
                    self.opt_denoiser.zero_grad()

                loss.backward()

                if not freeze_denoiser:
                    torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                    self.opt_denoiser.step()

                torch.nn.utils.clip_grad_norm_(self.trm.parameters(), 1.0)
                self.opt_trm.step()
```

### 4.2 Inference Pipeline

```python
def predict(self, x_noisy, return_denoised=False):
    """
    End-to-end prediction

    Args:
        x_noisy: Raw noisy time series [B, L]
        return_denoised: Whether to return denoised series

    Returns:
        prediction: Model output [B, 1]
        (optional) x_denoised: Denoised series [B, L]
    """
    self.denoiser.eval()
    self.trm.eval()

    with torch.no_grad():
        # Stage 1: Denoise
        x_denoised = denoise(self.denoiser, x_noisy,
                            T_prime=500, omega=1.0, n_seeds=5)

        # Stage 2: Predict
        prediction = self.trm(x_denoised)

    if return_denoised:
        return prediction, x_denoised
    return prediction
```

---

## 5. Data Pipeline

### 5.1 Dataset Structure

```python
class FinancialTimeSeriesDataset(Dataset):
    """
    Dataset for financial time series with rolling window
    """
    def __init__(self,
                 csv_path: str,
                 window_size: int = 60,
                 stride: int = 20,
                 prediction_horizon: int = 1,
                 task: str = 'classification'):
        """
        Args:
            csv_path: Path to CSV file (columns: date, open, high, low, close, volume)
            window_size: Length of input sequence
            stride: Rolling window stride
            prediction_horizon: Steps ahead to predict (1, 5, or 10)
            task: 'classification' (return sign) or 'regression' (return value)
        """
        self.df = pd.read_csv(csv_path)
        self.window_size = window_size
        self.stride = stride
        self.horizon = prediction_horizon
        self.task = task

        # Extract close prices
        self.prices = self.df['close'].values

        # Generate windows
        self.windows = []
        self.targets = []

        for i in range(0, len(self.prices) - window_size - horizon, stride):
            # Input window
            window = self.prices[i:i+window_size]

            # Target (log return)
            future_price = self.prices[i+window_size+horizon-1]
            current_price = self.prices[i+window_size-1]
            log_return = np.log(future_price / current_price)

            if task == 'classification':
                # Binary: 1 if positive return, 0 otherwise
                target = 1.0 if log_return > 0 else 0.0
            else:
                # Regression: actual log return
                target = log_return

            self.windows.append(window)
            self.targets.append(target)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        # Normalize windows (z-score per window)
        self.windows = self._normalize(self.windows)

    def _normalize(self, windows):
        """Per-window z-score normalization"""
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True) + 1e-8
        return (windows - mean) / std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            'close_prices': torch.FloatTensor(self.windows[idx]),
            'returns': torch.FloatTensor([self.targets[idx]])
        }
```

### 5.2 Data Loading

```python
def create_dataloaders(config):
    """
    Create train/val/test dataloaders
    """
    # Load datasets
    train_dataset = FinancialTimeSeriesDataset(
        csv_path='train.csv',
        window_size=60,
        stride=20,
        prediction_horizon=config.horizon,
        task=config.task
    )

    val_dataset = FinancialTimeSeriesDataset(
        csv_path='val.csv',
        window_size=60,
        stride=20,
        prediction_horizon=config.horizon,
        task=config.task
    )

    test_dataset = FinancialTimeSeriesDataset(
        csv_path='test.csv',
        window_size=60,
        stride=20,
        prediction_horizon=config.horizon,
        task=config.task
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
```

---

## 6. Evaluation Metrics

### 6.1 Classification Metrics

```python
def evaluate_classification(model, dataloader):
    """
    Evaluate classification performance
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['close_prices'].cuda()
            y = batch['returns'].cuda()

            # Predict
            logits = model.predict(x)
            preds = (torch.sigmoid(logits) > 0.5).float()

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Compute metrics
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()

    # F1 Score
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # MCC (Matthews Correlation Coefficient)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return {
        'f1': f1,
        'mcc': mcc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
```

### 6.2 Trading Metrics

```python
def evaluate_trading(model, dataloader, strategy='macd'):
    """
    Evaluate trading performance

    Args:
        strategy: 'macd', 'bollinger', 'following', or 'countering'
    """
    model.eval()

    # Get predictions and denoised series
    all_denoised = []
    all_original = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['close_prices'].cuda()
            _, x_denoised = model.predict(x, return_denoised=True)

            all_denoised.append(x_denoised.cpu().numpy())
            all_original.append(x.cpu().numpy())

    denoised = np.concatenate(all_denoised)
    original = np.concatenate(all_original)

    # Generate trading signals
    if strategy == 'macd':
        signals = generate_macd_signals(denoised)
    elif strategy == 'bollinger':
        signals = generate_bollinger_signals(denoised)
    elif strategy == 'following':
        signals = generate_following_signals(model, denoised)
    elif strategy == 'countering':
        signals = generate_countering_signals(model, denoised)

    # Compute returns
    returns = compute_returns(original, signals)

    # Metrics
    lor = compute_long_only_return(returns, signals)
    lsr = compute_long_short_return(returns, signals)
    num_trades = (signals != 0).sum()

    return {
        'long_only_return': lor,
        'long_short_return': lsr,
        'num_trades': num_trades,
        'avg_return_per_trade': lor / (num_trades + 1e-8)
    }

def generate_macd_signals(prices, fast=12, slow=26, signal=9):
    """MACD trading signals"""
    ema_fast = pd.Series(prices[:, -1]).ewm(span=fast).mean()
    ema_slow = pd.Series(prices[:, -1]).ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()

    signals = np.zeros(len(prices))
    signals[macd > signal_line] = 1   # Buy
    signals[macd < signal_line] = -1  # Sell

    return signals

def generate_bollinger_signals(prices, window=20, num_std=2):
    """Bollinger Bands trading signals"""
    series = pd.Series(prices[:, -1])
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    signals = np.zeros(len(prices))
    signals[prices[:, -1] < lower] = 1   # Buy (oversold)
    signals[prices[:, -1] > upper] = -1  # Sell (overbought)

    return signals

def compute_returns(prices, signals):
    """Compute returns based on signals"""
    price_changes = np.diff(prices[:, -1]) / prices[:-1, -1]
    returns = price_changes * signals[:-1]  # Align dimensions
    return returns

def compute_long_only_return(returns, signals):
    """Cumulative return for long positions only"""
    long_returns = returns[signals[:-1] == 1]
    return np.sum(long_returns)

def compute_long_short_return(returns, signals):
    """Cumulative return for long and short positions"""
    return np.sum(returns)
```

---

## 7. Training Scripts

### 7.1 Stage 1: Train Denoiser

```python
# train_denoiser.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

def main(args):
    # Create model
    denoiser = ConditionalDiffusionDenoiser(
        input_dim=1,
        seq_len=60,
        d_model=512,
        n_heads=8,
        n_layers=6
    ).cuda()

    # Create dataset
    train_dataset = FinancialTimeSeriesDataset(
        csv_path=args.train_csv,
        window_size=60,
        stride=20,
        prediction_horizon=1,
        task='classification'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # Training loop
    for epoch in range(args.epochs):
        denoiser.train()
        total_loss = 0

        for batch in train_loader:
            x = batch['close_prices'].cuda()

            # Compute loss
            loss = train_step_with_cfg(denoiser, x, p_uncond=0.1)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

        # Save checkpoint
        if epoch % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader)
            }, f'checkpoints/denoiser_epoch_{epoch}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='train.csv')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    main(args)
```

### 7.2 Stage 2: Train TRM

```python
# train_trm.py

import torch
from torch.utils.data import DataLoader
import argparse

def main(args):
    # Load pretrained denoiser
    denoiser = ConditionalDiffusionDenoiser(...).cuda()
    denoiser.load_state_dict(torch.load(args.denoiser_ckpt)['model_state_dict'])
    denoiser.eval()

    # Create TRM model
    trm = TRMRecursiveReasoning(
        input_dim=60,
        hidden_dim=256,
        n_high_cycles=3,
        n_low_cycles=4,
        n_layers=2,
        n_heads=8
    ).cuda()

    # Dataset
    train_dataset = FinancialTimeSeriesDataset(
        csv_path=args.train_csv,
        window_size=60,
        stride=20,
        prediction_horizon=args.horizon,
        task='classification'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    # Optimizer (adam-atan2)
    optimizer = torch.optim.Adam(
        trm.parameters(),
        lr=1e-4,
        weight_decay=1.0
    )

    # Training loop
    for epoch in range(args.epochs):
        trm.train()
        total_loss = 0

        for batch in train_loader:
            x = batch['close_prices'].cuda()
            y = batch['returns'].cuda()

            # Denoise (no grad)
            with torch.no_grad():
                x_denoised = denoise(denoiser, x, T_prime=500, n_seeds=1)

            # Predict
            pred = trm(x_denoised)
            loss = F.binary_cross_entropy_with_logits(pred, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

        # Save checkpoint
        if epoch % 5000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': trm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader)
            }, f'checkpoints/trm_epoch_{epoch}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='train.csv')
    parser.add_argument('--denoiser_ckpt', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--horizon', type=int, default=1)
    args = parser.parse_args()
    main(args)
```

---

## 8. Configuration Files

### 8.1 Hydra Config Structure

```yaml
# config/config.yaml

defaults:
  - denoiser: diffusion_vp
  - trm: recursive
  - dataset: sp500_daily
  - _self_

# Training
train:
  stage1_epochs: 100
  stage2_epochs: 50000
  batch_size: 128
  num_workers: 4
  device: cuda
  seed: 42

# Paths
paths:
  data_dir: ./data
  checkpoint_dir: ./checkpoints
  log_dir: ./logs

# Evaluation
eval:
  metrics: ['f1', 'mcc', 'accuracy']
  trading_strategies: ['macd', 'bollinger']
  prediction_horizons: [1, 5, 10]
```

```yaml
# config/denoiser/diffusion_vp.yaml

type: vp_sde
architecture:
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  dropout: 0.1
  time_embed_dim: 128

diffusion:
  timesteps: 1000
  beta_schedule: linear
  beta_min: 0.0001
  beta_max: 0.02

training:
  optimizer: adamw
  lr: 1e-4
  weight_decay: 1e-4
  ema_decay: 0.9999

guidance:
  p_uncond: 0.1
  omega: 1.0

inference:
  T_prime: 500
  n_seeds: 5
  corrector_steps: 1
  eta_tv: 0.01
  eta_fourier: 0.01
```

```yaml
# config/trm/recursive.yaml

architecture:
  input_dim: 60
  hidden_dim: 256
  n_high_cycles: 3
  n_low_cycles: 4
  n_layers: 2
  n_heads: 8
  dropout: 0.1

training:
  optimizer: adam
  lr: 1e-4
  weight_decay: 1.0
  ema_decay: 0.999

task:
  type: classification  # or regression
```

```yaml
# config/dataset/sp500_daily.yaml

name: sp500_daily
csv_path: ./data/train.csv
window_size: 60
stride: 20
split_ratio: [0.8, 0.1, 0.1]  # train/val/test
```

---

## 9. Deployment

### 9.1 Inference API

```python
# inference.py

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load models
denoiser = torch.load('checkpoints/denoiser_final.pt').cuda()
trm = torch.load('checkpoints/trm_final.pt').cuda()
denoiser.eval()
trm.eval()

class PredictionRequest(BaseModel):
    prices: list[float]  # Length 60

class PredictionResponse(BaseModel):
    prediction: float
    denoised_prices: list[float]

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Prepare input
    prices = np.array(request.prices, dtype=np.float32)
    prices = (prices - prices.mean()) / (prices.std() + 1e-8)  # Normalize
    x = torch.FloatTensor(prices).unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        x_denoised = denoise(denoiser, x, T_prime=500, n_seeds=3)
        prediction = trm(x_denoised)
        prediction = torch.sigmoid(prediction).item()

    return PredictionResponse(
        prediction=prediction,
        denoised_prices=x_denoised[0].cpu().numpy().tolist()
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

---

## 10. Expected Performance

### 10.1 Baseline Comparisons

| Method | 1day F1/MCC | 1hour F1/MCC | 5min F1/MCC |
|--------|-------------|--------------|-------------|
| Original | 0.604/0.014 | 0.452/-0.003 | 0.435/0.007 |
| EMA | 0.656/0.120 | 0.558/0.085 | 0.494/0.021 |
| DAE | 0.676/0.183 | 0.563/0.065 | 0.151/0.029 |
| **VP-SDE** | **0.719/0.323** | **0.806/0.329** | **0.798/0.313** |
| **TRM+Diffusion** | **0.75+/0.40+** | **0.85+/0.40+** | **0.80+/0.35+** |

### 10.2 Trading Performance

| Strategy | LOR | LSR | NoT |
|----------|-----|-----|-----|
| MACD (Ori) | -0.6 | -0.6 | 160 |
| MACD (TRM+Diffusion) | **1.2** | **1.2** | **40** |
| Bollinger (Ori) | -0.5 | -0.5 | 54 |
| Bollinger (TRM+Diffusion) | **1.0** | **1.0** | **18** |

---

## 11. Implementation Checklist

- [ ] Stage 1: Diffusion Denoiser
  - [ ] Conditional transformer architecture
  - [ ] VP-SDE training loop
  - [ ] Classifier-free guidance
  - [ ] Inference with predictor-corrector
  - [ ] TV and Fourier loss guidance

- [ ] Stage 2: TRM Model
  - [ ] Recursive reasoning blocks
  - [ ] Answer update mechanism
  - [ ] High/low-level iteration loops
  - [ ] Training with denoised data

- [ ] Data Pipeline
  - [ ] CSV loading and preprocessing
  - [ ] Rolling window generation
  - [ ] Per-window normalization
  - [ ] Train/val/test splits

- [ ] Training Scripts
  - [ ] Stage 1 training script
  - [ ] Stage 2 training script
  - [ ] End-to-end fine-tuning
  - [ ] Checkpointing and logging

- [ ] Evaluation
  - [ ] Classification metrics (F1, MCC)
  - [ ] Trading metrics (LOR, LSR, NoT)
  - [ ] Visualization tools

- [ ] Deployment
  - [ ] FastAPI inference server
  - [ ] Model optimization (ONNX, TorchScript)
  - [ ] Docker containerization

---

**End of Specification**
