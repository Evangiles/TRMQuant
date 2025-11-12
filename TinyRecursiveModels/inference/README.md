# Inference Pipeline

Denoise complete datasets using trained group-specific Mamba models.

## Quick Start

### 1. Denoise Entire Dataset

```bash
python TinyRecursiveModels/inference/denoise_dataset.py \
    --input_csv TinyRecursiveModels/train.csv \
    --output_csv TinyRecursiveModels/train_denoised.csv \
    --models_dir TinyRecursiveModels/trained_models \
    --stride 30
```

**Parameters**:
- `--input_csv`: Input data (e.g., train.csv)
- `--output_csv`: Output denoised data
- `--models_dir`: Directory with trained model checkpoints
- `--window_size`: Window size (default: 60, must match training)
- `--stride`: Overlap stride (default: 30, smaller = smoother)
- `--num_steps`: Denoising steps (default: 50)
- `--batch_size`: Inference batch size (default: 64)

**Expected Time**: ~1 hour for 8,990 rows

---

### 2. Visualize Results

```bash
python TinyRecursiveModels/inference/visualize_denoising.py \
    --original TinyRecursiveModels/train.csv \
    --denoised TinyRecursiveModels/train_denoised.csv \
    --output_dir TinyRecursiveModels/visualizations \
    --n_samples 3
```

**Outputs**:
- Time series comparison plots
- Distribution comparisons
- Metrics summary JSON

---

## How It Works

### 1. Overlap Averaging

```
Original:  |----window1----|
                  |----window2----|
                        |----window3----|

Denoised:  |----averaged from all overlapping windows----|
```

- Creates overlapping windows (stride < window_size)
- Denoises each window independently
- Averages overlapping regions for smooth reconstruction

### 2. Normalization

```python
# Same normalization as training
normalized = (data - mean) / std
denoised = model(normalized)
reconstructed = denoised * std + mean
```

### 3. Cluster-wise Processing

Each cluster's features are:
1. Extracted from dataset
2. Normalized
3. Split into overlapping windows
4. Denoised with cluster-specific model
5. Reconstructed with overlap averaging
6. Denormalized
7. Written back to dataset

---

## Metrics

### Total Variation (TV) Reduction
```
TV = Σ |x[t+1] - x[t]|
TV_reduction = (TV_original - TV_denoised) / TV_original × 100%
```
**Good**: 20-40% reduction (smoother)
**Bad**: <10% (no effect) or >60% (over-smoothed)

### Correlation
```
corr = corrcoef(original, denoised)
```
**Good**: >0.95 (signal preserved)
**Bad**: <0.90 (signal distorted)

### Signal-to-Noise Ratio (SNR)
```
SNR = 10 × log10(signal_power / noise_power)
```
**Good**: >10 dB (clean)
**Bad**: <5 dB (noisy)

---

## Expected Results

### Mean-Reverting Clusters (0, 1, 2, 3, 6)
- **TV reduction**: 30-50% (spikes smoothed)
- **Correlation**: 0.95-0.98
- **Visual**: Smoother, less jumpy

### Trending Cluster (4)
- **TV reduction**: 10-20% (trends preserved)
- **Correlation**: 0.98-0.99
- **Visual**: High-freq noise removed, trends intact

### Random Walk Cluster (5)
- **TV reduction**: 15-25% (balanced)
- **Correlation**: 0.96-0.98
- **Visual**: Moderate smoothing

---

## Troubleshooting

### Issue: Output same as input
**Cause**: Models not trained or denoising steps too low
**Fix**: Train models longer or increase `--num_steps`

### Issue: Over-smoothed (correlation <0.90)
**Cause**: Too many denoising steps
**Fix**: Reduce `--num_steps` to 20-30

### Issue: Still noisy (TV reduction <10%)
**Cause**: Model undertrained
**Fix**: Train for more epochs (50-100)

### Issue: Slow inference
**Cause**: Small stride or large batch size
**Fix**: Increase stride to 40-50 or reduce batch size

---

## Advanced Usage

### Custom Stride for Different Smoothness

```bash
# Smoother (more overlap)
--stride 15  # Heavy averaging

# Faster (less overlap)
--stride 45  # Lighter averaging
```

### Selective Denoising (Only Some Clusters)

Edit `denoise_dataset.py`:
```python
# Only denoise clusters 0, 1, 2
for cluster_id in [0, 1, 2]:  # Instead of range(n_clusters)
    ...
```

---

## File Structure

```
trained_models/
  cluster_0_best.pt  # 38 features, 1.1M params
  cluster_1_best.pt  # 16 features
  cluster_2_best.pt  # 20 features
  ...

train.csv             # Original data
train_denoised.csv    # Denoised data

visualizations/
  cluster_0_E13.png
  cluster_0_D4.png
  ...
  metrics_summary.json
```

---

## Performance

### Inference Speed (A100 GPU)
- **Window creation**: 1 sec
- **Denoising**: 30-40 sec per cluster
- **Reconstruction**: 2 sec per cluster
- **Total**: ~5-10 minutes for entire dataset

### Memory Usage
- **Peak**: ~2GB GPU memory (cluster 0, batch_size=64)
- **Disk**: Same as input (~50MB for train.csv)
