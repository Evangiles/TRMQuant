# Installation Guide

## Requirements

- Python >= 3.10
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 8GB+ VRAM (for training)

---

## Installation Methods

### Method 1: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Or manually add packages
uv add torch numpy pandas scikit-learn xgboost lightgbm catboost mamba-ssm causal-conv1d tqdm einops
```

### Method 2: Using pip + requirements.txt

```bash
# Install dependencies
pip install -r requirements.txt

# For CUDA support, install PyTorch separately:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Method 3: Using pip + pyproject.toml

```bash
pip install -e .
```

---

## Verify Installation

```bash
# Test PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test Mamba SSM
python -c "from mamba_ssm import Mamba; print('Mamba SSM: OK')"

# Test ML libraries
python -c "import xgboost, lightgbm, catboost; print('ML libraries: OK')"

# Test scikit-learn
python -c "from sklearn.linear_model import Ridge; print('scikit-learn: OK')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
Mamba SSM: OK
ML libraries: OK
scikit-learn: OK
```

---

## GPU Setup (CUDA)

### Check CUDA version
```bash
nvidia-smi
nvcc --version
```

### Install PyTorch with correct CUDA version

For CUDA 11.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Troubleshooting

### Issue: "No module named 'sklearn'"
```bash
# sklearn is deprecated, use scikit-learn
uv add scikit-learn
# or
pip install scikit-learn
```

### Issue: "Mamba SSM import error"
```bash
# Install build dependencies
uv add causal-conv1d mamba-ssm
# or
pip install causal-conv1d mamba-ssm
```

### Issue: "CUDA out of memory"
```bash
# Reduce batch size in training
--batch_size 32  # or lower

# Use gradient accumulation
--accumulation_steps 2
```

### Issue: "CatBoost not found"
```bash
# CatBoost is optional
uv add catboost
# or
pip install catboost
```

---

## Development Setup

```bash
# Clone repository
git clone https://github.com/Evangiles/TRMQuant.git
cd TRMQuant

# Install dependencies
uv sync

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Docker Setup (Optional)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project
COPY . /workspace
WORKDIR /workspace

# Run
CMD ["python3", "TinyRecursiveModels/train/split_train_val.py"]
```

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# 3. Start training
cd TRMwithQuant
python TinyRecursiveModels/train/split_train_val.py
bash TinyRecursiveModels/train/train_all_clusters.sh
```

---

## Dependencies Overview

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| numpy | >=1.24.0 | Numerical computing |
| pandas | >=2.0.0 | Data manipulation |
| scikit-learn | >=1.3.0 | ML models (Ridge, LinearRegression) |
| xgboost | >=2.0.0 | Gradient boosting |
| lightgbm | >=4.0.0 | Gradient boosting |
| catboost | >=1.2.0 | Gradient boosting (optional) |
| mamba-ssm | >=1.0.0 | State space models |
| causal-conv1d | >=1.0.0 | Mamba dependency |
| tqdm | >=4.65.0 | Progress bars |
| einops | >=0.7.0 | Tensor operations |

---

## Next Steps

After installation, follow the [WORKFLOW.md](TinyRecursiveModels/WORKFLOW.md) for leak-free training and evaluation.
