# Leak-Free Denoising Evaluation Workflow

## 완료된 단계
- ✅ Feature leakage 제거 (94 features)
- ✅ Train/Val split (80%/20% = 7192/1798 rows)
- ✅ Learning rate 최적화 (5e-4)

## 디노이징 훈련 후 실행 순서

### 1️⃣ 전체 데이터셋 디노이징 (train + val)
```bash
# Kaggle에서 실행
python TinyRecursiveModels/inference/denoise_dataset.py \
    --input_csv TinyRecursiveModels/train.csv \
    --output_csv train_denoised_v2.csv \
    --models_dir TinyRecursiveModels/trained_models \
    --device cuda
```

**예상 시간**: ~10-15분 (8990 rows)

---

### 2️⃣ Val set 추출
```bash
# train_denoised_v2.csv에서 마지막 20% 추출
python -c "
import pandas as pd
df = pd.read_csv('train_denoised_v2.csv')
df_val = df.iloc[7192:].copy()
df_val.to_csv('val_denoised.csv', index=False)
print(f'Extracted {len(df_val)} rows to val_denoised.csv')
"
```

---

### 3️⃣ Val set으로 IC 검증
```bash
python TinyRecursiveModels/evaluation/validate_denoising.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised.csv \
    --n_splits 3
```

**검증 내용**:
- Information Coefficient (IC)
- RMSE, R², MSE
- 5개 ML 모델 (LinearRegression, Ridge, XGBoost, LightGBM, CatBoost)
- 3-fold Purged/Embargo CV

---

### 4️⃣ Val set으로 Trading 검증
```bash
python TinyRecursiveModels/evaluation/validate_trading_signals.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised.csv \
    --n_splits 3
```

**검증 내용**:
- Sharpe Ratio
- Cumulative Returns
- Maximum Drawdown (MDD)
- Win Rate
- Profit Factor
- Long top 20% / Short bottom 20% strategy

---

### 5️⃣ 결과 다운로드
```bash
# Kaggle에서 결과 파일 다운로드
# - train_denoised_v2.csv
# - val_denoised.csv
# - TinyRecursiveModels/evaluation_results/*.csv
```

---

## 전체 One-Liner (Kaggle Cell)

```python
# Step 1: Denoise full dataset
!python TinyRecursiveModels/inference/denoise_dataset.py \
    --input_csv TinyRecursiveModels/train.csv \
    --output_csv train_denoised_v2.csv \
    --device cuda

# Step 2: Extract val set
import pandas as pd
df = pd.read_csv('train_denoised_v2.csv')
df_val = df.iloc[7192:].copy()
df_val.to_csv('val_denoised.csv', index=False)
print(f'✓ Extracted {len(df_val)} rows to val_denoised.csv')

# Step 3: IC Validation
!python TinyRecursiveModels/evaluation/validate_denoising.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised.csv \
    --n_splits 3

# Step 4: Trading Validation
!python TinyRecursiveModels/evaluation/validate_trading_signals.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised.csv \
    --n_splits 3
```

---

## 예상 결과

### 이전 (Leakage 있음, 전체 학습):
- IC 개선: +260%
- Sharpe 개선: +149%

### 현재 (Leak-free, 80% 학습):
- IC 개선: +150-200% (예상)
- Sharpe 개선: +80-120% (예상)
- **더 보수적이지만 신뢰할 수 있는 결과**

---

## 문제 발생 시

### Val set이 없다면:
```bash
python TinyRecursiveModels/train/split_train_val.py
```

### 모델 파일이 없다면:
```bash
# 모델 체크포인트 확인
ls -lh TinyRecursiveModels/trained_models/
# cluster_0_best.pt ~ cluster_6_best.pt 있어야 함
```

### Import 에러:
```bash
pip install xgboost lightgbm catboost
```
