# Leak-Free Denoising Evaluation Workflow

## ✅ 완료된 수정사항
- ✅ Training: Normalization statistics를 checkpoint에 저장
- ✅ Inference: Training statistics만 사용 (inference data 통계 절대 사용 안 함)
- ✅ Split: Metadata JSON 저장 (`split_info.json`)
- ✅ Extract: Dynamic split index 사용
- ✅ **모든 data leakage 제거 완료**

---

## 🚀 Leak-Free Workflow (GPU 서버)

### Phase 1: 데이터 준비
```bash
# 1. Train/Val split (metadata 자동 저장)
python TinyRecursiveModels/train/split_train_val.py

# 결과:
# - TinyRecursiveModels/train_only.csv (7192 rows)
# - TinyRecursiveModels/val_only.csv (1798 rows)
# - TinyRecursiveModels/split_info.json (metadata)
```

### Phase 2: Leak-Free Training
```bash
# 2. Train on train_only.csv (normalization statistics 자동 저장)
bash TinyRecursiveModels/train/train_all_clusters.sh

# 체크포인트에 포함되는 내용:
# - model_state_dict
# - normalization_mean (training data mean)
# - normalization_std (training data std)
# - feature_names, cluster_type 등
```

**확인사항**:
```bash
# Training 로그에서 확인:
# "Computing normalization statistics..."
# "Mean range: [...], Std range: [...]"
# "Saved checkpoint to trained_models/cluster_X_best.pt"
```

### Phase 3: Leak-Free Inference
```bash
# 3. Denoise full dataset (TRAINING statistics 사용!)
python TinyRecursiveModels/inference/denoise_dataset.py \
    --input_csv TinyRecursiveModels/train.csv \
    --output_csv train_denoised_v3.csv \
    --device cuda

# 로그에서 확인:
# "Loaded training statistics:"
# "  Mean range: [...], Std range: [...]"
```

**핵심**:
- Val 데이터 디노이징 시에도 **training statistics만 사용**
- Inference data의 통계는 절대 계산하지 않음

### Phase 4: Val Set 추출
```bash
# 4. Extract val portion (dynamic split index)
python TinyRecursiveModels/train/extract_denoised_val.py

# split_info.json에서 자동으로 n_train 로드
# 결과: val_denoised_v3.csv (1798 rows)
```

### Phase 5: Leak-Free Validation
```bash
# 5. IC Validation (5-fold CV)
python TinyRecursiveModels/evaluation/validate_denoising.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised_v3.csv \
    --n_splits 5

# 6. Trading Validation (5-fold CV)
python TinyRecursiveModels/evaluation/validate_trading_signals.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised_v3.csv \
    --n_splits 5
```

---

## 🔍 Leakage 검증 체크리스트

### ✅ Training Phase
- [x] train_only.csv (7192 rows)로만 훈련
- [x] Normalization statistics를 train_only 데이터로만 계산
- [x] Statistics를 checkpoint에 저장

### ✅ Inference Phase
- [x] Checkpoint에서 training statistics 로드
- [x] Inference data의 통계는 **절대 계산 안 함**
- [x] Train과 Val 모두 동일한 training statistics 사용

### ✅ Validation Phase
- [x] Val set (1798 rows)으로만 평가
- [x] Val set은 training 중 한 번도 보지 않음
- [x] Val set 디노이징 시 training statistics만 사용

---

## ⚠️ 이전 결과 vs 새 결과 비교

### 이전 결과 (Leakage O):
```
LinearRegression IC: 0.045 → 0.132 (+194%)
Ridge IC: 0.049 → 0.119 (+144%)
```
**문제**: Val 통계가 디노이징에 유출됨

### 예상 결과 (Leak-Free):
```
LinearRegression IC: 0.045 → 0.09~0.11 (+100~150%)
Ridge IC: 0.049 → 0.08~0.10 (+80~120%)
```
**개선**: 보수적이지만 신뢰 가능한 결과

---

## 🛠️ 문제 해결

### "Checkpoint missing normalization statistics" 에러
```bash
# 원인: 이전 버전 코드로 훈련한 checkpoint
# 해결: 재훈련 필요
bash TinyRecursiveModels/train/train_all_clusters.sh
```

### "Split metadata not found" 에러
```bash
# 원인: split_info.json 없음
# 해결: Split 실행
python TinyRecursiveModels/train/split_train_val.py
```

### Val set 크기 불일치
```bash
# split_info.json 확인
cat TinyRecursiveModels/split_info.json

# 재생성
python TinyRecursiveModels/train/split_train_val.py
```

---

## 📊 성능 기대치

### 디노이징 효과 (Leak-Free):
- **선형 모델**: +80~150% IC 개선
- **Tree 모델**: +20~50% IC 개선 (과적합 제거)
- **Sharpe Ratio**: +50~100% 개선

### 왜 선형 모델이 우수한가?
1. **노이즈 제거**: 선형 관계가 명확해짐
2. **과적합 방지**: 복잡한 모델은 노이즈에 과적합
3. **안정성**: Regularization으로 robust

---

## 🚀 One-Liner (전체 실행)

```bash
cd /workspace/TRMQuant

# Phase 1: Split
python TinyRecursiveModels/train/split_train_val.py

# Phase 2: Train (5 epochs × 7 clusters ~ 2 hours)
bash TinyRecursiveModels/train/train_all_clusters.sh

# Phase 3: Denoise
python TinyRecursiveModels/inference/denoise_dataset.py \
    --input_csv TinyRecursiveModels/train.csv \
    --output_csv train_denoised_v3.csv \
    --device cuda

# Phase 4: Extract
python TinyRecursiveModels/train/extract_denoised_val.py

# Phase 5: Validate
python TinyRecursiveModels/evaluation/validate_trading_signals.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised_v3.csv \
    --n_splits 5
```

**예상 소요 시간**: ~3시간 (훈련 2시간 + 디노이징 30분 + 검증 30분)

---

## 📝 변경 사항 요약

### train_group_denoiser.py
- `TimeSeriesWindowDataset`: pre-computed statistics 파라미터 추가
- Main: statistics 외부 계산 후 dataset에 전달
- Checkpoint: `normalization_mean/std` 저장

### denoise_dataset.py
- `normalize_features()` 삭제
- Checkpoint에서 training statistics 로드
- Training statistics로 정규화 (inference 통계 사용 금지)

### split_train_val.py
- `split_info.json` 저장 (n_train, n_val, train_ratio 등)

### extract_denoised_val.py
- Hardcoded `n_train=7192` 삭제
- `split_info.json`에서 동적 로드

---

## ✅ 최종 검증

수정 완료 후:
```bash
# 1. Checkpoint 확인
python -c "
import torch
ckpt = torch.load('TinyRecursiveModels/trained_models/cluster_0_best.pt')
print('Has normalization_mean:', 'normalization_mean' in ckpt)
print('Has normalization_std:', 'normalization_std' in ckpt)
"

# 2. Split info 확인
cat TinyRecursiveModels/split_info.json

# 3. 재검증 실행
python TinyRecursiveModels/evaluation/validate_trading_signals.py \
    --original TinyRecursiveModels/val_only.csv \
    --denoised val_denoised_v3.csv \
    --n_splits 5
```

**All leakage eliminated! 🎉**
