# Feature Engineering for S&P 500 Prediction

**목적**: 피처 엔지니어링 효과를 ML 모델로 빠르게 검증하여 TRM 모델 개선 가이드 제공

## 프로젝트 구조

```
FeatureEngineering/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 전처리 완료 데이터
│   └── features/         # 피처 메타데이터
├── src/
│   ├── preprocessing.py      # 전처리 파이프라인
│   ├── feature_engineering.py # 피처 생성
│   ├── validation.py         # 누수 검증, IC 계산
│   └── models/              # ML 모델들
├── scripts/             # 실행 스크립트
├── configs/             # 설정 파일
└── results/             # 결과 저장
```

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 베이스라인 생성
python scripts/create_baseline_features.py

# 3. 모델 훈련
python scripts/train_all_models.py --data baseline --models lightgbm

# 4. Phase 1 피처 생성
python scripts/create_phase1_features.py

# 5. 비교 평가
python scripts/evaluate_models.py --compare baseline,phase1
```

## Phase 진행

- **Baseline**: 원본 피처 + 기본 전처리
- **Phase 1**: 12개 핵심 피처 (모멘텀, 변동성, 평균회귀)
- **Phase 2**: +가족별 PCA (차원 축소)
- **Phase 3**: +안정성 필터 (RankIC 기반)

## 주요 메트릭

- **Sharpe Ratio**: 위험 조정 수익률
- **Information Coefficient (IC)**: 예측력
- **Max Drawdown**: 최대 손실
- **Hit Rate**: 방향 정확도
- **Turnover**: 거래 비용 추정

## TRM 모델 적용 가이드

1. Phase3에서 가장 높은 IC를 보이는 피처 선택
2. LightGBM feature importance 상위 20개 피처 우선
3. TRM 입력으로 사용하여 성능 개선 확인
