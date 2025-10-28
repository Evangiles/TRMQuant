# TRMQuant 프로젝트 진행 현황 및 핸드오프 가이드

이 문서는 현재 코드베이스 구조, 학습 파이프라인, 핵심 설정/명령어, 그리고 다음 단계 권고안을 요약합니다. 새 코딩 에이전트가 이 문서만 보고 즉시 작업을 이어갈 수 있도록 작성되었습니다.

## 리포 구조(핵심)
- 데이터/유틸
  - `TinyRecursiveModels/train.csv` — 시계열 데이터(피처 + 타깃 3종)
  - `TinyRecursiveModels/tools/analyze_csv.py` — 결측/상관 분석
  - `TinyRecursiveModels/tools/check_leakage.py` — 전처리 누수 점검
  - `TinyRecursiveModels/rl/data/preprocess.py` — 로딩/결측/표준화 유틸
- 환경/메트릭/EMA
  - `TinyRecursiveModels/rl/envs/market_env.py` — 마켓 환경(단일/벡터화), 보상 옵션, O(1) 롤링 변동성
  - `TinyRecursiveModels/rl/metrics.py` — Adjusted Sharpe 구현
  - `TinyRecursiveModels/rl/utils/ema.py` — EMA 헬퍼(기본 사용: USE_EMA=1)
- 모델
  - `TinyRecursiveModels/models/recursive_reasoning/trm_supervised_recursive.py` — 지도학습용 TRM(재귀, z_H/z_L)
  - `TinyRecursiveModels/models/recursive_reasoning/trm_ppo.py` — PPO 정책/가치 헤드 포함 재귀 인코더
- PPO 학습
  - `TinyRecursiveModels/rl/ppo.py` — RolloutBuffer(add_batch/clear/is_full), TBPTT, PPO 업데이트
  - `TinyRecursiveModels/train_ppo.py` — 벡터화 롤아웃, TBPTT, EMA/Sharpe 로깅, overshoot 방지
- 지도학습/파인튜닝
  - `TinyRecursiveModels/train_supervised.py` — Z-score 타깃, Huber, EMA 검증
  - `TinyRecursiveModels/train_finetune_custom.py` — Sharpe 정렬 커스텀 손실, EMA, 체크포인트

## 데이터/전처리
- 타깃: `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`.
- 결측 처리
  - 지도/파인튜닝: `--impute ffill|zero_flag` (split-safe). zero_flag는 NaN→0 + `_nan` 플래그 추가(표준화 제외).
  - PPO: `--impute legacy|diffusion`.
- 표준화: 항상 train split 통계로 fit → 전체 transform.
- 누수 점검: `tools/check_leakage.py`로 전처리 파이프라인 검증(현재 조합들 OK).

## 모델 개요
- Supervised(TRMSupervisedRecursive): H/L 순환, H_cycles−1 부트스트랩(no-grad) + 마지막 패스 역전파. 마지막 시점에서 할당 예측(tanh → [0,2]).
- PPO(TRMPPOTemporalEncoder): 재귀 인코더 + 배우(mean/logstd, tanh-squash→[0,2]), 비평가(value). carry(z_H,z_L) 유지. 수치 안정화 클램프 포함.

## 학습 워크플로우
### 지도학습(사전학습)
- 스크립트: `train_supervised.py`
- 타깃: 롤링 z-score 기반 a*=clip(1+k·z,0,2), k=0.5(치환 가능)
- 손실: Huber. EMA로 검증, 최고 점수 체크포인트 `checkpoints_supervised.pt` 저장.
- 예시:
```
python TinyRecursiveModels/train_supervised.py \
  --drop_high_missing 0 --impute zero_flag --epochs 50
```

### 커스텀 손실 파인튜닝(Sharpe 정렬)
- 스크립트: `train_finetune_custom.py`
- Warm-start: `checkpoints_supervised.pt`
- 손실: L = -Sharpe + λ_vol·cap + λ_gap·underperf + λ_turn·turnover (모두 torch 연산)
- 예시:
```
python TinyRecursiveModels/train_finetune_custom.py \
  --drop_high_missing 0 --impute zero_flag --epochs 30 --lr 3e-5 \
  --lambda_vol 1.0 --lambda_gap 0.1 --lambda_turn 0.0 \
  --warm_start TinyRecursiveModels/checkpoints_supervised.pt
```

### PPO
- 스크립트: `train_ppo.py`
- 개선점: 완전 벡터화 롤아웃(배치 act/step), `RolloutBuffer.add_batch`, TBPTT, carry 전달, EMA 검증.
- 보상: `--reward_mode alpha|sharpe` + `--risk_penalty_lambda`(변동성 초과 패널티).
- overshoot 방지: split 내 처리 샘플 카운터로 while 종료, 완료 env는 `finished_mask`로 제외.
- 예시(VRAM 완화):
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python TinyRecursiveModels/train_ppo.py \
  --warm_start TinyRecursiveModels/checkpoints_supervised.pt \
  --hidden_size 128 --num_heads 2 --H_cycles 2 --L_cycles 3 \
  --num_envs 4 --tbptt_len 8 --rollout_steps 256 --epochs 30 --lr 3e-5
```

## 주요 CLI 요약
- `train_ppo.py`
  - `--path` `--impute legacy|diffusion` `--epochs` `--seed` `--lr`
  - `--hidden_size` `--num_heads` `--H_cycles` `--L_cycles`
  - `--warm_start` `--reward_mode alpha|sharpe` `--risk_penalty_lambda`
  - `--tbptt_len` `--num_envs` `--rollout_steps`
- `train_supervised.py` / `train_finetune_custom.py`
  - 공통 전처리: `--drop_high_missing` `--drop_threshold` `--impute ffill|zero_flag`
  - 파인튜닝 추가: `--lambda_vol` `--lambda_gap` `--lambda_turn` `--lr` `--warm_start`

## 성능/메모리 팁
- OOM 시: `--num_envs`, `--tbptt_len`, `--hidden_size`, `--H_cycles`, `--L_cycles`, `--rollout_steps` 축소.
- CUDA 단편화 완화: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- 필요 시 AMP(fp16) 고려. PPO 버퍼 carry half-precision 저장 검토 가능.

## 주의 사항
- Warm start는 `strict=False`로 일부 미스매치 허용. 완전 재사용 원하면 hidden/head/L_layers 맞추기 권장.
- PPO 버퍼 용량: `rollout_steps * num_envs`. 업데이트 트리거: `buffer.is_full()` 또는 에피소드 종료.
- 평가 메트릭: `rl/metrics.py`의 Adjusted Sharpe 사용.

## 권장 기본값(현재)
- Supervised: epochs 50, Huber, EMA on.
- Finetune: epochs 30, lr 3e-5, λ_vol 1.0, λ_gap 0.1, λ_turn 0.0.
- PPO: hidden 128, heads 2, H 2, L 3, num_envs 4, tbptt_len 8, rollout_steps 256.

## 다음 단계 제안
- 지도 타깃 고도화(quantile/리스크 캡/신호 스케일링).
- PPO `reward_mode=sharpe`를 대회 메트릭에 더 정밀 정렬(안정성 유의).
- 행동 공간 안정화: [0.5,1.5] 클립 또는 `{0,0.5,1,1.5,2}` 디스크리트 스냅 실험.
- PPO 버퍼 carry half-precision 저장으로 VRAM 추가 절감.

---
Maintained by: TRMQuant RL/SL pipeline
