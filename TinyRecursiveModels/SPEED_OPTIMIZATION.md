# PPO 훈련 속도 최적화 가이드

## 현재 상황

**관찰된 문제:**
- VRAM 사용량: 33% (충분한 여유)
- GPU Utilization: 40% (낮음)
- 훈련 속도: ~60 it/s

**근본 원인:**
```python
# 배치 크기 = 1 (순차 실행)
carry, action, logprob, value = policy.act(carry, obs_t, deterministic=False)
# → GPU는 대부분 시간에 놀고 있음
```

---

## 병목 지점 분석

### 1. 롤아웃 단계 (데이터 수집)

**문제점:**
```python
while not done:  # 7132 스텝
    obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)  # [1, 60, F]
    carry, action, logprob, value = policy.act(carry, obs_t)  # batch=1
    a = float(action.squeeze(0).cpu().numpy())  # ← CPU-GPU 동기화!
    next_obs, reward, done, info = env.step(a)
```

**병목:**
1. 배치 크기 = 1 → GPU underutilized
2. 7132번의 CPU-GPU 동기화
3. Python 루프 오버헤드

**GPU 활용도:** ~10-20%

### 2. PPO 업데이트 단계

**문제점:**
```python
# 시퀀스 길이 = 32, 배치 = ~128 (OK)
for t in range(32):
    carry, logprob, value = policy.act(carry, obs_seq[:, t])  # [128, 60, F]
```

**병목:**
1. 시퀀스 길이 32는 짧음 (메모리는 충분)
2. TBPTT로 인한 순차 실행

**GPU 활용도:** ~60-80% (더 나음)

---

## 해결책 우선순위

### ⭐⭐⭐ 1. 병렬 환경 (가장 효과적)

**개념:**
```python
# 현재: 1개 환경
env = MarketEnv(...)  # batch=1

# 개선: N개 환경 병렬 실행
envs = [MarketEnv(...) for _ in range(N)]  # batch=N
```

**효과:**
- GPU 활용도: 40% → 85%
- 속도 향상: **5-10배**
- VRAM 증가: 33% → 60-70% (여전히 괜찮음)

**구현 난이도:** 중간 (환경 벡터화 필요)

**구현 방법:**

```python
class VectorizedMarketEnv:
    def __init__(self, features, num_envs=8):
        self.num_envs = num_envs
        self.envs = [MarketEnv(...) for _ in range(num_envs)]

    def reset(self, split):
        return np.stack([env.reset(split) for env in self.envs])  # [N, 60, F]

    def step(self, actions):  # actions: [N]
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

# 사용
envs = VectorizedMarketEnv(X, num_envs=8)
obs = envs.reset("train")  # [8, 60, F]

carry = policy.initial_carry(batch_size=8)
carry, actions, logprobs, values = policy.act(carry, obs)  # [8, ...]
```

**권장 num_envs:**
- VRAM 33% 사용 → `num_envs=8` (VRAM ~70%)
- Hidden_size=256 → `num_envs=16` 가능
- Hidden_size=512 → `num_envs=4-8`

---

### ⭐⭐ 2. 더 긴 TBPTT 시퀀스

**현재:**
```python
tbptt_len = 32  # 짧음
```

**개선:**
```python
tbptt_len = 128  # 4배 증가
# 또는
tbptt_len = 256  # 8배 증가
```

**효과:**
- GPU 활용도: +10-15%
- 속도 향상: **1.2-1.5배**
- VRAM 증가: 33% → 45-50%

**장점:**
- 구현 쉬움 (인자 하나만 변경)
- 장기 의존성 학습 개선

**단점:**
- 그래디언트 안정성 약간 감소
- VRAM 증가

**권장 설정:**
```bash
python train_ppo.py --tbptt_len 128  # VRAM 여유 있으면
```

---

### ⭐ 3. Mixed Precision (FP16)

**현재:**
```python
forward_dtype: "bfloat16"  # 이미 적용됨!
```

**개선:**
```python
# PyTorch AMP 사용
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    carry, action, logprob, value = policy.act(carry, obs_t)

# 업데이트 시
with autocast():
    loss = policy_loss + vf_loss - ent_loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**효과:**
- GPU 활용도: +5-10%
- 속도 향상: **1.1-1.3배**
- VRAM 감소: 33% → 25%

**주의:**
- 수치 안정성 확인 필요
- NaN 발생 가능성

---

### ⭐ 4. DataLoader Pin Memory

**현재:**
```python
# 버퍼 데이터가 CPU에 있음
buffer.obs  # CPU tensor
```

**개선:**
```python
# GPU로 전송 시 pin_memory 사용
obs_seq = obs_seq.to(device, non_blocking=True)  # 이미 적용됨!
```

**효과:**
- 속도 향상: **1.05-1.1배** (약간)
- CPU-GPU 전송 오버헤드 감소

---

## 권장 최적화 순서

### Phase 1: 즉시 적용 가능 (난이도 낮음)

```bash
# 1. 더 긴 TBPTT
python train_ppo.py --tbptt_len 128

# 2. 환경 변수 튜닝
USE_EMA=1 EMA_MU=0.999 python train_ppo.py
```

**예상 효과:** 1.2-1.5배 속도 향상

---

### Phase 2: 병렬 환경 구현 (난이도 중간)

**단계:**
1. `VectorizedMarketEnv` 클래스 작성
2. `train_ppo.py` 수정:
   - `batch_size=1` → `batch_size=num_envs`
   - Buffer 로직 조정
3. 테스트 및 검증

**예상 효과:** 5-10배 속도 향상 (가장 큼!)

---

### Phase 3: 고급 최적화 (난이도 높음)

1. **Mixed Precision Training**
   - `torch.cuda.amp` 통합
   - NaN 체크 및 스케일링

2. **Gradient Accumulation**
   - 더 큰 effective batch size
   - 메모리 효율적

3. **JIT Compilation**
   ```python
   policy = torch.jit.script(policy)
   ```

**예상 효과:** 추가 1.2-1.5배 속도 향상

---

## 최종 예상 성능

| 최적화 단계 | 속도 (it/s) | GPU Util | VRAM | 구현 난이도 |
|------------|------------|----------|------|------------|
| 현재 | 60 | 40% | 33% | - |
| Phase 1 (TBPTT) | 75 | 50% | 40% | ⭐ 쉬움 |
| Phase 2 (병렬 환경) | 450 | 85% | 65% | ⭐⭐ 중간 |
| Phase 3 (고급) | 600 | 90% | 60% | ⭐⭐⭐ 어려움 |

**결론:**
- **Phase 1**: 지금 바로 가능 (5분)
- **Phase 2**: 가장 효과적 (2-3시간 구현)
- **Phase 3**: 선택 사항

---

## Phase 1 즉시 적용

```bash
# 현재 실행 중이면 중단하고 다시 시작
python train_ppo.py \
    --path train.csv \
    --epochs 10 \
    --lr 3e-4 \
    --hidden_size 256 \
    --tbptt_len 128 \  # ← 32에서 128로 증가
    --reward_mode alpha
```

이것만으로도 **1.2-1.5배** 속도 향상이 예상됩니다!

---

## Phase 2 구현 가이드

VectorizedMarketEnv 구현이 필요하면 알려주세요. 전체 코드를 작성해드리겠습니다.
