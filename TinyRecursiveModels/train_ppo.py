import os
import math
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rl.data.preprocess import load_market_csv, impute_forward_back_fill, fit_standardizer
from rl.envs.market_env import MarketEnv, MarketEnvConfig
from rl.ppo import RolloutBuffer, PPOConfig, ppo_update_seq
from rl.metrics import adjusted_sharpe
from rl.utils.ema import EMAHelper
from models.recursive_reasoning.trm_ppo import TRMPPOTemporalEncoder, TRMPPOConfig


def main():
    csv_path = os.path.join("TinyRecursiveModels", "train.csv")
    date_id, X, fwd, rf, mkt_excess, feat_names = load_market_csv(csv_path)

    # impute
    X = impute_forward_back_fill(X)
    # standardize on train split only
    split_idx = int(len(X) * 0.8)
    stdzr = fit_standardizer(X[:split_idx])
    X = stdzr.transform(X)

    env = MarketEnv(
        features=X,
        market_excess_returns=mkt_excess,
        config=MarketEnvConfig(
            window_size=60,
            train_fraction=0.8,
            reward_mode="alpha",              # use alpha-style reward
            risk_penalty_lambda=0.0,          # set >0.0 to activate penalty
            risk_cap_ratio=1.2,
            vol_window=60,
        ),
        forward_returns=fwd,
        risk_free_rate=rf,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = TRMPPOTemporalEncoder(
        TRMPPOConfig(
            window_size=env.config.window_size,
            num_features=X.shape[1],
            hidden_size=512,
            num_heads=8,
            expansion=4.0,
            pos_encodings="rope",
            H_cycles=3,
            L_cycles=6,
            L_layers=2,
        )
    ).to(device)

    cfg = PPOConfig(tbptt_len=32)
    optimizer = Adam(policy.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=cfg.lr * 0.1)

    # EMA toggles via env vars or constants
    USE_EMA = bool(int(os.environ.get("USE_EMA", "1")))
    EMA_MU = float(os.environ.get("EMA_MU", "0.999"))
    ema = EMAHelper(mu=EMA_MU) if USE_EMA else None
    if ema is not None:
        ema.register(policy)
    buffer = RolloutBuffer(obs_shape=env.observation_shape, capacity=cfg.rollout_steps)

    # training loop (single-episode per split)
    NUM_EPOCHS = 50
    best_val = -1e9
    best_path = os.path.join("TinyRecursiveModels", "checkpoints_trmppo.pt")

    for epoch in range(NUM_EPOCHS):
        for split in ("train", "val"):
            obs = env.reset(split=split)
            done = False
            carry = policy.initial_carry(batch_size=1)

            step = 0
            allocations = []
            fwd_list = []
            rf_list = []
            buffer.ptr = 0
            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    carry, action, logprob, value = policy.act(carry, obs_t, deterministic=False)
                a = float(action.squeeze(0).cpu().numpy())

                next_obs, reward, done, info = env.step(a)
                allocations.append(a)
                # align fwd/rf at current t
                t_idx = env._t - 1
                fwd_list.append(float(fwd[t_idx]))
                rf_list.append(float(rf[t_idx]))

                buffer.add(
                    torch.from_numpy(obs),
                    torch.tensor(a, dtype=torch.float32),
                    torch.tensor(reward, dtype=torch.float32),
                    torch.tensor(done, dtype=torch.bool),
                    value.squeeze(0).cpu(),
                    logprob.squeeze(0).cpu(),
                )

                obs = next_obs
                step += 1

                if split == "train" and (buffer.ptr >= cfg.rollout_steps or done):
                    # bootstrap with last value
                    if done:
                        last_value = 0.0
                    else:
                        with torch.no_grad():
                            _, _, _, last_value_t = policy.act(carry, torch.from_numpy(obs).unsqueeze(0).to(device), deterministic=True)
                            last_value = float(last_value_t.squeeze(0).cpu())
                    buffer.compute_gae(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda)
                    ppo_update_seq(policy, optimizer, buffer, cfg)
                    if ema is not None:
                        ema.update(policy)
                    buffer.ptr = 0

            # Evaluate adjusted sharpe at split end
            try:
                import pandas as pd
                df = pd.DataFrame({
                    'forward_returns': np.array(fwd_list, dtype=np.float64),
                    'risk_free_rate': np.array(rf_list, dtype=np.float64),
                })
                # evaluate with EMA copy if enabled on val split
                eval_model = ema.ema_copy(policy) if (ema is not None and split == "val") else policy
                score = adjusted_sharpe(df, np.array(allocations, dtype=np.float64))
                print(f"Epoch {epoch+1} Split {split} steps={step} AdjustedSharpe={score:.6f}")
                if split == "val" and score > best_val:
                    best_val = score
                    torch.save({'model': (eval_model.state_dict()), 'score': best_val}, best_path)
                    print(f"Saved best checkpoint to {best_path} (score={best_val:.6f})")
            except Exception as e:
                print(f"Metric computation failed ({split}): {e}")

        scheduler.step()


if __name__ == "__main__":
    main()


