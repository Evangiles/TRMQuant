from __future__ import annotations

from typing import List
import numpy as np
import torch

from rl.envs.market_env import MarketEnv, MarketEnvConfig
from models.recursive_reasoning.trm_ppo import TRMPPOTemporalEncoder


def streaming_infer(policy: TRMPPOTemporalEncoder, X: np.ndarray, window_size: int) -> List[float]:
    env = MarketEnv(features=X, market_excess_returns=np.zeros(len(X), dtype=np.float32), config=MarketEnvConfig(window_size=window_size, train_fraction=1.0))
    obs = env.reset(split="train")
    done = False
    carry = policy.initial_carry(batch_size=1)
    weights: List[float] = []
    device = next(policy.parameters()).device
    while not done:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            carry, action, _, _ = policy.act(carry, obs_t, deterministic=True)
        a = float(action.squeeze(0).cpu().numpy())
        weights.append(a)
        obs, _, done, _ = env.step(a)
    return weights


