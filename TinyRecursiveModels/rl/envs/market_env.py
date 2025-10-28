from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from collections import deque

import numpy as np


@dataclass
class MarketEnvConfig:
    window_size: int = 60
    train_fraction: float = 0.8
    # reward shaping
    reward_mode: str = "excess"  # "excess" or "alpha"
    risk_penalty_lambda: float = 0.0
    risk_cap_ratio: float = 1.2
    vol_window: int = 60


class MarketEnv:
    """
    Simple gym-like environment for market allocation on S&P500 excess returns.

    - Observation: last `window_size` days of features, shape (window_size, num_features)
    - Action: allocation a_t in [0, 2]
    - Reward: r_t = a_t * market_forward_excess_returns[t]
    - Done: end of split
    """

    def __init__(
        self,
        features: np.ndarray,
        market_excess_returns: np.ndarray,
        config: Optional[MarketEnvConfig] = None,
        forward_returns: Optional[np.ndarray] = None,
        risk_free_rate: Optional[np.ndarray] = None,
    ) -> None:
        assert features.ndim == 2, "features must be [T, F]"
        assert market_excess_returns.ndim == 1, "market_excess_returns must be [T]"
        assert features.shape[0] == market_excess_returns.shape[0], "T mismatch"

        self.features = features.astype(np.float32)
        self.returns = market_excess_returns.astype(np.float32)  # market excess
        self.config = config or MarketEnvConfig()

        self.T, self.F = self.features.shape

        # optional raw returns for precise vol calc
        self.forward_returns = forward_returns.astype(np.float32) if forward_returns is not None else None
        self.risk_free_rate = risk_free_rate.astype(np.float32) if risk_free_rate is not None else None

        # time split indices
        split_idx = int(self.T * self.config.train_fraction)
        self._splits = {
            "train": (self.config.window_size, split_idx - 1),
            "val": (split_idx, self.T - 1),
        }

        self._t0: int = 0
        self._t: int = 0
        self._t_end: int = 0
        self._hist_strategy: deque[float] = deque(maxlen=self.config.vol_window)
        self._hist_market: deque[float] = deque(maxlen=self.config.vol_window)
        self._sum_s: float = 0.0
        self._sum2_s: float = 0.0
        self._sum_m: float = 0.0
        self._sum2_m: float = 0.0

    @property
    def observation_shape(self) -> Tuple[int, int]:
        return (self.config.window_size, self.F)

    def _obs_at(self, t: int) -> np.ndarray:
        start = t - self.config.window_size
        end = t
        return self.features[start:end]

    def reset(self, split: str = "train") -> np.ndarray:
        assert split in self._splits, f"Unknown split {split}"
        start, end = self._splits[split]
        # episode spans [start, end], first actionable step is at t=start
        self._t0 = start
        self._t = start
        self._t_end = end
        self._hist_strategy.clear()
        self._hist_market.clear()
        self._sum_s = self._sum2_s = 0.0
        self._sum_m = self._sum2_m = 0.0
        return self._obs_at(self._t)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        # clip action to [0, 2]
        a = float(np.clip(action, 0.0, 2.0))

        # base returns at current t
        mkt_excess_t = float(self.returns[self._t])
        strat_excess_t = a * mkt_excess_t
        # alpha or excess reward
        if self.config.reward_mode == "alpha":
            base_reward = strat_excess_t - mkt_excess_t
        else:
            base_reward = strat_excess_t

        # compute rolling vol penalty
        penalty = 0.0
        lam = self.config.risk_penalty_lambda
        if lam > 0.0:
            # prefer non-excess vol if raw returns are provided
            if self.forward_returns is not None and self.risk_free_rate is not None:
                fwd_t = float(self.forward_returns[self._t])
                rf_t = float(self.risk_free_rate[self._t])
                strat_ret_t = rf_t * (1.0 - a) + a * fwd_t
                mkt_ret_t = fwd_t
            else:
                strat_ret_t = strat_excess_t
                mkt_ret_t = mkt_excess_t

            # rolling statistics O(1)
            def _push(val, dq: deque, s_attr: str, s2_attr: str):
                old = None
                if len(dq) == dq.maxlen:
                    old = dq[0]
                dq.append(val)
                # update sums
                s = getattr(self, s_attr)
                s2 = getattr(self, s2_attr)
                if old is not None:
                    s -= old
                    s2 -= old * old
                s += val
                s2 += val * val
                setattr(self, s_attr, s)
                setattr(self, s2_attr, s2)

            _push(strat_ret_t, self._hist_strategy, "_sum_s", "_sum2_s")
            _push(mkt_ret_t, self._hist_market, "_sum_m", "_sum2_m")
            n = len(self._hist_market)
            if n >= self.config.vol_window:
                mean_s = self._sum_s / n
                var_s = max(0.0, self._sum2_s / n - mean_s * mean_s)
                s_port = float(var_s ** 0.5)
                mean_m = self._sum_m / n
                var_m = max(0.0, self._sum2_m / n - mean_m * mean_m)
                s_mkt = float(var_m ** 0.5)
                cap = self.config.risk_cap_ratio
                if s_mkt > 0.0:
                    penalty = lam * max(0.0, s_port - cap * s_mkt)

        r = base_reward - penalty

        # advance time
        self._t += 1
        done = self._t > self._t_end

        if done:
            next_obs = np.zeros(self.observation_shape, dtype=np.float32)
        else:
            next_obs = self._obs_at(self._t)

        info = {"allocation": a, "base_reward": base_reward, "penalty": penalty}
        return next_obs, r, done, info


class VectorizedMarketEnv:
    """
    Vectorized environment wrapper for parallel rollout.

    Runs N independent MarketEnv instances in parallel to improve GPU utilization.

    Args:
        num_envs: Number of parallel environments (default: 8)
        **kwargs: Arguments passed to each MarketEnv instance

    Example:
        >>> envs = VectorizedMarketEnv(num_envs=8, features=X, market_excess_returns=mkt_excess, config=cfg)
        >>> obs = envs.reset("train")  # [N, window_size, F]
        >>> obs, rewards, dones, infos = envs.step(actions)  # actions: [N]
    """

    def __init__(self, num_envs: int = 8, **env_kwargs) -> None:
        self.num_envs = num_envs
        self.envs = [MarketEnv(**env_kwargs) for _ in range(num_envs)]

        # Cache properties from first env
        self.observation_shape = self.envs[0].observation_shape
        self.config = self.envs[0].config
        self._splits = self.envs[0]._splits

    def reset(self, split: str = "train") -> np.ndarray:
        """
        Reset all environments.

        Returns:
            obs: [num_envs, window_size, num_features]
        """
        obs_list = [env.reset(split) for env in self.envs]
        return np.stack(obs_list, axis=0)  # [N, window_size, F]

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step all environments with vectorized actions.

        Args:
            actions: [num_envs] array of actions

        Returns:
            obs: [num_envs, window_size, F]
            rewards: [num_envs]
            dones: [num_envs]
            infos: list of dicts, length num_envs
        """
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"

        results = [env.step(float(a)) for env, a in zip(self.envs, actions)]

        obs = np.stack([r[0] for r in results], axis=0)      # [N, window_size, F]
        rewards = np.array([r[1] for r in results], dtype=np.float32)  # [N]
        dones = np.array([r[2] for r in results], dtype=bool)          # [N]
        infos = [r[3] for r in results]                      # list of dicts

        return obs, rewards, dones, infos


