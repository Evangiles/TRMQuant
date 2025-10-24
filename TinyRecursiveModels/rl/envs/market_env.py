from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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
        self._hist_strategy: list[float] = []
        self._hist_market: list[float] = []

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

            self._hist_strategy.append(strat_ret_t)
            self._hist_market.append(mkt_ret_t)
            W = self.config.vol_window
            if len(self._hist_market) >= W:
                s_port = float(np.std(self._hist_strategy[-W:], ddof=1))
                s_mkt = float(np.std(self._hist_market[-W:], ddof=1))
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


