from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterator

import numpy as np
import torch
from torch import nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    rollout_steps: int = 4096
    update_epochs: int = 10
    num_minibatches: int = 8
    tbptt_len: int = 32


class RolloutBuffer:
    def __init__(self, obs_shape: Tuple[int, int], capacity: int) -> None:
        T, F = obs_shape
        self.obs = torch.zeros((capacity, T, F), dtype=torch.float32)
        self.actions = torch.zeros((capacity,), dtype=torch.float32)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.bool)
        self.values = torch.zeros((capacity,), dtype=torch.float32)
        self.logprobs = torch.zeros((capacity,), dtype=torch.float32)
        self.advantages = torch.zeros((capacity,), dtype=torch.float32)
        self.returns = torch.zeros((capacity,), dtype=torch.float32)
        self.ptr = 0
        # episode ids to avoid crossing boundaries in sequences
        self.ep_id = torch.zeros((capacity,), dtype=torch.int32)
        self._ep_counter = 0

    def add(self, obs, action, reward, done, value, logprob):
        i = self.ptr
        self.obs[i].copy_(obs)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.logprobs[i] = logprob
        self.ep_id[i] = self._ep_counter
        self.ptr += 1
        if bool(done):
            self._ep_counter += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        adv = 0.0
        for i in range(self.ptr - 1, -1, -1):
            nonterminal = 1.0 - float(self.dones[i].item())
            delta = self.rewards[i] + gamma * last_value * nonterminal - self.values[i]
            adv = delta + gamma * lam * adv * nonterminal
            self.advantages[i] = adv
            self.returns[i] = self.advantages[i] + self.values[i]
            last_value = self.values[i].item()

    def get_minibatches(self, num_minibatches: int):
        idxs = torch.randperm(self.ptr)
        mb_size = self.ptr // num_minibatches
        for k in range(num_minibatches):
            s = k * mb_size
            e = (k + 1) * mb_size if k < num_minibatches - 1 else self.ptr
            batch = idxs[s:e]
            yield (
                self.obs[batch],
                self.actions[batch],
                self.returns[batch],
                self.advantages[batch],
                self.values[batch],
                self.logprobs[batch],
            )

    def get_sequence_minibatches(self, tbptt_len: int, num_minibatches: int) -> Iterator[Tuple[torch.Tensor, ...]]:
        # collect start indices where a full sequence of length L fits and does not cross episode boundary
        starts = []
        for i in range(0, self.ptr - tbptt_len):
            same_ep = (self.ep_id[i] == self.ep_id[i + tbptt_len - 1]).item()
            if same_ep:
                starts.append(i)
        starts = torch.tensor(starts, dtype=torch.int64)
        if len(starts) == 0:
            return
        perm = starts[torch.randperm(len(starts))]
        mb_size = max(1, len(perm) // num_minibatches)
        for k in range(num_minibatches):
            s = k * mb_size
            e = (k + 1) * mb_size if k < num_minibatches - 1 else len(perm)
            batch_starts = perm[s:e]
            B = len(batch_starts)
            L = tbptt_len
            # build sequences
            obs_seq = torch.stack([self.obs[i:i+L] for i in batch_starts], dim=0)              # [B,L,T,F]
            act_seq = torch.stack([self.actions[i:i+L] for i in batch_starts], dim=0)           # [B,L]
            ret_seq = torch.stack([self.returns[i:i+L] for i in batch_starts], dim=0)           # [B,L]
            adv_seq = torch.stack([self.advantages[i:i+L] for i in batch_starts], dim=0)        # [B,L]
            val_seq = torch.stack([self.values[i:i+L] for i in batch_starts], dim=0)            # [B,L]
            logp_seq = torch.stack([self.logprobs[i:i+L] for i in batch_starts], dim=0)         # [B,L]
            done_seq = torch.stack([self.dones[i:i+L] for i in batch_starts], dim=0)            # [B,L]
            mask_seq = (~done_seq).to(torch.float32)                                            # [B,L]
            yield (obs_seq, act_seq, ret_seq, adv_seq, val_seq, logp_seq, mask_seq)


def ppo_update(policy: nn.Module, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer, cfg: PPOConfig):
    advantages = buffer.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(cfg.update_epochs):
        for obs, actions, returns, advs, old_values, old_logprobs in buffer.get_minibatches(cfg.num_minibatches):
            # forward
            carry = policy.initial_carry(batch_size=obs.shape[0])
            carry, action_pred, logprob_pred, value_pred = policy.act(carry, obs, deterministic=False)

            # PPO losses
            ratio = torch.exp(logprob_pred - old_logprobs)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advs
            policy_loss = -torch.min(surr1, surr2).mean()

            value_clip = old_values + torch.clamp(value_pred - old_values, -cfg.clip_eps, cfg.clip_eps)
            value_losses = (value_pred - returns) ** 2
            value_losses_clipped = (value_clip - returns) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            entropy_bonus = -(torch.exp(logprob_pred) * logprob_pred).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()


def ppo_update_seq(policy: nn.Module, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer, cfg: PPOConfig):
    advantages = buffer.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(cfg.update_epochs):
        for obs_seq, act_seq, ret_seq, adv_seq, val_seq, logp_seq, mask_seq in buffer.get_sequence_minibatches(cfg.tbptt_len, cfg.num_minibatches):
            B, L, T, F = obs_seq.shape
            device = next(policy.parameters()).device
            obs_seq = obs_seq.to(device)
            act_seq = act_seq.to(device)
            ret_seq = ret_seq.to(device)
            adv_seq = adv_seq.to(device)
            val_seq = val_seq.to(device)
            logp_seq = logp_seq.to(device)
            mask_seq = mask_seq.to(device)

            carry = policy.initial_carry(batch_size=B)
            logp_preds = []
            value_preds = []
            for t in range(L):
                carry, a_pred, logp_pred, v_pred = policy.act(carry, obs_seq[:, t], deterministic=False)
                # mask invalid steps
                m = mask_seq[:, t]
                logp_preds.append(logp_pred * m)
                value_preds.append(v_pred * m)
                # reset carry where done==1 for next step
                if (~m.bool()).any():
                    reset_idx = (~m.bool()).nonzero(as_tuple=False).squeeze(-1)
                    init = policy.initial_carry(batch_size=len(reset_idx))
                    carry.z_H[reset_idx] = init.z_H
                    carry.z_L[reset_idx] = init.z_L

            logp_preds = torch.stack(logp_preds, dim=1)  # [B,L]
            value_preds = torch.stack(value_preds, dim=1)  # [B,L]

            ratio = torch.exp(logp_preds - logp_seq)
            surr1 = ratio * adv_seq
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_seq
            policy_loss = -torch.min(surr1, surr2)
            policy_loss = (policy_loss * mask_seq).sum() / (mask_seq.sum() + 1e-8)

            value_clip = val_seq + torch.clamp(value_preds - val_seq, -cfg.clip_eps, cfg.clip_eps)
            value_losses = (value_preds - ret_seq) ** 2
            value_losses_clipped = (value_clip - ret_seq) ** 2
            value_loss = torch.max(value_losses, value_losses_clipped)
            value_loss = 0.5 * (value_loss * mask_seq).sum() / (mask_seq.sum() + 1e-8)

            # entropy approx from logprob (no explicit dist entropy here)
            entropy_bonus = -(torch.exp(logp_preds) * logp_preds)
            entropy_bonus = (entropy_bonus * mask_seq).sum() / (mask_seq.sum() + 1e-8)

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()


