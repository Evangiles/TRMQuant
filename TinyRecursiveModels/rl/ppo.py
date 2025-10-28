from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterator, Dict

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
    rollout_steps: int = 1024
    update_epochs: int = 6
    num_minibatches: int = 8
    tbptt_len: int = 8


class RolloutBuffer:
    def __init__(self, obs_shape: Tuple[int, int], capacity: int, hidden_size: int, window_size: int) -> None:
        T, F = obs_shape
        self.capacity = capacity
        self.obs = torch.zeros((capacity, T, F), dtype=torch.float32)
        self.actions = torch.zeros((capacity,), dtype=torch.float32)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.bool)
        self.values = torch.zeros((capacity,), dtype=torch.float32)
        self.logprobs = torch.zeros((capacity,), dtype=torch.float32)
        self.advantages = torch.zeros((capacity,), dtype=torch.float32)
        self.returns = torch.zeros((capacity,), dtype=torch.float32)

        # Carry state storage for recurrent policy
        self.carries_z_H = torch.zeros((capacity, window_size, hidden_size), dtype=torch.float32)
        self.carries_z_L = torch.zeros((capacity, window_size, hidden_size), dtype=torch.float32)

        self.ptr = 0
        # episode ids to avoid crossing boundaries in sequences
        self.ep_id = torch.zeros((capacity,), dtype=torch.int32)
        self._ep_counter = 0

    def add(self, obs, action, reward, done, value, logprob, carry):
        """
        Add transition to buffer.

        Args:
            obs: Observation [T, F]
            action: Action (scalar)
            reward: Reward (scalar)
            done: Done flag (bool)
            value: Value estimate (scalar)
            logprob: Log probability (scalar)
            carry: TRMPPOCarry with z_H [1, T, D], z_L [1, T, D]
        """
        i = self.ptr
        self.obs[i].copy_(obs)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.logprobs[i] = logprob

        # Store carry states (squeeze batch dimension)
        self.carries_z_H[i].copy_(carry.z_H.squeeze(0).cpu())
        self.carries_z_L[i].copy_(carry.z_L.squeeze(0).cpu())

        self.ep_id[i] = self._ep_counter
        self.ptr += 1
        if bool(done):
            self._ep_counter += 1

    def add_batch(self, obs_b: torch.Tensor, act_b: torch.Tensor, rew_b: torch.Tensor, done_b: torch.Tensor,
                  val_b: torch.Tensor, logp_b: torch.Tensor, carry_b) -> None:
        """
        Add a batch of transitions to buffer in one shot.

        Args:
            obs_b: [N, T, F]
            act_b: [N]
            rew_b: [N]
            done_b: [N]
            val_b: [N]
            logp_b: [N]
            carry_b: TRMPPOCarry with z_H/z_L [N, T, D]
        """
        N = int(obs_b.shape[0])
        if self.ptr + N > self.capacity:
            N = max(0, self.capacity - self.ptr)
            if N == 0:
                return
            obs_b = obs_b[:N]
            act_b = act_b[:N]
            rew_b = rew_b[:N]
            done_b = done_b[:N]
            val_b = val_b[:N]
            logp_b = logp_b[:N]
            carry_H = carry_b.z_H[:N]
            carry_L = carry_b.z_L[:N]
        else:
            carry_H = carry_b.z_H
            carry_L = carry_b.z_L

        s, e = self.ptr, self.ptr + N
        # ensure CPU float32/bool tensors
        self.obs[s:e].copy_(obs_b.detach().to(dtype=self.obs.dtype, device=self.obs.device))
        self.actions[s:e].copy_(act_b.detach().to(dtype=self.actions.dtype, device=self.actions.device))
        self.rewards[s:e].copy_(rew_b.detach().to(dtype=self.rewards.dtype, device=self.rewards.device))
        self.dones[s:e].copy_(done_b.detach().to(dtype=self.dones.dtype, device=self.dones.device))
        self.values[s:e].copy_(val_b.detach().to(dtype=self.values.dtype, device=self.values.device))
        self.logprobs[s:e].copy_(logp_b.detach().to(dtype=self.logprobs.dtype, device=self.logprobs.device))
        self.carries_z_H[s:e].copy_(carry_H.detach().to(dtype=self.carries_z_H.dtype, device=self.carries_z_H.device))
        self.carries_z_L[s:e].copy_(carry_L.detach().to(dtype=self.carries_z_L.dtype, device=self.carries_z_L.device))

        # episode id accounting
        done_cpu = self.dones[s:e]
        for i in range(N):
            self.ep_id[s + i] = self._ep_counter
            if bool(done_cpu[i].item()):
                self._ep_counter += 1

        self.ptr = e

    def clear(self) -> None:
        self.ptr = 0
        self._ep_counter = 0

    def is_full(self) -> bool:
        return self.ptr >= self.capacity

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
        """
        Generate sequence minibatches with initial carry states.

        Returns:
            Iterator yielding (obs_seq, act_seq, ret_seq, adv_seq, val_seq, logp_seq, mask_seq, carry_H_init, carry_L_init)
            where carry_H_init and carry_L_init are the initial carry states at each sequence start [B, T, D]
        """
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

            # Vectorized sequence extraction using advanced indexing
            # Create index tensor: [B, L] where each row is [i, i+1, ..., i+L-1]
            indices = batch_starts.unsqueeze(1) + torch.arange(L, device=batch_starts.device).unsqueeze(0)  # [B, L]

            # Extract sequences using advanced indexing (much faster than for loop)
            obs_seq = self.obs[indices]                    # [B, L, T, F]
            act_seq = self.actions[indices]                # [B, L]
            ret_seq = self.returns[indices]                # [B, L]
            adv_seq = self.advantages[indices]             # [B, L]
            val_seq = self.values[indices]                 # [B, L]
            logp_seq = self.logprobs[indices]              # [B, L]
            done_seq = self.dones[indices]                 # [B, L]
            mask_seq = (~done_seq).to(torch.float32)       # [B, L]

            # Extract initial carry states (only at sequence start)
            carry_H_init = self.carries_z_H[batch_starts]  # [B, T, D]
            carry_L_init = self.carries_z_L[batch_starts]  # [B, T, D]

            yield (obs_seq, act_seq, ret_seq, adv_seq, val_seq, logp_seq, mask_seq, carry_H_init, carry_L_init)


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


def ppo_update_seq(policy: nn.Module, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer, cfg: PPOConfig) -> Dict[str, float]:
    advantages = buffer.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # metric accumulators
    agg = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, clip_frac=0.0, adv_mean=0.0, adv_std=0.0)
    batches = 0

    for _ in range(cfg.update_epochs):
        for obs_seq, act_seq, ret_seq, adv_seq, val_seq, logp_seq, mask_seq, carry_H_init, carry_L_init in buffer.get_sequence_minibatches(cfg.tbptt_len, cfg.num_minibatches):
            B, L, T, F = obs_seq.shape
            device = next(policy.parameters()).device
            obs_seq = obs_seq.to(device, non_blocking=True)
            act_seq = act_seq.to(device, non_blocking=True)
            ret_seq = ret_seq.to(device, non_blocking=True)
            adv_seq = adv_seq.to(device, non_blocking=True)
            val_seq = val_seq.to(device, non_blocking=True)
            logp_seq = logp_seq.to(device, non_blocking=True)
            mask_seq = mask_seq.to(device, non_blocking=True)

            # Use stored carry states instead of initial_carry()
            from models.recursive_reasoning.trm_ppo import TRMPPOCarry
            carry = TRMPPOCarry(
                z_H=carry_H_init.to(device, non_blocking=True),
                z_L=carry_L_init.to(device, non_blocking=True)
            )
            logp_preds = []
            value_preds = []
            for t in range(L):
                carry, a_pred, logp_pred, v_pred = policy.act(carry, obs_seq[:, t], deterministic=False)
                # mask invalid steps
                m = mask_seq[:, t]
                logp_preds.append(logp_pred * m)
                value_preds.append(v_pred * m)
                # Note: No need for manual carry reset!
                # Sequences are constructed to not cross episode boundaries,
                # so carry naturally flows through valid timesteps only.

            logp_preds = torch.stack(logp_preds, dim=1)  # [B,L]
            value_preds = torch.stack(value_preds, dim=1)  # [B,L]

            # stabilize ratio by clipping log prob difference
            dlogp = torch.clamp(logp_preds - logp_seq, min=-20.0, max=20.0)
            ratio = torch.exp(dlogp)
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

            # clip fraction
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).to(torch.float32)
            clip_frac = (clip_frac * mask_seq).sum() / (mask_seq.sum() + 1e-8)

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            # accumulate metrics
            agg['policy_loss'] += float(policy_loss.detach().cpu())
            agg['value_loss'] += float(value_loss.detach().cpu())
            agg['entropy'] += float(entropy_bonus.detach().cpu())
            agg['clip_frac'] += float(clip_frac.detach().cpu())
            agg['adv_mean'] += float((adv_seq * mask_seq).sum().cpu() / (mask_seq.sum().cpu() + 1e-8))
            # std over masked entries
            adv_masked = adv_seq[mask_seq.bool()]
            agg['adv_std'] += float(adv_masked.std().cpu()) if adv_masked.numel() > 1 else 0.0
            batches += 1

    if batches > 0:
        for k in agg:
            agg[k] /= batches
    return agg


