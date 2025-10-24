from typing import Tuple, Dict
from dataclasses import dataclass

import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear


@dataclass
class TRMPPOCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


class TRMPPOConfig(BaseModel):
    # data
    window_size: int
    num_features: int

    # transformer
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    pos_encodings: str = "rope"  # rope|learned

    # TRM recursion
    H_cycles: int = 3
    L_cycles: int = 6
    L_layers: int = 2

    forward_dtype: str = "bfloat16"


class TRMPPOBlock(nn.Module):
    def __init__(self, config: TRMPPOConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class TRMPPOTemporalEncoder(nn.Module):
    def __init__(self, config: TRMPPOConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # feature projection
        self.x_proj = CastedLinear(config.num_features, config.hidden_size, bias=True)

        # positional encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.window_size,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(config.window_size, config.hidden_size, init_std=1.0 / math.sqrt(config.hidden_size), cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # transformer layers
        self.L_level = nn.ModuleList([TRMPPOBlock(config) for _ in range(config.L_layers)])

        # initial states
        self.H_init = nn.Buffer(torch.zeros(config.hidden_size, dtype=self.forward_dtype), persistent=True)
        self.L_init = nn.Buffer(torch.zeros(config.hidden_size, dtype=self.forward_dtype), persistent=True)

        # actor/critic heads
        self.actor_mu = CastedLinear(config.hidden_size, 1, bias=True)
        self.actor_logstd = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.critic = CastedLinear(config.hidden_size, 1, bias=True)

    def _positional(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hasattr(self, "rotary_emb"):
            return dict(cos_sin=self.rotary_emb())
        else:
            return dict(cos_sin=None)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        h = self.x_proj(x.to(self.forward_dtype))
        if hasattr(self, "embed_pos"):
            h = 0.707106781 * (h + self.embed_pos.embedding_weight.to(self.forward_dtype))
        seq_info = self._positional(h)
        for layer in self.L_level:
            h = layer(hidden_states=h, **seq_info)
        return h  # [B, T, D]

    def initial_carry(self, batch_size: int) -> TRMPPOCarry:
        B = batch_size
        T = self.config.window_size
        D = self.config.hidden_size
        device = next(self.parameters()).device
        dtype = self.forward_dtype
        return TRMPPOCarry(
            z_H=torch.zeros(B, T, D, dtype=dtype, device=device),
            z_L=torch.zeros(B, T, D, dtype=dtype, device=device),
        )

    def forward(self, carry: TRMPPOCarry, obs: torch.Tensor) -> Tuple[TRMPPOCarry, Dict[str, torch.Tensor]]:
        # obs: [B, T, F]
        # Encode inputs
        x_enc = self._encode(obs)

        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = z_L + x_enc
                    z_L = self._encode_step(z_L)
                z_H = z_H + z_L
                z_H = self._encode_step(z_H)

        # 1 with grad
        for _ in range(self.config.L_cycles):
            z_L = z_L + x_enc
            z_L = self._encode_step(z_L)
        z_H = z_H + z_L
        z_H = self._encode_step(z_H)

        # pooled representation
        pooled = z_H.mean(dim=1).to(torch.float32)  # [B, D]

        mu = self.actor_mu(pooled).squeeze(-1)  # [B]
        log_std = self.actor_logstd.expand_as(mu)  # [B]
        value = self.critic(pooled).squeeze(-1)  # [B]

        new_carry = TRMPPOCarry(z_H=z_H.detach(), z_L=z_L.detach())

        return new_carry, {"mu": mu, "log_std": log_std, "value": value}

    def _encode_step(self, h: torch.Tensor) -> torch.Tensor:
        seq_info = self._positional(h)
        for layer in self.L_level:
            h = layer(hidden_states=h, **seq_info)
        return h

    @staticmethod
    def _tanh_squash(u: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.tanh(u)
        # log det Jacobian term: sum log(1 - tanh(u)^2)
        log_det = torch.log(1.0 - a * a + eps)
        return a, log_det

    def act(self, carry: TRMPPOCarry, obs: torch.Tensor, deterministic: bool = False) -> Tuple[TRMPPOCarry, torch.Tensor, torch.Tensor, torch.Tensor]:
        new_carry, out = self.forward(carry, obs)
        mu, log_std, value = out["mu"], out["log_std"], out["value"]

        std = torch.exp(log_std)
        if deterministic:
            u = mu
            base_log_prob = torch.zeros_like(mu)
        else:
            noise = torch.randn_like(mu)
            u = mu + std * noise
            base_log_prob = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))

        a_tanh, log_det = self._tanh_squash(u)
        action = (a_tanh + 1.0)  # map (-1,1) -> (0,2)
        log_prob = (base_log_prob - log_det).sum(dim=-1, keepdim=False)

        return new_carry, action.clamp(0.0, 2.0), log_prob, value


