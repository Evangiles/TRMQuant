from dataclasses import dataclass
import math
import torch
from torch import nn
from pydantic import BaseModel

from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear


class TRMSupervisedConfig(BaseModel):
    window_size: int
    num_features: int
    hidden_size: int = 256
    num_heads: int = 4
    expansion: float = 4.0
    pos_encodings: str = "rope"
    L_layers: int = 2
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"


class Block(nn.Module):
    def __init__(self, cfg: TRMSupervisedConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=cfg.hidden_size,
            head_dim=cfg.hidden_size // cfg.num_heads,
            num_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=cfg.hidden_size, expansion=cfg.expansion)
        self.norm_eps = cfg.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class TRMSupervised(nn.Module):
    def __init__(self, cfg: TRMSupervisedConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.forward_dtype = getattr(torch, cfg.forward_dtype)

        self.x_proj = CastedLinear(cfg.num_features, cfg.hidden_size, bias=True)

        if cfg.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=cfg.hidden_size // cfg.num_heads,
                max_position_embeddings=cfg.window_size,
                base=cfg.rope_theta,
            )
        elif cfg.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(cfg.window_size, cfg.hidden_size, init_std=1.0 / math.sqrt(cfg.hidden_size), cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_layers)])
        self.head = CastedLinear(cfg.hidden_size, 1, bias=True)

    def _pos(self, h: torch.Tensor):
        if hasattr(self, "rotary_emb"):
            return dict(cos_sin=self.rotary_emb())
        else:
            return dict(cos_sin=None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, T, F]
        h = self.x_proj(obs.to(self.forward_dtype))
        if hasattr(self, "embed_pos"):
            h = 0.707106781 * (h + self.embed_pos.embedding_weight.to(self.forward_dtype))
        seq_info = self._pos(h)
        for layer in self.layers:
            h = layer(hidden_states=h, **seq_info)
        pooled = h.mean(dim=1).to(torch.float32)
        # map to [0,2] via tanh squash centered at 1.0
        a = torch.tanh(self.head(pooled).squeeze(-1))
        return (a + 1.0).clamp(0.0, 2.0)


