from dataclasses import dataclass
import math
import torch
from torch import nn
from pydantic import BaseModel

from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear


class TRMSupRecConfig(BaseModel):
    window_size: int
    num_features: int
    hidden_size: int = 256
    num_heads: int = 4
    expansion: float = 4.0
    pos_encodings: str = "rope"  # "rope" | "learned"
    L_layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 4
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"


class _Block(nn.Module):
    def __init__(self, cfg: TRMSupRecConfig) -> None:
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
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class _Reasoning(nn.Module):
    def __init__(self, cfg: TRMSupRecConfig):
        super().__init__()
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(cfg.L_layers)])

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        h = hidden_states + input_injection
        for layer in self.layers:
            h = layer(hidden_states=h, **kwargs)
        return h


class TRMSupervisedRecursive(nn.Module):
    def __init__(self, cfg: TRMSupRecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.forward_dtype = getattr(torch, cfg.forward_dtype)

        # Feature projection
        self.x_proj = CastedLinear(cfg.num_features, cfg.hidden_size, bias=True)

        # Positional encodings
        if cfg.pos_encodings == "rope":
            self.rotary = RotaryEmbedding(
                dim=cfg.hidden_size // cfg.num_heads,
                max_position_embeddings=cfg.window_size,
                base=cfg.rope_theta,
            )
        elif cfg.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                cfg.window_size,
                cfg.hidden_size,
                init_std=1.0 / math.sqrt(cfg.hidden_size),
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning module and initial states
        self.reason = _Reasoning(cfg)
        self.H_init = nn.Buffer(torch.zeros(cfg.hidden_size, dtype=self.forward_dtype), persistent=True)
        self.L_init = nn.Buffer(torch.zeros(cfg.hidden_size, dtype=self.forward_dtype), persistent=True)

        # Allocation head
        self.head = CastedLinear(cfg.hidden_size, 1, bias=True)

    def _seq_info(self):
        if hasattr(self, "rotary"):
            return dict(cos_sin=self.rotary())
        else:
            return dict(cos_sin=None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, T, F]
        x = self.x_proj(obs.to(self.forward_dtype))
        if hasattr(self, "embed_pos"):
            x = 0.707106781 * (x + self.embed_pos.embedding_weight.to(self.forward_dtype))

        B, T, D = x.shape
        # Initialize recursive states per sequence (no cross-batch carry in supervised)
        z_H = self.H_init.expand(B, D).unsqueeze(1).expand(B, T, D)
        z_L = self.L_init.expand(B, D).unsqueeze(1).expand(B, T, D)

        seq_info = self._seq_info()

        # H_cycles-1 without grad (bootstrap pass)
        with torch.no_grad():
            for _ in range(self.cfg.H_cycles - 1):
                for _ in range(self.cfg.L_cycles):
                    z_L = self.reason(z_L, z_H + x, **seq_info)
                z_H = self.reason(z_H, z_L, **seq_info)

        # Final pass with gradients
        for _ in range(self.cfg.L_cycles):
            z_L = self.reason(z_L, z_H + x, **seq_info)
        z_H = self.reason(z_H, z_L, **seq_info)

        # Predict allocation from last timestep (most recent info)
        last = z_H[:, -1, :].to(torch.float32)
        a = torch.tanh(self.head(last).squeeze(-1))  # (-1, 1)
        return (a + 1.0).clamp(0.0, 2.0)


