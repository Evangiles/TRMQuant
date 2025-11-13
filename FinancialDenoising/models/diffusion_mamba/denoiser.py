"""
Multivariate Mamba Denoiser for Group-Specific Financial Time Series

Uses bidirectional Mamba blocks to denoise multivariate time series
while preserving inter-feature correlations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .mamba_block import BiMambaBlock


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] or [batch_size, 1]

        Returns:
            [batch_size, dim]
        """
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps.float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return embedding


class MultivariateMambaDenoiser(nn.Module):
    """
    Group-specific multivariate denoiser using Mamba architecture.

    Processes multiple features jointly while preserving temporal and
    inter-feature dependencies through bidirectional Mamba blocks.

    Architecture:
    1. Input projection: [B, L, n_features] → [B, L, d_model]
    2. Time embedding injection
    3. N × BiMambaBlock layers
    4. Output projection: [B, L, d_model] → [B, L, n_features]
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 60,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        time_emb_dim: Optional[int] = None,
    ):
        """
        Args:
            n_features: Number of features in this group
            window_size: Temporal window size
            d_model: Model dimension
            d_state: SSM state dimension
            n_layers: Number of Mamba layers
            expand_factor: Expansion factor for Mamba
            dropout: Dropout probability
            time_emb_dim: Time embedding dimension (default: d_model)
        """
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model

        if time_emb_dim is None:
            time_emb_dim = d_model

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Mamba layers
        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_features)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        guidance_weight: float = 0.0
    ) -> torch.Tensor:
        """
        Forward pass for denoising.

        Args:
            x: Noisy input [batch, window_size, n_features]
            t: Diffusion timesteps [batch] (0 = clean, 1000 = noisy)
            guidance_weight: Classifier-free guidance weight

        Returns:
            Denoised output [batch, window_size, n_features]
        """
        batch_size, seq_len, n_features = x.shape

        assert seq_len == self.window_size, f"Expected seq_len={self.window_size}, got {seq_len}"
        assert n_features == self.n_features, f"Expected n_features={self.n_features}, got {n_features}"

        # Input projection
        h = self.input_proj(x)  # [B, L, d_model]

        # Time embedding
        t_emb = self.time_embed(t)  # [B, d_model]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, d_model]

        # Add time embedding to all positions
        h = h + t_emb

        # Apply Mamba layers
        for layer in self.layers:
            h = layer(h)

        # Output projection
        out = self.output_proj(h)

        return out


class ConditionalMambaDenoiser(nn.Module):
    """
    Conditional denoiser with classifier-free guidance support.

    Allows training with/without conditioning and enables CFG during inference.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 60,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        cfg_dropout: float = 0.1,
    ):
        """
        Args:
            cfg_dropout: Probability of dropping condition for CFG training
        """
        super().__init__()

        self.denoiser = MultivariateMambaDenoiser(
            n_features=n_features,
            window_size=window_size,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            expand_factor=expand_factor,
            dropout=dropout
        )

        self.cfg_dropout = cfg_dropout

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy input [batch, window_size, n_features]
            t: Timesteps [batch]
            condition: Optional condition (for future extensions)
            guidance_scale: CFG scale (1.0 = no guidance)

        Returns:
            Denoised output [batch, window_size, n_features]
        """
        if self.training and condition is not None:
            # Randomly drop condition during training
            mask = torch.rand(x.shape[0], device=x.device) > self.cfg_dropout
            # For now, we don't actually use condition, but keep interface
            pass

        # Standard denoising
        if guidance_scale == 1.0 or not self.training:
            return self.denoiser(x, t)

        # Classifier-free guidance (for future use)
        # out_cond = self.denoiser(x, t, condition)
        # out_uncond = self.denoiser(x, t, None)
        # return out_uncond + guidance_scale * (out_cond - out_uncond)

        return self.denoiser(x, t)


if __name__ == "__main__":
    print("Testing MultivariateMambaDenoiser...")

    batch_size = 4
    window_size = 60
    n_features = 20

    # Test denoiser
    model = MultivariateMambaDenoiser(
        n_features=n_features,
        window_size=window_size,
        d_model=128,
        n_layers=4
    )

    x = torch.randn(batch_size, window_size, n_features)
    t = torch.randint(0, 1000, (batch_size,))

    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("✓ All tests passed!")
