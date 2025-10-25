"""Financial time series denoiser using diffusion model."""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] timesteps

        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Conv1dBlock(nn.Module):
    """1D Convolution block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]

        Returns:
            out: [B, C', L']
        """
        return self.act(self.norm(self.conv(x)))


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]

        Returns:
            out: [B, L, D]
        """
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class FinancialDenoiser(nn.Module):
    """Lightweight diffusion denoiser for financial time series.

    Architecture: 1D CNN Encoder + Transformer + 1D CNN Decoder
    Optimized for window_size=60 univariate financial features.
    """

    def __init__(
        self,
        window_size: int = 60,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        time_embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model

        # Time embedding for diffusion timestep
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 1D CNN Encoder (extract local patterns)
        self.encoder = nn.Sequential(
            Conv1dBlock(1, 64, kernel_size=5, padding=2),      # [B, 64, 60]
            Conv1dBlock(64, 128, kernel_size=3, padding=1),    # [B, 128, 60]
            Conv1dBlock(128, d_model, kernel_size=3, padding=1),  # [B, 256, 60]
        )

        # Condition projection (element-wise injection)
        self.cond_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )

        # Positional encoding for sequence
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # Transformer (global dependencies)
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        # 1D CNN Decoder (reconstruct signal)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, 128, kernel_size=3, padding=1),  # [B, 128, 60]
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),      # [B, 64, 60]
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=5, padding=2),        # [B, 1, 60]
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through denoiser.

        Args:
            x: Noised input [B, L] where L=window_size
            t: Diffusion timestep [B] in [0, num_timesteps-1]
            cond: Condition (original x_0) [B, L], optional

        Returns:
            score: Predicted score ∇log p(x_t|cond) [B, L]
        """
        B, L = x.shape
        assert L == self.window_size, f"Expected L={self.window_size}, got {L}"

        # Time embedding [B, D]
        t_emb = self.time_embed(t)

        # Expand input to [B, 1, L] for conv
        x_expanded = x.unsqueeze(1)

        # CNN Encoder [B, D, L]
        h = self.encoder(x_expanded)

        # Condition injection (if provided)
        if cond is not None:
            cond_expanded = cond.unsqueeze(1)  # [B, 1, L]
            cond_emb = self.cond_proj(cond_expanded)  # [B, D, L]
            h = h + cond_emb

        # Add time embedding (broadcast across sequence)
        # h: [B, D, L], t_emb: [B, D] -> [B, D, 1]
        h = h + t_emb.unsqueeze(-1)

        # Transpose for transformer [B, L, D]
        h = h.transpose(1, 2)

        # Add positional encoding
        h = h + self.pos_embed

        # Transformer blocks
        for block in self.transformer:
            h = block(h)

        # Transpose back [B, D, L]
        h = h.transpose(1, 2)

        # CNN Decoder [B, 1, L]
        score = self.decoder(h)

        # Squeeze back to [B, L]
        score = score.squeeze(1)

        return score
