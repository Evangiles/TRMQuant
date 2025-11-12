"""
Mamba SSM Block Implementation

Simplified Mamba (State Space Model) with selective scan mechanism.
O(L) complexity for sequence processing (vs Transformer's O(L²)).

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core component.

    Uses input-dependent parameters (selective mechanism) for efficient
    long-range dependency modeling with O(L) complexity.

    State equation:
        h_t = A_t * h_{t-1} + B_t * x_t
        y_t = C_t * h_t

    where A_t, B_t, C_t are input-dependent (selective).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand_factor: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension (default: 16)
            expand_factor: Expansion factor for intermediate dimension
            dt_rank: Rank for delta (timestep) projection
            dt_min: Minimum delta value for stability
            dt_max: Maximum delta value for stability
            dt_init: Delta initialization strategy
            dt_scale: Delta scaling factor
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # Compute dt_rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameters
        # Delta (timestep) projection - input-dependent
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A (state transition matrix)
        # Use positive initialization for log-space stability
        A = torch.rand(self.d_inner, self.d_state) + 0.5  # Range [0.5, 1.5]
        self.A_log = nn.Parameter(torch.log(A))  # Log-space for stability

        # Initialize D (skip connection parameter)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # B and C projections (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Delta parameters
        self.dt_min = dt_min
        self.dt_max = dt_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with selective SSM.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]

        # Compute selective parameters (input-dependent)
        x_dbl = self.x_proj(x_proj)  # [B, L, dt_rank + 2*d_state]

        # Split into delta, B, C
        delta, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        # Compute actual delta (timestep)
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner]
        delta = torch.clamp(delta, self.dt_min, self.dt_max)

        # Discretize A matrix
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Selective scan (simplified associative scan)
        y = self.selective_scan(x_proj, delta, A, B, C)

        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_proj

        # Gated activation
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective scan operation (simplified parallel scan).

        This is a simplified version. Full Mamba uses hardware-optimized
        parallel scan algorithms for efficiency.

        Args:
            x: Input [B, L, d_inner]
            delta: Timestep [B, L, d_inner]
            A: State transition [d_inner, d_state]
            B: Input matrix [B, L, d_state]
            C: Output matrix [B, L, d_state]

        Returns:
            Output [B, L, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []

        # Sequential scan (can be parallelized with associative scan)
        for t in range(seq_len):
            # Get current timestep parameters
            dt_t = delta[:, t, :]  # [B, d_inner]
            B_t = B[:, t, :]  # [B, d_state]
            C_t = C[:, t, :]  # [B, d_state]
            x_t = x[:, t, :]  # [B, d_inner]

            # Discretize: A_bar = exp(A * dt)
            # Simplified: A_bar ≈ (1 + A * dt)
            A_bar = 1.0 + A.unsqueeze(0) * dt_t.unsqueeze(-1)  # [B, d_inner, d_state]

            # B_bar = B * dt
            B_bar = B_t.unsqueeze(1) * dt_t.unsqueeze(-1)  # [B, d_inner, d_state]

            # Update hidden state: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)

            # Compute output: y = C * h
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1)  # [B, d_inner]

            outputs.append(y_t)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]

        return y


class MambaBlock(nn.Module):
    """
    Single Mamba block with layer norm and residual connection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            expand_factor: Expansion factor for SSM
            dropout: Dropout probability
        """
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            expand_factor=expand_factor
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        # Pre-norm + residual
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = x + residual

        return x


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block for processing sequences in both directions.

    Processes temporal dimension bidirectionally for better context modeling.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            expand_factor: Expansion factor
            dropout: Dropout probability
        """
        super().__init__()

        # Forward and backward Mamba blocks
        self.forward_block = MambaBlock(d_model, d_state, expand_factor, dropout)
        self.backward_block = MambaBlock(d_model, d_state, expand_factor, dropout)

        # Combine forward and backward outputs
        self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        # Forward direction
        forward_out = self.forward_block(x)

        # Backward direction (flip sequence)
        backward_in = torch.flip(x, dims=[1])
        backward_out = self.backward_block(backward_in)
        backward_out = torch.flip(backward_out, dims=[1])

        # Combine
        combined = torch.cat([forward_out, backward_out], dim=-1)
        output = self.output_proj(combined)

        return output


if __name__ == "__main__":
    # Test
    print("Testing Mamba blocks...")

    batch_size = 4
    seq_len = 60
    d_model = 128

    x = torch.randn(batch_size, seq_len, d_model)

    # Test SelectiveSSM
    ssm = SelectiveSSM(d_model)
    out = ssm(x)
    print(f"SelectiveSSM output shape: {out.shape}")

    # Test MambaBlock
    mamba = MambaBlock(d_model)
    out = mamba(x)
    print(f"MambaBlock output shape: {out.shape}")

    # Test BiMambaBlock
    bi_mamba = BiMambaBlock(d_model)
    out = bi_mamba(x)
    print(f"BiMambaBlock output shape: {out.shape}")

    print("✓ All tests passed!")
