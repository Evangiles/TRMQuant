"""Variance Preserving SDE for diffusion models."""

import torch
import numpy as np


class VPSDE:
    """Variance Preserving Stochastic Differential Equation.

    Forward: dx = -β(t)/2 * x dt + √β(t) dw
    Reverse: dx = [-β(t)/2 * x - β(t) * score] dt + √β(t) dw̄
    """

    def __init__(
        self,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        num_timesteps: int = 1000,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_min, beta_max, num_timesteps)

        # Precompute alphas
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta(t) for continuous time t ∈ [0, 1]."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_alpha(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Get alpha_cumprod for discrete timestep index."""
        # Index on CPU, preserve device of result
        if t_idx.is_cuda:
            return self.alphas_cumprod[t_idx.cpu()].to(t_idx.device)
        return self.alphas_cumprod[t_idx]

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t_idx: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion: x_t = √α_t * x_0 + √(1-α_t) * ε.

        Args:
            x0: Original data [B, L]
            t_idx: Timestep indices [B] in [0, num_timesteps-1]
            noise: Optional noise [B, L], sampled if None

        Returns:
            x_t: Noised data [B, L]
            noise: The noise used [B, L]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Get alpha for this timestep (index on CPU, then move to device)
        alpha = self.alphas_cumprod[t_idx.cpu()].to(x0.device)

        # Reshape for broadcasting [B, 1]
        alpha = alpha.view(-1, 1)

        # Apply noise
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)

        x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def reverse_step(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        score: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Perform one reverse diffusion step (predictor).

        Args:
            x_t: Noised data at timestep t [B, L]
            t_idx: Current timestep index (int)
            score: Predicted score ∇log p(x_t) [B, L]
            noise: Optional noise for stochastic sampling [B, L]

        Returns:
            x_{t-1}: Denoised data at timestep t-1 [B, L]
        """
        if t_idx == 0:
            return x_t

        # Index on CPU for compatibility
        beta_t = self.betas[t_idx].to(x_t.device)
        alpha_t = self.alphas_cumprod[t_idx].to(x_t.device)

        # Predictor step
        # x_{t-1} = (1 / √(1-β_t)) * (x_t + β_t * score) + √β_t * z
        sqrt_one_minus_beta = torch.sqrt(1.0 - beta_t)
        x_pred = (1.0 / sqrt_one_minus_beta) * (x_t + beta_t * score)

        # Add stochastic noise
        if noise is None:
            noise = torch.randn_like(x_t)
        x_pred = x_pred + torch.sqrt(beta_t) * noise

        return x_pred

    def corrector_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        step_size: float = 2e-5,
    ) -> torch.Tensor:
        """Langevin MCMC corrector step.

        Args:
            x_t: Current state [B, L]
            score: Predicted score [B, L]
            step_size: Langevin step size

        Returns:
            x_corrected: Corrected state [B, L]
        """
        noise = torch.randn_like(x_t)
        step_size_tensor = torch.tensor(step_size, device=x_t.device, dtype=x_t.dtype)
        x_corrected = x_t + step_size_tensor * score + torch.sqrt(2 * step_size_tensor) * noise
        return x_corrected
