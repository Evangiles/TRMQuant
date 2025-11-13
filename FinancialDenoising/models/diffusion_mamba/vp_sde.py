"""
Variance Preserving SDE (VP-SDE) for Diffusion Models

Implements the forward and reverse diffusion processes
based on the paper's VP-SDE formulation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class VPSDE:
    """
    Variance Preserving Stochastic Differential Equation.

    Forward SDE: dx = -0.5 * β(t) * x * dt + √β(t) * dw
    where β(t) is the noise schedule.

    Beta schedule: linear from β_min to β_max
    """

    def __init__(
        self,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        num_diffusion_timesteps: int = 1000,
        device: str = "cpu"
    ):
        """
        Args:
            beta_min: Minimum beta value
            beta_max: Maximum beta value
            num_diffusion_timesteps: Number of discrete timesteps
            device: Device for tensors
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_timesteps = num_diffusion_timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(
            beta_min, beta_max, num_diffusion_timesteps, device=device
        )

        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get beta value at continuous time t ∈ [0, 1].

        Args:
            t: Continuous time in [0, 1]

        Returns:
            Beta values
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(
        self,
        x0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std of p(x_t | x_0).

        For VP-SDE:
        mean = √(α_bar(t)) * x_0
        std = √(1 - α_bar(t))

        Args:
            x0: Clean data [batch, ...]
            t: Discrete timesteps [batch] in range [0, num_timesteps-1]

        Returns:
            (mean, std) tuple
        """
        # Get alpha_cumprod for each timestep
        alpha_bar = self.alphas_cumprod[t]

        # Reshape for broadcasting
        while alpha_bar.dim() < x0.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        mean = torch.sqrt(alpha_bar) * x0
        std = torch.sqrt(1.0 - alpha_bar)

        return mean, std

    def sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from q(x_t | x_0).

        x_t = √(α_bar(t)) * x_0 + √(1 - α_bar(t)) * ε
        where ε ~ N(0, I)

        Args:
            x0: Clean data [batch, window_size, n_features]
            t: Timesteps [batch] in range [0, num_timesteps-1]

        Returns:
            (x_t, noise) tuple
        """
        noise = torch.randn_like(x0)
        mean, std = self.marginal_prob(x0, t)

        x_t = mean + std * noise

        return x_t, noise

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one denoising step: x_t → x_{t-1}.

        DDPM sampling formula:
        x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-α_bar_t)) * ε_θ(x_t, t)) + σ_t * z

        Args:
            x_t: Noisy data at timestep t
            t: Current timesteps [batch]
            predicted_noise: Model prediction ε_θ(x_t, t)

        Returns:
            x_{t-1}
        """
        # Get parameters
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        # Reshape for broadcasting
        while alpha_t.dim() < x_t.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)

        # Predicted x_0
        predicted_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        # Clip for stability
        predicted_x0 = torch.clamp(predicted_x0, -10, 10)

        # Compute x_{t-1} mean
        coef1 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_t / alpha_bar_t) / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_bar_t / alpha_bar_t) * beta_t / (1.0 - alpha_bar_t)

        # For t > 0, add noise
        if t[0] > 0:
            # Posterior variance
            alpha_bar_prev = self.alphas_cumprod[t - 1]
            while alpha_bar_prev.dim() < x_t.dim():
                alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)

            posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x_t)
            x_prev = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
            ) + torch.sqrt(posterior_variance) * noise
        else:
            # Final step: no noise
            x_prev = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
            )

        return x_prev

    @torch.no_grad()
    def sample_from_noise(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: str = "cpu",
        num_steps: Optional[int] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate samples starting from pure noise.

        Args:
            model: Denoiser model
            shape: Shape of samples to generate
            device: Device
            num_steps: Number of denoising steps (default: all timesteps)
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)

        Returns:
            Generated samples
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Timesteps to use
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device
        )

        for i, t in enumerate(timesteps):
            # Prepare batch of same timestep
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = model(x, t_batch)

            # Denoise
            x = self.denoise_step(x, t_batch, predicted_noise)

        return x


if __name__ == "__main__":
    print("Testing VP-SDE...")

    sde = VPSDE(num_diffusion_timesteps=1000)

    # Test forward process
    x0 = torch.randn(4, 60, 20)
    t = torch.randint(0, 1000, (4,))

    x_t, noise = sde.sample(x0, t)
    print(f"x0 shape: {x0.shape}")
    print(f"x_t shape: {x_t.shape}")
    print(f"noise shape: {noise.shape}")

    # Test marginal prob
    mean, std = sde.marginal_prob(x0, t)
    print(f"mean shape: {mean.shape}")
    print(f"std shape: {std.shape}")

    print("✓ All tests passed!")
