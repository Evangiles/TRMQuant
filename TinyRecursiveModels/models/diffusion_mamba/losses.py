"""
Guidance Losses for Financial Time Series Denoising

Implements Total Variation (TV) loss and Fourier loss
for guidance in the diffusion denoising process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TVLoss(nn.Module):
    """
    Total Variation Loss for encouraging smoothness.

    Penalizes rapid changes in the time series, useful for
    spiky or mean-reverting features.

    TV(x) = Σ |x[t+1] - x[t]|
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute TV loss.

        Args:
            x: Input tensor [batch, seq_len, n_features]

        Returns:
            Scalar loss or [batch, n_features] if reduction='none'
        """
        # Compute differences
        diff = x[:, 1:, :] - x[:, :-1, :]  # [B, L-1, F]

        # L1 norm of differences
        tv = torch.abs(diff)

        if self.reduction == "mean":
            return tv.mean()
        elif self.reduction == "sum":
            return tv.sum()
        elif self.reduction == "none":
            return tv.sum(dim=1)  # [B, F]
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class FourierLoss(nn.Module):
    """
    Fourier Loss for suppressing high-frequency noise.

    Penalizes high-frequency components in the frequency domain,
    useful for smooth, trending features.

    Applies FFT and penalizes high-frequency magnitudes.
    """

    def __init__(
        self,
        high_freq_threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Args:
            high_freq_threshold: Fraction of frequencies to penalize (0.5 = top half)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.high_freq_threshold = high_freq_threshold
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier loss.

        Args:
            x: Input tensor [batch, seq_len, n_features]

        Returns:
            Scalar loss or [batch, n_features] if reduction='none'
        """
        batch_size, seq_len, n_features = x.shape

        # Apply FFT along time dimension
        # Move features to batch dimension for parallel FFT
        x_reshaped = x.transpose(1, 2)  # [B, F, L]
        x_flat = x_reshaped.reshape(-1, seq_len)  # [B*F, L]

        # Real FFT (input is real-valued)
        fft = torch.fft.rfft(x_flat, dim=-1)  # [B*F, L//2+1]
        magnitude = torch.abs(fft)

        # Identify high-frequency components
        freq_len = magnitude.shape[-1]
        cutoff_idx = int(freq_len * (1.0 - self.high_freq_threshold))

        # Penalize high frequencies
        high_freq_magnitude = magnitude[:, cutoff_idx:]
        loss = high_freq_magnitude.pow(2).sum(dim=-1)  # [B*F]

        # Reshape back
        loss = loss.reshape(batch_size, n_features)  # [B, F]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss  # [B, F]
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class GuidanceLoss(nn.Module):
    """
    Combined guidance loss for cluster-specific denoising.

    Combines TV loss and Fourier loss based on cluster characteristics:
    - Mean-reverting / spiky clusters: higher TV weight
    - Smooth / trending clusters: higher Fourier weight
    """

    def __init__(
        self,
        cluster_type: str,
        tv_weight: float = 1.0,
        fourier_weight: float = 1.0,
        high_freq_threshold: float = 0.5
    ):
        """
        Args:
            cluster_type: 'mean_reverting', 'trending', or 'random_walk'
            tv_weight: Weight for TV loss
            fourier_weight: Weight for Fourier loss
            high_freq_threshold: Threshold for Fourier loss
        """
        super().__init__()

        self.cluster_type = cluster_type

        # Adjust weights based on cluster type
        if cluster_type == "mean_reverting":
            # Emphasize smoothness (suppress spikes)
            self.tv_weight = tv_weight * 2.0
            self.fourier_weight = fourier_weight * 0.5
        elif cluster_type == "trending":
            # Emphasize low-frequency (preserve trends)
            self.tv_weight = tv_weight * 0.5
            self.fourier_weight = fourier_weight * 2.0
        else:  # random_walk or default
            # Balanced
            self.tv_weight = tv_weight
            self.fourier_weight = fourier_weight

        self.tv_loss = TVLoss(reduction="mean")
        self.fourier_loss = FourierLoss(
            high_freq_threshold=high_freq_threshold,
            reduction="mean"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute combined guidance loss.

        Args:
            x: Denoised output [batch, seq_len, n_features]

        Returns:
            Scalar loss
        """
        loss = 0.0

        if self.tv_weight > 0:
            loss += self.tv_weight * self.tv_loss(x)

        if self.fourier_weight > 0:
            loss += self.fourier_weight * self.fourier_loss(x)

        return loss


class DenoisingLoss(nn.Module):
    """
    Complete denoising loss combining reconstruction and guidance.

    Loss = MSE(predicted_noise, true_noise) + λ * GuidanceLoss(denoised)
    """

    def __init__(
        self,
        cluster_type: str = "random_walk",
        guidance_weight: float = 0.1,
        tv_weight: float = 1.0,
        fourier_weight: float = 1.0
    ):
        """
        Args:
            cluster_type: Cluster characteristic
            guidance_weight: Overall guidance weight λ
            tv_weight: TV loss weight
            fourier_weight: Fourier loss weight
        """
        super().__init__()

        self.guidance_weight = guidance_weight
        self.mse_loss = nn.MSELoss()

        if guidance_weight > 0:
            self.guidance_loss = GuidanceLoss(
                cluster_type=cluster_type,
                tv_weight=tv_weight,
                fourier_weight=fourier_weight
            )
        else:
            self.guidance_loss = None

    def forward(
        self,
        predicted_noise: torch.Tensor,
        true_noise: torch.Tensor,
        denoised_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            predicted_noise: Model prediction [batch, seq_len, n_features]
            true_noise: True noise [batch, seq_len, n_features]
            denoised_output: Denoised signal (for guidance loss)

        Returns:
            Total loss
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(predicted_noise, true_noise)

        # Guidance loss (if provided)
        if self.guidance_loss is not None and denoised_output is not None:
            guide_loss = self.guidance_loss(denoised_output)
            total_loss = recon_loss + self.guidance_weight * guide_loss
        else:
            total_loss = recon_loss

        return total_loss


if __name__ == "__main__":
    print("Testing guidance losses...")

    batch_size = 4
    seq_len = 60
    n_features = 20

    x = torch.randn(batch_size, seq_len, n_features)

    # Test TV loss
    tv_loss = TVLoss()
    loss_tv = tv_loss(x)
    print(f"TV Loss: {loss_tv.item():.4f}")

    # Test Fourier loss
    fourier_loss = FourierLoss()
    loss_fourier = fourier_loss(x)
    print(f"Fourier Loss: {loss_fourier.item():.4f}")

    # Test combined guidance loss
    guidance = GuidanceLoss(cluster_type="mean_reverting")
    loss_guidance = guidance(x)
    print(f"Guidance Loss (mean_reverting): {loss_guidance.item():.4f}")

    # Test denoising loss
    predicted_noise = torch.randn_like(x)
    true_noise = torch.randn_like(x)
    denoising_loss = DenoisingLoss(cluster_type="trending", guidance_weight=0.1)
    loss_total = denoising_loss(predicted_noise, true_noise, x)
    print(f"Total Denoising Loss: {loss_total.item():.4f}")

    print("✓ All tests passed!")
