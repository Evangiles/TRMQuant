"""Auxiliary loss functions for diffusion denoising."""

import torch
import torch.nn.functional as F


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total Variation loss for smoothness.

    Args:
        x: Time series [B, L]

    Returns:
        loss: Scalar TV loss
    """
    if x.shape[-1] <= 1:
        return torch.tensor(0.0, device=x.device)

    diff = x[:, 1:] - x[:, :-1]
    return torch.abs(diff).mean()


def tv_gradient(x: torch.Tensor) -> torch.Tensor:
    """Gradient of TV loss w.r.t. x.

    Args:
        x: Time series [B, L]

    Returns:
        grad: Gradient [B, L]
    """
    if x.shape[-1] <= 1:
        return torch.zeros_like(x)

    grad = torch.zeros_like(x)

    # Forward differences
    grad[:, :-1] = torch.sign(x[:, 1:] - x[:, :-1])

    # Backward differences
    grad[:, 1:] -= torch.sign(x[:, 1:] - x[:, :-1])

    return grad


def fourier_loss(
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Fourier loss for frequency preservation.

    Args:
        x_t: Noised/denoised series [B, L]
        x_0: Original series [B, L]
        threshold: Amplitude threshold for filtering

    Returns:
        loss: Scalar Fourier loss
    """
    # FFT
    fft_xt = torch.fft.rfft(x_t, dim=-1)
    fft_x0 = torch.fft.rfft(x_0, dim=-1)

    # Filter low-amplitude frequencies
    mask = torch.abs(fft_x0) > threshold
    fft_x0_filtered = fft_x0 * mask

    # MSE in frequency domain
    loss = F.mse_loss(fft_xt, fft_x0_filtered)

    return loss


def fourier_gradient(
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Gradient of Fourier loss w.r.t. x_t.

    Args:
        x_t: Current series [B, L]
        x_0: Original series [B, L]
        threshold: Amplitude threshold

    Returns:
        grad: Gradient [B, L]
    """
    L = x_t.shape[-1]

    # FFT
    fft_xt = torch.fft.rfft(x_t, dim=-1)
    fft_x0 = torch.fft.rfft(x_0, dim=-1)

    # Filter
    mask = torch.abs(fft_x0) > threshold
    fft_x0_filtered = fft_x0 * mask

    # Gradient in frequency domain
    grad_fft = 2.0 * (fft_xt - fft_x0_filtered) / (fft_xt.numel())

    # IFFT back to time domain
    grad = torch.fft.irfft(grad_fft, n=L, dim=-1)

    return grad
