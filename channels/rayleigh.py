"""
Rayleigh fading block-diagonal MIMO channel for Blind-MIMOSC.

Generates i.i.d. complex Gaussian channel matrices and applies them:
  Y = sum_i H0_i X_i + N,   N ~ CN(0, sigma_n^2 I)

Shape conventions (using project notation from CLAUDE.md):
  H0  : (batch, Nu, Nr*K, Nt*K)  complex
  X   : (batch, Nu, Nt*K, T)     complex
  Y   : (batch, Nr*K, T)         complex
  N   : (batch, Nr*K, T)         complex
"""
import math
import numpy as np
import torch
from typing import Tuple


def generate_rayleigh_channel(
    batch_size: int,
    Nu: int,
    Nr: int,
    Nt: int,
    K: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate block-diagonal Rayleigh fading channel matrices.

    Each of K blocks is an independent Nr×Nt complex Gaussian realization.
    H0 = blkdiag(H_1, ..., H_K), shape (Nr*K, Nt*K).

    Args:
        batch_size: Number of channel realizations.
        Nu:        Number of users.
        Nr:        Number of receive antennas.
        Nt:        Number of transmit antennas per user.
        K:         Number of transmission blocks.
        device:    Computation device.

    Returns:
        H0: (batch, Nu, Nr*K, Nt*K) complex64 tensor.
    """
    # Independent blocks: (batch, Nu, K, Nr, Nt) ~ CN(0, I)
    real = torch.randn(batch_size, Nu, K, Nr, Nt, device=device)
    imag = torch.randn(batch_size, Nu, K, Nr, Nt, device=device)
    H_blocks = torch.complex(real, imag) / math.sqrt(2)  # unit variance per element

    # Assemble block-diagonal matrix
    H0 = torch.zeros(batch_size, Nu, Nr * K, Nt * K, dtype=torch.complex64, device=device)
    for k in range(K):
        r0, r1 = k * Nr, (k + 1) * Nr
        c0, c1 = k * Nt, (k + 1) * Nt
        H0[:, :, r0:r1, c0:c1] = H_blocks[:, :, k]
    return H0


def apply_channel(
    H0: torch.Tensor,
    X: torch.Tensor,
    snr_db: float,
    power: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    Apply block-fading MIMO channel (Eq. 2 in paper):
        Y = sum_{i=1}^{Nu} H0_i @ X_i + N

    Noise variance is set so that SNR = power / sigma_n^2 matches snr_db.

    Args:
        H0:     (batch, Nu, Nr*K, Nt*K) complex channel.
        X:      (batch, Nu, Nt*K, T) complex transmitted signal.
        snr_db: Target SNR in dB.
        power:  Expected signal power per element.

    Returns:
        Y:       (batch, Nr*K, T) complex received signal.
        sigma_n: Noise standard deviation (scalar float).
    """
    batch_size, Nu, NrK, NtK = H0.shape
    T = X.shape[-1]
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma_n = math.sqrt(power / snr_linear)

    # Aggregate received signal
    HX = torch.zeros(batch_size, NrK, T, dtype=torch.complex64, device=H0.device)
    for i in range(Nu):
        HX = HX + torch.bmm(H0[:, i], X[:, i])  # (batch, NrK, T)

    # Complex AWGN: real and imag each ~ N(0, sigma_n^2/2)
    scale = sigma_n / math.sqrt(2)
    noise = torch.complex(
        torch.randn_like(HX.real) * scale,
        torch.randn_like(HX.imag) * scale,
    )
    Y = HX + noise
    return Y, sigma_n


def apply_channel_with_noise(
    H0: torch.Tensor,
    X: torch.Tensor,
    sigma_n: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply channel with explicit noise sigma (for loss computation).

    Returns:
        Y: (batch, Nr*K, T) received signal.
        N: (batch, Nr*K, T) noise realization.
    """
    batch_size, Nu, NrK, NtK = H0.shape
    T = X.shape[-1]

    HX = torch.zeros(batch_size, NrK, T, dtype=torch.complex64, device=H0.device)
    for i in range(Nu):
        HX = HX + torch.bmm(H0[:, i], X[:, i])

    scale = sigma_n / math.sqrt(2)
    noise = torch.complex(
        torch.randn_like(HX.real) * scale,
        torch.randn_like(HX.imag) * scale,
    )
    Y = HX + noise
    return Y, noise


def compute_snr(
    H0: torch.Tensor,
    X: torch.Tensor,
    N: torch.Tensor,
) -> float:
    """
    Compute empirical SNR in dB (Eq. 52):
        SNR = ||sum_i H0_i X_i||^2_F / ||N||^2_F
    """
    Nu = H0.shape[1]
    HX = sum(torch.bmm(H0[:, i], X[:, i]) for i in range(Nu))
    snr = (HX.abs() ** 2).sum() / ((N.abs() ** 2).sum() + 1e-12)
    return 10.0 * torch.log10(snr).item()


def noise_schedule_exponential(
    sigma_1: float,
    sigma_J: float,
    J: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Exponential noise schedule (Section III-C):
        sigma_j = sigma_1 * (sigma_J / sigma_1)^(j / J),  j = 1, ..., J

    Returns: sigmas of shape (J+1,) where sigmas[0] = 0 and sigmas[j] = sigma_j.
    """
    j_vals = torch.arange(1, J + 1, dtype=torch.float32, device=device)
    sigmas = sigma_1 * (sigma_J / sigma_1) ** (j_vals / J)
    return torch.cat([torch.zeros(1, device=device), sigmas])  # (J+1,)


def lmmse_channel_estimate(
    Y: torch.Tensor,
    X_pilot: torch.Tensor,
    sigma_n: float,
    Nr: int,
    Nt: int,
    K: int,
) -> torch.Tensor:
    """
    LMMSE channel estimator for pilot-based baselines.

    Model: Y_pilot = H0 @ X_pilot + N
    LMMSE under Rayleigh prior (H0 ~ CN(0, I)):
        H_hat = Y_pilot @ X_pilot^H @ (X_pilot @ X_pilot^H + sigma_n^2 I)^{-1}

    Args:
        Y:       (batch, Nr*K, T_pilot) received pilot signal.
        X_pilot: (batch, Nt*K, T_pilot) known pilot matrix.
        sigma_n: Noise std.
        Nr, Nt, K: Channel dimensions.

    Returns:
        H_hat: (batch, Nr*K, Nt*K) estimated channel.
    """
    batch = Y.shape[0]
    NtK = Nt * K
    T_pilot = X_pilot.shape[-1]

    # Gram matrix: (batch, NtK, NtK)
    G = torch.bmm(X_pilot, X_pilot.conj().transpose(-2, -1))
    reg = (sigma_n ** 2) * torch.eye(NtK, dtype=Y.dtype, device=Y.device).unsqueeze(0)
    G_reg = G + reg  # (batch, NtK, NtK)

    # Cross-correlation: (batch, NrK, NtK)
    C = torch.bmm(Y, X_pilot.conj().transpose(-2, -1))

    # Solve: H_hat = C @ G_reg^{-1}
    H_hat = torch.linalg.solve(G_reg.transpose(-2, -1), C.transpose(-2, -1)).transpose(-2, -1)
    return H_hat


def generate_rayleigh_channel_sequence(
    batch_size: int,
    Nu: int,
    Nr: int,
    Nt: int,
    K: int,
    T: int,
    alpha: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate temporally correlated Rayleigh fading channel sequences
    using a complex AR(1) process:

        H_t = alpha * H_{t-1} + sqrt(1 - alpha^2) * W_t

    where W_t ~ CN(0, I) is an independent Rayleigh channel.

    Args:
        batch_size: Number of channel sequences.
        Nu: Number of users.
        Nr: Number of receive antennas.
        Nt: Number of transmit antennas per user.
        K: Number of transmission blocks.
        T: Sequence length.
        alpha: Temporal correlation coefficient.
        device: Computation device.

    Returns:
        H_seq: (batch, T, Nu, Nr*K, Nt*K) complex64 tensor.
    """
    if not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")

    H_seq = torch.empty(
        batch_size,
        T,
        Nu,
        Nr * K,
        Nt * K,
        dtype=torch.complex64,
        device=device,
    )

    # H_0 ~ CN(0, I)
    H_seq[:, 0] = generate_rayleigh_channel(
        batch_size, Nu, Nr, Nt, K, device
    )

    innovation_scale = math.sqrt(1.0 - alpha ** 2)

    for t in range(1, T):
        W_t = generate_rayleigh_channel(
            batch_size, Nu, Nr, Nt, K, device
        )

        H_seq[:, t] = (
            alpha * H_seq[:, t - 1]
            + innovation_scale * W_t
        )

    return H_seq