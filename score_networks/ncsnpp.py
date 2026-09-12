"""
NCSN++ score network for Blind-MIMOSC.

Used for:
  - Image score network q_{θ_D}: inputs real image, outputs score ∇_D ln q(D | σ)
  - Channel score network q_{θ_H}: inputs real/imag parts of H, outputs score

Architecture: NCSN++ (Song et al., 2021) — U-Net with residual blocks,
self-attention, and Fourier/positional time embedding.

Reference: Song et al., "Score-Based Generative Modeling through
Stochastic Differential Equations." ICLR 2021.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Time (noise level) embedding
# ---------------------------------------------------------------------------

def get_timestep_embedding(sigmas: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal + Fourier embedding for noise level σ.
    sigmas: (B,) tensor of noise levels.
    Returns: (B, embedding_dim) embeddings.
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=sigmas.device) / (half - 1)
    )
    args = sigmas[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, base_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.base_dim = base_dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        emb = get_timestep_embedding(sigma, self.base_dim)
        return self.net(emb)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with GroupNorm and optional time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 0,
                 num_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_proj = nn.Linear(time_dim, out_ch) if time_dim > 0 else None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.dropout(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttnBlock(nn.Module):
    """Self-attention at a single resolution."""

    def __init__(self, ch: int, num_heads: int = 1, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ---------------------------------------------------------------------------
# NCSN++ U-Net
# ---------------------------------------------------------------------------

class NCSNpp(nn.Module):
    """
    NCSN++ score network (simplified U-Net variant).

    Predicts the score s_θ(x, σ) ≈ ∇_x ln q_σ(x) = (x̂₀ - x) / σ²

    The network is conditioned on the noise level σ via time embedding.

    Args:
        in_channels:   Number of input channels.
        base_channels: Base channel count (default 128).
        ch_mults:      Channel multipliers per resolution level.
        num_res_blocks: Residual blocks per level.
        attn_resolutions: Spatial resolutions at which to apply self-attention.
        dropout:       Dropout probability.
        img_size:      Spatial size of the input (assumed square).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        ch_mults: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        img_size: int = 256,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_emb = TimeEmbedding(base_channels, time_dim)

        chs = [base_channels * m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.enc_blocks = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        ch_in = base_channels
        self._enc_chs = [ch_in]
        res = img_size
        for i, ch_out in enumerate(chs):
            for _ in range(num_res_blocks):
                self.enc_blocks.append(ResBlock(ch_in, ch_out, time_dim, dropout=dropout))
                if res in attn_resolutions:
                    self.enc_blocks.append(SelfAttnBlock(ch_out))
                else:
                    self.enc_blocks.append(nn.Identity())
                ch_in = ch_out
                self._enc_chs.append(ch_in)
            if i < len(chs) - 1:
                self.enc_down.append(Downsample(ch_in))
                res //= 2
            else:
                self.enc_down.append(nn.Identity())

        # Bottleneck
        self.mid1 = ResBlock(ch_in, ch_in, time_dim, dropout=dropout)
        self.mid_attn = SelfAttnBlock(ch_in)
        self.mid2 = ResBlock(ch_in, ch_in, time_dim, dropout=dropout)

        # Decoder (upsampling path)
        self.dec_blocks = nn.ModuleList()
        self.dec_up = nn.ModuleList()
        enc_chs_rev = list(reversed(self._enc_chs))
        chs_rev = list(reversed(chs))
        for i, ch_out in enumerate(chs_rev):
            skip_ch = enc_chs_rev[i]
            for j in range(num_res_blocks + 1):
                in_ch = ch_in + skip_ch if j == 0 else ch_out
                self.dec_blocks.append(ResBlock(in_ch, ch_out, time_dim, dropout=dropout))
                if res in attn_resolutions:
                    self.dec_blocks.append(SelfAttnBlock(ch_out))
                else:
                    self.dec_blocks.append(nn.Identity())
                ch_in = ch_out
            if i < len(chs_rev) - 1:
                self.dec_up.append(Upsample(ch_in))
                res *= 2
            else:
                self.dec_up.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(32, ch_in), ch_in)
        self.out_conv = nn.Conv2d(ch_in, in_channels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W) noisy input.
            sigma: (B,) noise level.

        Returns:
            score: (B, C, H, W)  ≈ (x̂₀ - x) / σ²
        """
        t_emb = self.time_emb(sigma)
        h = self.in_conv(x)

        # Encoder
        skips = [h]
        block_idx = 0
        for i, down in enumerate(self.enc_down):
            for _ in range(len([b for b in self.enc_blocks]) // len(self.enc_down) // 2):
                if block_idx >= len(self.enc_blocks):
                    break
                blk = self.enc_blocks[block_idx]
                h = blk(h, t_emb) if isinstance(blk, ResBlock) else blk(h)
                block_idx += 1
                blk2 = self.enc_blocks[block_idx]
                h = blk2(h) if not isinstance(blk2, ResBlock) else blk2(h, t_emb)
                block_idx += 1
                skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Decoder
        block_idx2 = 0
        for i, up in enumerate(self.dec_up):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for _ in range(len([b for b in self.dec_blocks]) // len(self.dec_up) // 2):
                if block_idx2 >= len(self.dec_blocks):
                    break
                blk = self.dec_blocks[block_idx2]
                h = blk(h, t_emb) if isinstance(blk, ResBlock) else blk(h)
                block_idx2 += 1
                blk2 = self.dec_blocks[block_idx2]
                h = blk2(h) if not isinstance(blk2, ResBlock) else blk2(h, t_emb)
                block_idx2 += 1
            h = up(h)

        h = F.silu(self.out_norm(h))
        # Epsilon predictor: output ≈ -eps (NOT divided by sigma).
        # Training target is -eps with sigma² loss weighting.
        # Tweedie at inference: D_hat = D_j + sigma * net_output.
        # Actual score = net_output / sigma (computed externally when needed).
        eps_pred = self.out_conv(h)
        return eps_pred


def get_sigmas(sigma_min: float, sigma_max: float, num_steps: int,
               device: torch.device) -> torch.Tensor:
    """Geometric sequence of noise levels."""
    return torch.exp(
        torch.linspace(math.log(sigma_min), math.log(sigma_max), num_steps, device=device)
    )


# ---------------------------------------------------------------------------
# Separate lightweight score net for channels (smaller input)
# ---------------------------------------------------------------------------

class ChannelScoreNet(nn.Module):
    """
    Score network for MIMO channel H.

    H is complex-valued (Nr*K × Nt*K). We treat real and imag parts as
    two input channels and flatten spatial dims to (Nr*K, Nt*K) as a 2D image.

    For small channel matrices, we use a simple fully-connected residual net.
    For large matrices (e.g. 128×16), we use a convolutional net.
    """

    def __init__(
        self,
        Nr: int,
        Nt: int,
        K: int,
        hidden_dim: int = 1024,
        num_layers: int = 8,
        time_dim: int = 512,
    ):
        super().__init__()
        self.NrK = Nr * K
        self.NtK = Nt * K
        self.in_dim = 2 * Nr * K * Nt * K  # real + imag flattened

        self.time_emb = TimeEmbedding(time_dim, time_dim)

        layers = []
        in_d = self.in_dim + time_dim
        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else self.in_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, H: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H:     (B, 2, NrK, NtK) tensor (real/imag stacked).
            sigma: (B,) noise level.

        Returns:
            score: (B, 2, NrK, NtK) score.
        """
        B = H.shape[0]
        h_flat = H.reshape(B, -1)
        t_emb = self.time_emb(sigma)
        inp = torch.cat([h_flat, t_emb], dim=-1)
        out = self.net(inp)
        # score = out.view_as(H)
        # # Score parameterization: divide by sigma^2
        # score = score / (sigma[:, None, None, None] ** 2 + 1e-8)
        return out.view_as(H)

class ChannelTemporalScoreNet(nn.Module):
    """
    Temporal score network for MIMO channel H.

    Models the conditional distribution:

        q(H_t | H_{t-1})

    The network receives:
      - H_t:     noisy current channel
      - H_prev:  previous clean channel
      - sigma:   noise level applied to H_t

    H_t and H_prev are represented using their real and imaginary parts
    as two channels.

    The network predicts -epsilon, matching the parameterization used
    by ChannelScoreNet.
    """

    def __init__(
        self,
        Nr: int,
        Nt: int,
        K: int,
        hidden_dim: int = 1024,
        num_layers: int = 8,
        time_dim: int = 512,
    ):
        super().__init__()

        self.NrK = Nr * K
        self.NtK = Nt * K

        # One complex H represented as:
        #   2 x NrK x NtK
        #
        # Therefore:
        #   h_dim = real + imag
        self.h_dim = 2 * Nr * K * Nt * K

        # We feed both H_t and H_{t-1}.
        self.in_dim = 2 * self.h_dim

        self.time_emb = TimeEmbedding(time_dim, time_dim)

        layers = []
        in_d = self.in_dim + time_dim

        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else self.h_dim
            layers.append(nn.Linear(in_d, out_d))

            if i < num_layers - 1:
                layers.append(nn.SiLU())

            in_d = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        H: torch.Tensor,
        H_prev: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            H:
                Noisy current channel.

                Shape:
                    (B, 2, NrK, NtK)

                Channel 0 = real part
                Channel 1 = imaginary part

            H_prev:
                Previous clean channel.

                Shape:
                    (B, 2, NrK, NtK)

            sigma:
                Noise level.

                Shape:
                    (B,)

        Returns:
            Predicted -epsilon with shape:

                (B, 2, NrK, NtK)
        """

        B = H.shape[0]

        # Flatten current and previous channels independently.
        h_flat = H.reshape(B, -1)
        h_prev_flat = H_prev.reshape(B, -1)

        # Encode the noise level.
        t_emb = self.time_emb(sigma)

        # Conditional input:
        #
        #   [ H_t | H_{t-1} | sigma_embedding ]
        #
        inp = torch.cat(
            [h_flat, h_prev_flat, t_emb],
            dim=-1,
        )

        out = self.net(inp)

        # Linear layer outputs a flat vector.
        # Restore the channel representation expected by the DSM loss.
        return out.view_as(H)



class ChannelScoreNet2ndOrder(nn.Module):
    """
    Second-order trace score network for channel H.
    Outputs tr(∇²_H ln q(H | σ)) — a scalar per batch element.
    """

    def __init__(
        self,
        Nr: int,
        Nt: int,
        K: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_dim: int = 128,
    ):
        super().__init__()
        self.in_dim = 2 * Nr * K * Nt * K
        self.time_emb = TimeEmbedding(time_dim, time_dim)
        layers = []
        in_d = self.in_dim + time_dim
        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, H: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Returns (B,) scalar trace scores."""
        B = H.shape[0]
        h_flat = H.view(B, -1)
        t_emb = self.time_emb(sigma)
        inp = torch.cat([h_flat, t_emb], dim=-1)
        return self.net(inp).squeeze(-1)


class ImageScoreNet2ndOrder(nn.Module):
    """
    Second-order trace score network for image D.
    Outputs tr(∇²_D ln q(D | σ)) — a scalar per batch element.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        img_size: int = 256,
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim, time_dim)
        # Simple CNN that maps image → scalar
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),  # 128
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 64
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 32
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1),  # 16
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        feat_dim = base_channels * 4 * 16
        self.head = nn.Sequential(
            nn.Linear(feat_dim + time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def forward(self, D: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            D:     (B, C, H, W) image.
            sigma: (B,) noise level.
        Returns:
            trace: (B,) scalar.
        """
        B = D.shape[0]
        feat = self.encoder(D).view(B, -1)
        t_emb = self.time_emb(sigma)
        return self.head(torch.cat([feat, t_emb], dim=-1)).squeeze(-1)
