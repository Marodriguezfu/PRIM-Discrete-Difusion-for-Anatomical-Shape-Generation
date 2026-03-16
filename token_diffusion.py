import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorShape4D = Union[Tuple[int, int, int, int], Sequence[int]]


def _get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """Return a valid GroupNorm group count that divides ``num_channels``."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def cosine_alpha_bar(T: int, s: float = 0.008, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    Compute the cumulative keep probability schedule using the cosine schedule.

    Args:
        T: Number of diffusion steps.
        s: Small offset used in the cosine schedule.
        device: Target device for the returned tensor.

    Returns:
        Tensor of shape ``[T + 1]`` with values in ``[1e-6, 1.0]``.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    steps = torch.arange(T + 1, device=device, dtype=torch.float32)
    schedule = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = schedule / schedule[0]
    return alpha_bar.clamp(1e-6, 1.0)


def q_sample_uniform_replace(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor,
    K: int,
    return_keep: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Corrupt clean tokens by uniformly replacing a subset of positions.

    Args:
        x0: Clean token tensor with shape ``[B, H, W, D]`` and integer dtype.
        t: Diffusion step tensor with shape ``[B]`` and values in ``[1, T]``.
        alpha_bar: Cumulative keep-probability schedule of shape ``[T + 1]``.
        K: Vocabulary size.
        return_keep: Whether to also return the boolean keep mask.

    Returns:
        ``x_t`` or ``(x_t, keep_mask)``.
    """
    if x0.ndim != 4:
        raise ValueError(f"x0 must have shape [B, H, W, D], got {tuple(x0.shape)}.")
    if t.ndim != 1:
        raise ValueError(f"t must have shape [B], got {tuple(t.shape)}.")
    if x0.shape[0] != t.shape[0]:
        raise ValueError("Batch size mismatch between x0 and t.")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")

    batch_size = x0.shape[0]
    keep_prob = alpha_bar[t].view(batch_size, 1, 1, 1)
    keep_mask = torch.rand_like(x0, dtype=torch.float32) < keep_prob
    random_tokens = torch.randint(0, K, x0.shape, device=x0.device)
    x_t = torch.where(keep_mask, x0, random_tokens)
    return (x_t, keep_mask) if return_keep else x_t


class ResBlock3D(nn.Module):
    """Simple residual 3D convolutional block with GroupNorm and SiLU."""

    def __init__(self, channels: int):
        super().__init__()
        num_groups = _get_num_groups(channels)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


class TokenDenoiser3D(nn.Module):
    """3D token denoiser operating on discrete latent volumes."""

    def __init__(self, K: int, T: int, d_model: int = 256, n_blocks: int = 6):
        super().__init__()
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}.")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}.")

        self.token_embedding = nn.Embedding(K, d_model)
        self.time_embedding = nn.Embedding(T + 1, d_model)
        self.input_conv = nn.Conv3d(d_model, d_model, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock3D(d_model) for _ in range(n_blocks)])
        self.output_conv = nn.Conv3d(d_model, K, kernel_size=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy token tensor with shape ``[B, H, W, D]``.
            t: Diffusion step tensor with shape ``[B]``.

        Returns:
            Logits over the token vocabulary with shape ``[B, K, H, W, D]``.
        """
        if x_t.ndim != 4:
            raise ValueError(f"x_t must have shape [B, H, W, D], got {tuple(x_t.shape)}.")
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}.")
        if x_t.shape[0] != t.shape[0]:
            raise ValueError("Batch size mismatch between x_t and t.")

        batch_size = x_t.shape[0]
        x = self.token_embedding(x_t).permute(0, 4, 1, 2, 3).contiguous()
        x = x + self.time_embedding(t).view(batch_size, -1, 1, 1, 1)
        x = self.input_conv(x)
        x = self.blocks(x)
        return self.output_conv(x)


@torch.no_grad()
def sample_tokens(
    model: nn.Module,
    T: int,
    K: int,
    shape: TensorShape4D,
    device: Union[str, torch.device],
    alpha_bar: torch.Tensor,
    allowed: Optional[torch.Tensor] = None,
    logit_bias: Optional[torch.Tensor] = None,
    temp_hi: float = 1.25,
    temp_lo: float = 1.05,
    temp_split: float = 0.6,
) -> torch.Tensor:
    """
    Sample a discrete token volume using iterative denoising.

    Args:
        model: Denoising model returning logits of shape ``[B, K, H, W, D]``.
        T: Number of reverse diffusion steps.
        K: Vocabulary size.
        shape: Output token shape ``(B, H, W, D)``.
        device: Device on which sampling will run.
        alpha_bar: Cumulative keep-probability schedule of shape ``[T + 1]``.
        allowed: Optional 1D tensor containing the only valid token ids.
        logit_bias: Optional bias tensor of shape ``[K]`` added to logits.
        temp_hi: Temperature used in earlier reverse steps.
        temp_lo: Temperature used in later reverse steps.
        temp_split: Fraction of the trajectory where the high temperature is used.

    Returns:
        Sampled token tensor with shape ``[B, H, W, D]``.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if len(shape) != 4:
        raise ValueError(f"shape must be a 4D tuple/list, got {shape}.")
    if not 0.0 <= temp_split <= 1.0:
        raise ValueError(f"temp_split must be in [0, 1], got {temp_split}.")

    batch_size, height, width, depth = map(int, shape)
    x = torch.randint(0, K, (batch_size, height, width, depth), device=device)

    allowed_mask = None
    if allowed is not None:
        allowed = allowed.to(device=device, dtype=torch.long).view(-1)
        if allowed.numel() == 0:
            raise ValueError("allowed cannot be empty.")
        if torch.any((allowed < 0) | (allowed >= K)):
            raise ValueError("allowed contains token ids outside [0, K).")
        allowed_mask = torch.zeros(K, dtype=torch.bool, device=device)
        allowed_mask[allowed] = True

    if logit_bias is not None:
        logit_bias = logit_bias.to(device=device, dtype=torch.float32)
        if logit_bias.numel() != K:
            raise ValueError(f"logit_bias must contain exactly K={K} elements.")
        logit_bias = logit_bias.view(1, K, 1, 1, 1)

    for step in range(T, 0, -1):
        current_t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        logits = model(x, current_t)

        if logit_bias is not None:
            logits = logits + logit_bias

        if allowed_mask is not None:
            invalid_mask = (~allowed_mask).view(1, K, 1, 1, 1)
            logits = logits.masked_fill(invalid_mask, float("-inf"))

        progress = step / float(T)
        temperature = temp_hi if progress > temp_split else temp_lo
        probs = (logits / temperature).softmax(dim=1)

        flat_probs = probs.permute(0, 2, 3, 4, 1).contiguous().view(-1, K)
        x0_hat = torch.multinomial(flat_probs, 1).squeeze(1).view(batch_size, height, width, depth)

        if step > 1:
            prev_t = torch.full((batch_size,), step - 1, device=device, dtype=torch.long)
            x = q_sample_uniform_replace(x0_hat, prev_t, alpha_bar, K)
        else:
            x = x0_hat

    return x
