import math
import torch

import torch.nn as nn
import torch.nn.functional as F

def cosine_alpha_bar(T: int, s: float = 0.008, device="cpu"):
    steps = torch.arange(T + 1, device=device, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab.clamp(1e-6, 1.0)

def q_sample_uniform_replace(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, K: int, return_keep: bool = False):
    """
    x0: [B,H,W,D] long
    t:  [B] long in [1..T]
    alpha_bar: [T+1]
    """
    B = x0.shape[0]
    ab = alpha_bar[t].view(B, 1, 1, 1)
    keep = (torch.rand_like(x0.float()) < ab)
    noise = torch.randint(0, K, x0.shape, device=x0.device)
    x_t = torch.where(keep, x0, noise)
    return (x_t, keep) if return_keep else x_t

class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, ch)
        self.gn2 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.gn1(x)))
        h = self.c2(F.silu(self.gn2(h)))
        return x + h

class TokenDenoiser3D(nn.Module):
    def __init__(self, K: int, T: int, d_model: int = 256, n_blocks: int = 6):
        super().__init__()
        self.tok = nn.Embedding(K, d_model)
        self.time = nn.Embedding(T + 1, d_model)
        self.in_conv = nn.Conv3d(d_model, d_model, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock3D(d_model) for _ in range(n_blocks)])
        self.out = nn.Conv3d(d_model, K, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        # x_t: [B,H,W,D] long
        B, H, W, D = x_t.shape
        h = self.tok(x_t).permute(0, 4, 1, 2, 3).contiguous()  # [B,C,H,W,D]
        h = h + self.time(t).view(B, -1, 1, 1, 1)
        h = self.in_conv(h)
        h = self.blocks(h)
        logits = self.out(h)  # [B,K,H,W,D]
        return logits

@torch.no_grad()
@torch.no_grad()
def sample_tokens(
    model,
    T: int,
    K: int,
    shape,
    device,
    alpha_bar,
    allowed=None,        # 1D tensor con ids permitidos
    logit_bias=None,     # 1D tensor [K] o None
    temp_hi=1.25,
    temp_lo=1.05,
    temp_split=0.6,
):
    """
    shape: (B,H,W,D)
    returns x0 tokens [B,H,W,D]
    """
    B, H, W, D = shape
    x = torch.randint(0, K, (B, H, W, D), device=device)

    if allowed is not None:
        allowed = allowed.to(device)
        allowed_mask = torch.zeros(K, dtype=torch.bool, device=device)
        allowed_mask[allowed] = True
    else:
        allowed_mask = None

    if logit_bias is not None:
        logit_bias = logit_bias.to(device).view(1, K, 1, 1, 1)

    for t in range(T, 0, -1):
        tcur = torch.full((B,), t, device=device, dtype=torch.long)
        logits = model(x, tcur)  # [B,K,H,W,D]

        if logit_bias is not None:
            logits = logits + logit_bias

        if allowed_mask is not None:
            # set disallowed logits to -inf
            mask = (~allowed_mask).view(1, K, 1, 1, 1)
            logits = logits.masked_fill(mask, float("-inf"))

        frac = t / float(T)
        temp = temp_hi if frac > temp_split else temp_lo
        p0 = (logits / temp).softmax(dim=1)

        flat = p0.permute(0, 2, 3, 4, 1).contiguous().view(-1, K)
        x0_hat = torch.multinomial(flat, 1).squeeze(1).view(B, H, W, D)

        if t > 1:
            tprev = torch.full((B,), t - 1, device=device, dtype=torch.long)
            x = q_sample_uniform_replace(x0_hat, tprev, alpha_bar, K)
        else:
            x = x0_hat

    return x
