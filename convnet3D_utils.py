"""Core 3D VQ-UNet building blocks used by the PRIM tokenizer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import torch.distributed as dist
except Exception:
    dist = None


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation."""
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.GroupNorm:
    """Group normalization helper."""
    return nn.GroupNorm(
        num_groups=8,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class VectorQuantizer2(nn.Module):
    """
    Vector-quantization layer with optional index remapping and dead-code revival.

    Notes:
        - `legacy=False` uses the corrected loss formulation.
        - `sane_index_shape=True` returns indices with spatial shape [B, H, W, D].
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        remap=None,
        unknown_index="random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ) -> None:
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.register_buffer("usage_ema", torch.zeros(self.n_e))
        self.usage_decay = 0.99
        self.dead_code_threshold = 1.0

        with torch.no_grad():
            self.embedding.weight.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed += 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        """Map raw codebook ids to a reduced set of used ids."""
        original_shape = inds.shape
        assert len(original_shape) > 1

        inds = inds.reshape(original_shape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1

        if self.unknown_index == "random":
            new[unknown] = torch.randint(
                0,
                self.re_embed,
                size=new[unknown].shape,
                device=new.device,
            )
        else:
            new[unknown] = self.unknown_index

        return new.reshape(original_shape)

    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        """Invert remapping and recover original codebook ids."""
        original_shape = inds.shape
        assert len(original_shape) > 1

        inds = inds.reshape(original_shape[0], -1)
        used = self.used.to(inds)

        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0

        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(original_shape)

    def forward(
        self,
        z: torch.Tensor,
        temp=None,
        rescale_logits: bool = False,
        return_logits: bool = False,
    ):
        """
        Quantize latent tensor `z`.

        Args:
            z: Tensor [B, C, H, W, D].

        Returns:
            quantized tensor, commitment/codebook loss, auxiliary info tuple
        """
        assert temp is None or temp == 1.0, "Only for Gumbel-compatible interface"
        assert rescale_logits is False, "Only for Gumbel-compatible interface"
        assert return_logits is False, "Only for Gumbel-compatible interface"

        z = rearrange(z, "b c h w d -> b h w d c").contiguous()
        z_flattened = z.view(-1, self.e_dim)

        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum(
                "bd,dn->bn",
                z_flattened,
                rearrange(self.embedding.weight, "n d -> d n"),
            )
        )

        min_encoding_indices = torch.argmin(distances, dim=1)

        if self.training:
            counts = torch.bincount(
                min_encoding_indices,
                minlength=self.n_e,
            ).float()

            if dist is not None and dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts)

            self.usage_ema.mul_(self.usage_decay).add_(
                counts * (1.0 - self.usage_decay)
            )

            dead = self.usage_ema < self.dead_code_threshold
            if dead.any():
                n_dead = int(dead.sum().item())
                rand_idx = torch.randint(
                    0,
                    z_flattened.shape[0],
                    (n_dead,),
                    device=z_flattened.device,
                )
                with torch.no_grad():
                    self.embedding.weight[dead].copy_(z_flattened[rand_idx])
                    self.usage_ema[dead] = self.dead_code_threshold

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, "b h w d c -> b c h w d").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0],
                z_q.shape[2],
                z_q.shape[3],
                z_q.shape[4],
            )

        encodings = F.one_hot(
            min_encoding_indices.reshape(-1),
            num_classes=self.n_e,
        ).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        )

        return z_q, loss, (perplexity, encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.Tensor, shape) -> torch.Tensor:
        """
        Recover quantized embeddings for the provided token indices.

        Args:
            indices: Flattened or spatial token indices.
            shape: Output view shape (B, H, W, D, C).
        """
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class ResnetBlock(nn.Module):
    """3D residual block."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv3d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class Upsample(nn.Module):
    """Trilinear upsampling optionally followed by a 3D convolution."""

    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        if self.with_conv:
            self.conv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class PositionalEncoding3D(nn.Module):
    """Sinusoidal positional encoding for 3D tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: Tensor of shape [B, X, Y, Z, C].

        Returns:
            Positional encoding with the same shape as `tensor`.
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5D.")

        batch_size, x, y, z, orig_ch = tensor.shape

        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class PositionalEncodingPermute3D(nn.Module):
    """Wrapper that accepts [B, C, X, Y, Z] tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class VQUNet3Dposv3(nn.Module):
    """Main 3D VQ-UNet tokenizer used in the PRIM pipeline."""

    def __init__(
        self,
        num_classes,
        inputchannels: int = 1,
        channels: int = 16,
        dropout: float = 0.0,
        n_embed: int = 1024,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        use_bias = True

        self.conv11 = nn.Conv3d(
            inputchannels,
            channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=use_bias,
        )
        self.conv12 = ResnetBlock(in_channels=channels, out_channels=channels, dropout=dropout)
        self.down1 = nn.Conv3d(
            channels,
            channels * 2,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=use_bias,
        )
        self.conv21 = ResnetBlock(
            in_channels=channels * 2,
            out_channels=channels * 2,
            dropout=dropout,
        )
        self.down2 = nn.Conv3d(
            channels * 2,
            channels * 4,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=use_bias,
        )
        self.conv31 = ResnetBlock(
            in_channels=channels * 4,
            out_channels=channels * 4,
            dropout=dropout,
        )
        self.down3 = nn.Conv3d(
            channels * 4,
            channels * 8,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=use_bias,
        )
        self.conv41 = ResnetBlock(
            in_channels=channels * 8,
            out_channels=channels * 8,
            dropout=dropout,
        )
        self.down4 = nn.Conv3d(
            channels * 8,
            channels * 16,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=use_bias,
        )
        self.conv51 = ResnetBlock(
            in_channels=channels * 16,
            out_channels=channels * 16,
            dropout=dropout,
        )

        self.quant_conv = nn.Conv3d(channels * 16, embed_dim, 1)
        self.p_enc_3d = PositionalEncodingPermute3D(embed_dim)
        self.quantize = VectorQuantizer2(
            n_embed,
            embed_dim,
            beta=0.25,
            sane_index_shape=True,
            legacy=False,
        )

        self.post_quant_conv = nn.Conv3d(embed_dim, channels * 16, 1)
        self.conv52 = ResnetBlock(
            in_channels=channels * 16,
            out_channels=channels * 16,
            dropout=dropout,
        )
        self.up4 = Upsample(channels * 16, True)
        self.conv42 = ResnetBlock(
            in_channels=channels * 16,
            out_channels=channels * 8,
            dropout=dropout,
        )
        self.up3 = Upsample(channels * 8, True)
        self.conv32 = ResnetBlock(
            in_channels=channels * 8,
            out_channels=channels * 4,
            dropout=dropout,
        )
        self.up2 = Upsample(channels * 4, True)
        self.conv22 = ResnetBlock(
            in_channels=channels * 4,
            out_channels=channels * 2,
            dropout=dropout,
        )
        self.up1 = Upsample(channels * 2, True)
        self.conv13 = ResnetBlock(
            in_channels=channels * 2,
            out_channels=channels,
            dropout=dropout,
        )
        self.conv14 = nn.Conv3d(
            channels,
            num_classes,
            kernel_size=1,
            padding=0,
            bias=use_bias,
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.down1(x1)
        x2 = self.conv21(x2)
        x3 = self.down2(x2)
        x3 = self.conv31(x3)
        x4 = self.down3(x3)
        x4 = self.conv41(x4)
        x5 = self.down4(x4)
        x5 = self.conv51(x5)
        x5 = self.quant_conv(x5)
        x5 = x5 + 0.1 * self.p_enc_3d(x5)
        return x5

    def decode_quant(self, quant: torch.Tensor) -> torch.Tensor:
        """Decode a quantized latent tensor into segmentation logits."""
        x5 = self.post_quant_conv(quant)
        x5 = self.conv52(x5)
        x4 = self.up4(x5)
        x4 = self.conv42(x4)
        x3 = self.up3(x4)
        x3 = self.conv32(x3)
        x2 = self.up2(x3)
        x2 = self.conv22(x2)
        x1 = self.up1(x2)
        x1 = self.conv13(x1)
        return self.conv14(x1)

    def forward(self, x: torch.Tensor):
        x5 = self._encode_features(x)
        if self.training:
            x5 = x5 + 0.01 * torch.randn_like(x5)

        quant, emb_loss, info = self.quantize(x5)
        indices = info[2]
        logits = self.decode_quant(quant)
        return logits, emb_loss, quant, indices

    def encode(
        self,
        x: torch.Tensor,
        add_noise: bool = False,
        return_quant: bool = False,
    ):
        """
        Encode input into token indices.

        Args:
            x: Tensor [B, C, H, W, D].
            add_noise: Whether to add Gaussian noise before quantization.
            return_quant: Whether to also return the quantized latent tensor.

        Returns:
            indices or (quant, emb_loss, indices)
        """
        x5 = self._encode_features(x)

        if add_noise:
            x5 = x5 + 0.01 * torch.randn_like(x5)

        quant, emb_loss, info = self.quantize(x5)
        indices = info[2]

        if return_quant:
            return quant, emb_loss, indices
        return indices
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices into segmentation logits.
        """
        if indices.ndim == 3:
            indices = indices.unsqueeze(0)
        if indices.ndim != 4:
            raise ValueError(
                f"indices must have shape [B, Ht, Wt, Dt], got {tuple(indices.shape)}"
            )

        if indices.dtype != torch.long:
            indices = indices.long()

        b, ht, wt, dt = indices.shape
        shape = (b, ht, wt, dt, self.quantize.e_dim)
        device = self.quantize.embedding.weight.device
        indices = indices.to(device)

        quant = self.quantize.get_codebook_entry(indices, shape=shape)
        return self.decode_quant(quant)

    @torch.no_grad()
    def decode_indices_to_seg(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices and return argmax labels [B, 1, H, W, D]."""
        logits = self.decode_indices(indices)
        return torch.argmax(logits, dim=1, keepdim=True)