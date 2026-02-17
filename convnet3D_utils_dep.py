import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from torch import einsum

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w z -> b h w z c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w z c -> b c h w z').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
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

        return x+h

class ResnetBlock2D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
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

        return x+h


class VQenc(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(VQenc, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv3d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv3d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv3d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv3d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock(in_channels=channels*16, out_channels=channels*16, dropout=dropout)
        self.quant_conv = torch.nn.Conv3d(channels*16, embed_dim, 1)
        self.quantize = VectorQuantizer2(n_embed, channels*16, beta=0.25)

    def forward(self, x):
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
        quant, emb_loss, info = self.quantize(x5)


        return emb_loss, quant, x1, x2, x3, x4

class VQdec(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(VQdec, self).__init__()

        use_bias = True

        self.post_quant_conv = torch.nn.Conv3d(embed_dim, channels*16, 1)
        self.conv52 = ResnetBlock(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv42 = ResnetBlock(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = ResnetBlock(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = ResnetBlock(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = ResnetBlock(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv3d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, quant, x1, x2, x3, x4):

        x5 = self.post_quant_conv(quant)
        x5 = self.conv52(x5)
        x4 = torch.cat([self.up4(x5), x4], dim=1)
        x4 = self.conv42(x4)
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = self.conv32(x3)
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = self.conv22(x2)
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = self.conv13(x1)
        x = self.conv14(x1)

        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            

    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
                .unsqueeze(1)
                .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels: 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels

class VQUNet3Dposv3(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(VQUNet3Dposv3, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv3d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv3d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv3d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv3d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock(in_channels=channels*16, out_channels=channels*16, dropout=dropout)
        self.quant_conv = torch.nn.Conv3d(channels*16, embed_dim, 1)
        self.p_enc_3d = PositionalEncodingPermute3D(channels*16)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25, sane_index_shape=True)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, channels*16, 1)
        self.conv52 = ResnetBlock(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 =  Upsample(channels*16, True)
        self.conv42 = ResnetBlock(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 =  Upsample(channels*8, True)
        self.conv32 = ResnetBlock(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = Upsample(channels*4, True)
        self.conv22 = ResnetBlock(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = Upsample(channels*2, True)
        self.conv13 = ResnetBlock(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv3d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
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
        x5 = self.p_enc_3d(x5)
        quant, emb_loss, info = self.quantize(x5)
        indices = info[2]
        x5 = self.post_quant_conv(quant)
        x5 = self.conv52(x5)
        x4 = torch.cat([self.up4(x5), x4], dim=1)
        x4 = self.conv42(x4)
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = self.conv32(x3)
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = self.conv22(x2)
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = self.conv13(x1)
        x = self.conv14(x1)

        return x, emb_loss, quant, indices

class VQUNet3Dposv4(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(VQUNet3Dposv4, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv3d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv3d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv3d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv3d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock(in_channels=channels*16, out_channels=channels*16, dropout=dropout)
        self.quant_conv1 = torch.nn.Conv3d(channels * 8, embed_dim, 1)
        self.p_enc_3d1 = PositionalEncodingPermute3D(channels * 16)
        self.quantize1 = VectorQuantizer2(n_embed*2, embed_dim, beta=0.25)
        self.post_quant_conv1 = torch.nn.Conv3d(embed_dim, channels * 8, 1)
        self.quant_conv2 = torch.nn.Conv3d(channels*16, embed_dim, 1)
        self.p_enc_3d2 = PositionalEncodingPermute3D(channels*16)
        self.quantize2 = VectorQuantizer2(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv2 = torch.nn.Conv3d(embed_dim, channels*16, 1)
        self.conv52 = ResnetBlock(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv42 = ResnetBlock(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = ResnetBlock(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = ResnetBlock(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = ResnetBlock(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv3d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.down1(x1)
        x2 = self.conv21(x2)
        x3 = self.down2(x2)
        x3 = self.conv31(x3)
        x4 = self.down3(x3)
        x4 = self.conv41(x4)
        x4x = self.quant_conv1(x4)
        x4x = self.p_enc_3d1(x4x)
        quant, emb_loss, info = self.quantize1(x4x)
        x4x = self.post_quant_conv1(quant)
        
        x5 = self.down4(x4)
        x5 = self.conv51(x5)
        x5 = self.quant_conv2(x5)
        x5 = self.p_enc_3d2(x5)
        quant1, emb_loss1, info1 = self.quantize2(x5)
        x5 = self.post_quant_conv2(quant1)
        x5 = self.conv52(x5)
        x4 = torch.cat([self.up4(x5), x4x], dim=1)
        x4 = self.conv42(x4)
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = self.conv32(x3)
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = self.conv22(x2)
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = self.conv13(x1)
        x = self.conv14(x1)

        return x, emb_loss, emb_loss1, quant