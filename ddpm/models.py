import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_sinusoidal_positional_embeddings(t, emb_dim):
    '''Fairseq implementation of sinusoidal positional embedding'''
    n_emb = t.shape[0]
    half_dim = emb_dim // 2
    max_positions = int(1e5)
    emb = math.log(max_positions) / (half_dim - 1) # arange(half_dim) / (half_dim - 1) = [0,...,1] 
    emb = torch.exp(torch.arange(half_dim) * -emb).to(t.device)
    emb = torch.outer(t, emb)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if emb_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(n_emb, 1)], dim=1) # Pad with zeros
    assert emb.shape == (n_emb, emb_dim)
    return emb

def group_norm(channels, n_groups=32):
    return nn.GroupNorm(num_groups=n_groups, num_channels=channels)

class AttentionBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = group_norm(channels)
        self.attn = NIN(in_channels=channels, out_channels=3*channels)
        self.proj_out = NIN(in_channels=channels, out_channels=channels)

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.norm(x)
        q, k, v = self.attn(y).split(self.channels, dim=1)
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).transpose(1, 2)
        # Maybe use flash attention?
        w = torch.bmm(q, k) * (C ** -0.5)
        w = F.softmax(w, dim=-1)
        y = torch.bmm(w, v).transpose(1, 2).contiguous().view(B, C, H, W)
        y = self.proj_out(y)
        return x + y

class NIN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        return self.layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class Downsample(nn.Module):

    def __init__(self, channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.layer = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        else:
            self.layer = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)
    
class Upsample(nn.Module):

    def __init__(self, channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if hasattr(self, "conv"):
            x = self.conv(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, t_emb_channels, dropout=0.0, conv_shortcut=False):
        super().__init__()
        self.norm_1 = group_norm(in_channels)
        self.norm_2 = group_norm(out_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.t_emb_proj = nn.Linear(in_features=t_emb_channels, out_features=out_channels)
        self.dropout = dropout

        if in_channels == out_channels:
            self.res_connection = nn.Identity()
        elif conv_shortcut:
            self.res_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        else:
            self.res_connection = NIN(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, t_emb):
        y = F.silu(self.norm_1(x))
        y = self.conv_1(y)
        y = y + self.t_emb_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        y = F.silu(self.norm_2(y))
        y = F.dropout(y, p=self.dropout)
        y = self.conv_2(y)
        y = y + self.res_connection(x)
        return y

class UNet(nn.Module):
    '''U-net following the DDPM paper'''

    def __init__(self,
                 image_size, # assume square images with dim = (size, size)
                 in_channels,
                 out_channels,
                 base_channels,
                 channel_mult,
                 dropout=0.0,
                 resample_with_conv=True,
        ):
        super().__init__()
        if image_size == 32:
            rescalings_to_16_res = 1
        elif image_size == 256:
            rescalings_to_16_res = 4
        else:
            raise ValueError("Only image size 32x32 and 256x256 are supported")
        n_res_blocks = 2

        self.dropout = dropout
        self.base_channels = base_channels
        self.channel_mult = channel_mult

        self.t_emb_channels = 4 * base_channels
        self.t_emb_proj = nn.Sequential(
            nn.Linear(base_channels, self.t_emb_channels),
            nn.SiLU(),
            nn.Linear(self.t_emb_channels, self.t_emb_channels)
        )

        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1)
        )

        self.encoder_blocks = nn.ModuleList()
        prev_channels = base_channels
        rescalings = 0
        channel_stack = [prev_channels]
        for lvl, multiplier in enumerate(channel_mult):
            block = nn.ModuleList()
            curr_channels = base_channels * multiplier
            for _ in range(n_res_blocks):
                res_block = ResBlock(
                    in_channels=prev_channels,
                    out_channels=curr_channels,
                    t_emb_channels=self.t_emb_channels,
                    dropout=dropout,
                )
                channel_stack.append(curr_channels)
                block.append(res_block)
                prev_channels = curr_channels
                if rescalings == rescalings_to_16_res:
                    block.append(AttentionBlock(curr_channels))
            if lvl < len(channel_mult) - 1:
                block.append(Downsample(curr_channels, use_conv=resample_with_conv))
                channel_stack.append(curr_channels)
                rescalings += 1
            self.encoder_blocks.append(block)

        self.mid_block = nn.ModuleList([
            ResBlock(
                in_channels=curr_channels,
                out_channels=curr_channels,
                t_emb_channels=self.t_emb_channels,
                dropout=dropout,
            ),
            AttentionBlock(channels=curr_channels),
            ResBlock(
                in_channels=curr_channels,
                out_channels=curr_channels,
                t_emb_channels=self.t_emb_channels,
                dropout=dropout,
            ),
        ])

        self.decoder_blocks = nn.ModuleList()
        prev_channels = curr_channels
        for lvl, mult in enumerate(reversed(channel_mult)):
            block = nn.ModuleList()
            curr_channels = base_channels * mult
            for _ in range(n_res_blocks + 1):
                in_channels = prev_channels + channel_stack.pop()
                res_block = ResBlock(
                    in_channels=in_channels,
                    out_channels=curr_channels,
                    t_emb_channels=self.t_emb_channels,
                    dropout=dropout,
                )
                block.append(res_block)
                prev_channels = curr_channels
                if rescalings == rescalings_to_16_res:
                    block.append(AttentionBlock(curr_channels))
            if lvl < len(channel_mult) - 1:
                block.append(Upsample(curr_channels, use_conv=resample_with_conv))
                rescalings -= 1
            self.decoder_blocks.append(block)

        assert not channel_stack

        self.out_block = nn.Sequential(
            group_norm(curr_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=curr_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = get_sinusoidal_positional_embeddings(t, self.base_channels)
        t_emb = self.t_emb_proj(t_emb)

        h = self.in_block(x)
        signals = [h]
        for block in self.encoder_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    signals.append(layer(signals[-1], t_emb))
                elif isinstance(layer, Downsample):
                    signals.append(layer(signals[-1]))
                else:
                    signals[-1] = layer(signals[-1])

        h = signals[-1]
        for layer in self.mid_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        for block in self.decoder_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, signals.pop()], dim=1)
                    h = layer(h, t_emb)
                else:
                    h = layer(h)

        assert not signals

        h = self.out_block(h)
        return h
