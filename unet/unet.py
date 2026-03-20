import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianFourierEmbedding(nn.Module):

    def __init__(self, emb_dim=512, scale=16.0):
        super().__init__()
        assert emb_dim % 2 == 0
        self.scale = scale
        self.W = nn.Parameter(torch.randn(emb_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        x = t[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

def get_sinusoidal_positional_embeddings(t, emb_dim, max_positions=10000):
    '''
    Return Fairseq-style sinusoidal positional embeddings.
    Args:
        t (torch.Tensor): Input timesteps.
        emb_dim (int): Embedding dimension.
        max_positions (int): Maximum position scale.
    Returns:
        torch.Tensor: Positional embeddings for `t`.
    '''
    n_emb = t.shape[0]
    half_dim = emb_dim // 2
    emb = math.log(max_positions) / (half_dim - 1) # arange(half_dim) / (half_dim - 1) = [0,...,1]
    emb = torch.exp(torch.arange(half_dim) * -emb).to(t.device)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if emb_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(n_emb, 1)], dim=-1) # Pad with zeros
    assert emb.shape == (n_emb, emb_dim)
    return emb

def group_norm(channels, n_groups=32):
    return nn.GroupNorm(num_groups=n_groups, num_channels=channels)

def zero_params(module):
    '''
    Set the parameters to zero and return the module.
    Args:
        module (nn.Module): Module whose parameters are zeroed.
    Returns:
        nn.Module: The same module with zeroed parameters.
    '''
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttentionBlock(nn.Module):

    def __init__(self, channels, n_heads=None, channels_per_head=None, use_sdpa=True):
        super().__init__()
        self.channels = channels
        if n_heads is None:
            assert channels % channels_per_head == 0
            self.n_heads = channels // channels_per_head
            self.channels_per_head = channels_per_head
        else:
            assert channels_per_head is None
            assert channels % n_heads == 0
            self.n_heads = n_heads
            self.channels_per_head = channels // n_heads
        self.norm = group_norm(channels)
        self.qkv = NIN(dim=1, in_channels=channels, out_channels=3*channels)
        self.proj_out = zero_params(NIN(dim=1, in_channels=channels, out_channels=channels))
        self.use_sdpa = use_sdpa

    def forward(self, x):
        B, C, *spatial = x.shape
        T = math.prod(spatial)
        x = x.reshape(B, C, T)
        h = self.norm(x)
        q, k, v = tuple(
            w.view(B, self.n_heads, self.channels_per_head, T).transpose(2, 3)
            for w in self.qkv(h).chunk(3, dim=1)
        )
        if self.use_sdpa:
            h = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(2, 3).contiguous().view(B, C, T)
        else:
            scale = self.channels_per_head ** -0.25
            qk = torch.einsum(
                "bhct,bhcs->bhts",
                q * scale,
                k * scale,
            )
            attn_coef = torch.softmax(qk.float(), dim=-1).type(qk.dtype) # apply softmax on float32 for numerical stability
            h = torch.einsum(
                "bhts,bhcs->bhct",
                attn_coef,
                v,
            ).contiguous().view(B, C, T)
        h = self.proj_out(h)
        return (x + h).reshape(B, C, *spatial)

class NIN(nn.Module):

    def __init__(self, dim, in_channels, out_channels):
        super().__init__()
        self.dim = dim
        if dim == 1:
            self.layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        elif dim == 2:
            self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1).to(memory_format=torch.channels_last)
        else:
            raise ValueError("NIN only supports 1 and 2 dimensions!")

    def forward(self, x):
        if self.dim == 2 and x.dim() == 4 and x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.layer(x)

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

    def __init__(self, in_channels, out_channels, t_emb_dim, dropout=0.0, conv_shortcut=False):
        super().__init__()
        self.norm_1 = group_norm(in_channels)
        self.layer_1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
        self.norm_2 = group_norm(out_channels)
        self.layer_2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_params(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)),
        )
        self.t_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=2*out_channels),
        )
        if in_channels == out_channels:
            self.res_connection = nn.Identity()
        elif conv_shortcut:
            # This may cause issues with grad-layout when using torch compile
            self.res_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        else:
            self.res_connection = NIN(dim=2, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, t_emb):
        h = self.layer_1(self.norm_1(x))
        t_emb_out = self.t_emb_proj(t_emb)
        while len(t_emb_out.shape) < len(h.shape):
            t_emb_out = t_emb_out.unsqueeze(-1)
        scale, shift = torch.chunk(t_emb_out, 2, axis=1)
        h = self.norm_2(h) * (1 + scale) + shift
        h = self.layer_2(h)
        return h + self.res_connection(x)

class UNet(nn.Module):

    def __init__(self,
                 image_size,
                 in_channels,
                 out_channels,
                 base_channels,
                 channel_mult,
                 n_res_blocks_per_level=2,
                 n_attention_heads=1,
                 channels_per_head=None,
                 attention_resolutions=(16,),
                 dropout=0.0,
                 resample_with_conv=True,
                 continuous_time=False,
                 use_sdpa=True, # to use F.scaled_dot_product_attention
        ):
        super().__init__()
        assert image_size.bit_count() == 1
        assert all(x.bit_count() == 1 for x in attention_resolutions)
        attention_levels = [int(math.log2(image_size) - math.log2(x)) for x in attention_resolutions]
        self.n_res_blocks_per_level = n_res_blocks_per_level

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult

        assert (n_attention_heads is None) + (channels_per_head is None) == 1
        self.n_attention_heads = n_attention_heads
        self.channels_per_head = channels_per_head
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.continuous_time = continuous_time
        self.use_sdpa = use_sdpa

        t_emb_dim = 4 * base_channels
        t_emb_dim_in = t_emb_dim if self.continuous_time else base_channels
        if continuous_time:
            self.gaussian_fourier_emb = GaussianFourierEmbedding(emb_dim=t_emb_dim_in)

        self.t_emb_proj = nn.Sequential(
            nn.Linear(t_emb_dim_in, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )

        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1)
        )

        self.encoder_modules = nn.ModuleList()
        prev_channels = base_channels
        rescale_counter = 0
        channel_stack = [prev_channels]
        for lvl, multiplier in enumerate(channel_mult):
            curr_channels = base_channels * multiplier
            for _ in range(n_res_blocks_per_level):
                res_block = ResBlock(
                    in_channels=prev_channels,
                    out_channels=curr_channels,
                    t_emb_dim=t_emb_dim,
                    dropout=dropout,
                )
                channel_stack.append(curr_channels)
                self.encoder_modules.append(res_block)
                prev_channels = curr_channels
                if rescale_counter in attention_levels:
                    self.encoder_modules.append(
                        AttentionBlock(
                            channels=curr_channels,
                            n_heads=self.n_attention_heads,
                            channels_per_head=self.channels_per_head,
                            use_sdpa=self.use_sdpa,
                        )
                    )
            if lvl < len(channel_mult) - 1:
                self.encoder_modules.append(Downsample(curr_channels, use_conv=resample_with_conv))
                channel_stack.append(curr_channels)
                rescale_counter += 1

        self.mid_block = nn.ModuleList([
            ResBlock(
                in_channels=curr_channels,
                out_channels=curr_channels,
                t_emb_dim=t_emb_dim,
                dropout=dropout,
            ),
            AttentionBlock(
                channels=curr_channels,
                n_heads=self.n_attention_heads,
                channels_per_head=self.channels_per_head,
                use_sdpa=self.use_sdpa,
            ),
            ResBlock(
                in_channels=curr_channels,
                out_channels=curr_channels,
                t_emb_dim=t_emb_dim,
                dropout=dropout,
            ),
        ])

        self.decoder_modules = nn.ModuleList()
        prev_channels = curr_channels
        for lvl, mult in enumerate(reversed(channel_mult)):
            curr_channels = base_channels * mult
            for _ in range(n_res_blocks_per_level + 1):
                in_channels = prev_channels + channel_stack.pop()
                res_block = ResBlock(
                    in_channels=in_channels,
                    out_channels=curr_channels,
                    t_emb_dim=t_emb_dim,
                    dropout=dropout,
                )
                self.decoder_modules.append(res_block)
                prev_channels = curr_channels
                if rescale_counter == attention_levels:
                    self.decoder_modules.append(
                        AttentionBlock(
                            channels=curr_channels,
                            n_heads=self.n_attention_heads,
                            channels_per_head=self.channels_per_head,
                            use_sdpa=self.use_sdpa,
                        )
                    )
            if lvl < len(channel_mult) - 1:
                self.decoder_modules.append(Upsample(curr_channels, use_conv=resample_with_conv))
                rescale_counter -= 1

        assert not channel_stack

        self.out_block = nn.Sequential(
            group_norm(curr_channels),
            nn.SiLU(),
            zero_params(nn.Conv2d(in_channels=curr_channels, out_channels=out_channels, kernel_size=3, padding=1))
        )

    def forward(self, x, t):
        if self.continuous_time:
            t_emb = self.gaussian_fourier_emb(t)
        else:
            t_emb = get_sinusoidal_positional_embeddings(t, self.base_channels)
        t_emb = self.t_emb_proj(t_emb)

        h = self.in_block(x)
        hs = [h]
        for module in self.encoder_modules:
            if isinstance(module, ResBlock):
                hs.append(module(hs[-1], t_emb))
            elif isinstance(module, Downsample):
                hs.append(module(hs[-1]))
            else:
                hs[-1] = module(hs[-1])

        h = hs[-1]
        for module in self.mid_block:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)

        for module in self.decoder_modules:
            if isinstance(module, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)

        assert not hs

        h = self.out_block(h)
        return h
