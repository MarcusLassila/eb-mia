from generative_models.agm import AbstractGenerativeModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from contextlib import nullcontext

class ResBlockDown(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual_connection = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=2, padding=0)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1) # Downsample
        )

    def forward(self, x):
        x = self.conv_block(x) + self.residual_connection(x)
        return x

class ResBlockUp(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual_connection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_ch),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        x = self.conv_block(x) + self.residual_connection(x)
        return x

class Encoder(nn.Module):

    def __init__(self, in_ch, in_dim, latent_dim):
        super().__init__()
        self.conv_emb = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResBlockDown(in_ch=64, out_ch=128),
            ResBlockDown(in_ch=128, out_ch=128),
            ResBlockDown(in_ch=128, out_ch=256),
            nn.Flatten(),
        )
        hidden_dim = 256 * (in_dim // 8) ** 2
        self.lin_mean = nn.Linear(hidden_dim, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.conv_emb(x)
        mean = self.lin_mean(h)
        logvar = self.lin_logvar(h)
        return mean, logvar

class Decoder(nn.Module):

    def __init__(self, out_ch, latent_dim, out_dim):
        super().__init__()
        self.pre_deconv_size = out_dim // 8
        hidden_dim = 256 * self.pre_deconv_size ** 2
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.residual_block = nn.Sequential(
            ResBlockUp(in_ch=256, out_ch=128),
            ResBlockUp(in_ch=128, out_ch=128),
            ResBlockUp(in_ch=128, out_ch=64),
            nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.latent_proj(x).view(x.shape[0], 256, self.pre_deconv_size, self.pre_deconv_size)
        return self.residual_block(h)

class VAE_Network(nn.Module):

    def __init__(self, in_ch, in_dim, latent_dim):
        super().__init__()
        assert in_dim % 8 == 0
        self.in_ch = in_ch
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            in_ch=in_ch,
            in_dim=in_dim,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            out_ch=in_ch,
            latent_dim=latent_dim,
            out_dim=in_dim,
        )

    def forward(self, x, n_rsamples=1):
        mean, logvar = self.encode(x)
        zs = [self.rsample(mean, logvar) for _ in range(n_rsamples)]
        x_hat = [self.decode(z) for z in zs]
        return x_hat, mean, logvar

    def encode(self, x):
        x = (x + 1.0) / 2.0  # Assumes data in [-1,1]
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)
    
    @torch.inference_mode()
    def sample(self, batch_size):
        self.decoder.eval()
        device = next(self.decoder.parameters()).device
        noise = torch.randn(batch_size, self.latent_dim).to(device)
        samples = self.decode(noise)
        return samples
    
    def rsample(self, mean, logvar):
        '''Sample latent variable from encoder mean and log variance using the reparameterization trick.'''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

class VAE(AbstractGenerativeModel):

    def __init__(self, in_ch, in_dim, latent_dim, n_rsamples=1):
        self.network = VAE_Network(
            in_ch=in_ch,
            in_dim=in_dim,
            latent_dim=latent_dim,
        )
        self.n_rsamples = n_rsamples

    def move_to(self, device):
        if isinstance(self.network, DDP):
            assert next(iter(self.network.parameters())).device == torch.device(device), "Should not move a DDP wrapped network to another device"
        else:
            self.network.to(device)

    def per_sample_loss(self, x, network_override=None):
        network = network_override if network_override is not None else self.network
        x_hat, enc_mean, enc_logvar = network(x, n_rsamples=self.n_rsamples)
        kl_div = -0.5 * torch.sum(1 + enc_logvar - enc_mean ** 2 - enc_logvar.exp(), dim=1)
        nll = torch.zeros(x.shape[0], device=x.device)
        for xh in x_hat:
            nll += F.mse_loss(input=xh, target=x, reduction="none").sum(dim=(1, 2, 3))
        nll /= self.n_rsamples
        assert kl_div.shape == nll.shape == (x.shape[0],)
        return kl_div + nll

    def loss(self, x, autocast_context=nullcontext(), network_override=None):
        with autocast_context:
            loss_value = self.per_sample_loss(x, network_override=network_override).mean()
        return loss_value

    @torch.inference_mode()
    def sample(self, batch_size):
        return self.network.sample(batch_size)
