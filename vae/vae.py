import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockDown(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual_connection = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=2, padding=0)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1) # Downsample
        )

    def forward(self, x):
        x = self.conv_block(x) + self.residual_connection(x)
        return F.relu(x)

class ResBlockUp(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual_connection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        x = self.conv_block(x) + self.residual_connection(x)
        return F.relu(x)

class Encoder(nn.Module):

    def __init__(self, in_ch, in_dim, latent_dim):
        super().__init__()
        self.conv_emb = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            ResBlockDown(in_ch=32, out_ch=64),
            ResBlockDown(in_ch=64, out_ch=128),
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
            ResBlockUp(in_ch=128, out_ch=64),
            ResBlockUp(in_ch=64, out_ch=32),
            nn.Conv2d(in_channels=32, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.latent_proj(x).view(x.shape[0], 256, self.pre_deconv_size, self.pre_deconv_size)
        return self.residual_block(h)

class VAE(nn.Module):

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

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return VAE.rsample(mean, logvar)
    
    def decode(self, z):
        return self.decoder(z)

    def loss(self, x):
        enc_mean, enc_logvar = self.encoder(x)
        kl_div = -0.5 * torch.sum(1 + enc_logvar - enc_mean ** 2 - enc_logvar.exp())
        z = VAE.rsample(enc_mean, enc_logvar)
        y = self.decode(z)
        nll = F.mse_loss(input=y, target=x, reduction="sum")
        return (kl_div + nll) / x.shape[0]

    @torch.no_grad()
    def generate(self, batch_size):
        device = next(self.decoder.parameters()).device
        noise = torch.randn(batch_size, self.latent_dim).to(device)
        samples = self.decode(noise)
        return samples

    @staticmethod
    def rsample(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
