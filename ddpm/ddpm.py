from agm.agm import AbstractGenerativeModel
from unet.unet import UNet

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from contextlib import nullcontext


class DDPM(AbstractGenerativeModel):

    def __init__(
        self,
        beta,
        channel_mult,
        image_dim,
        base_channels=128,
        dropout=0.0,
        resample_with_conv=True,
    ):
        super().__init__()
        self.image_dim = image_dim
        assert self.image_dim[1] == self.image_dim[2], "Only square images are supported"
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.T = beta.shape[0]
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, dim=0))
        self.model = UNet(
            image_size=self.image_dim[1],
            in_channels=self.image_dim[0],
            out_channels=self.image_dim[0],
            base_channels=base_channels,
            channel_mult=channel_mult,
            dropout=dropout,
            resample_with_conv=resample_with_conv,
        )

    def forward(self, x, t):
        return self.model.forward(x, t)

    def per_sample_loss(self, x, t, autocast_context=nullcontext()):
        eps = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
        with autocast_context:
            noise_pred = self.model(z, t)
            loss = F.mse_loss(input=noise_pred, target=eps, reduction="none").mean(dim=(1, 2, 3))
        return loss
    
    def loss(self, x, autocast_context=nullcontext()):
        t = torch.randint(low=0, high=self.T, size=(x.shape[0],), device=x.device)
        return self.per_sample_loss(x, t, autocast_context).mean()

    @torch.inference_mode()
    def sample(self, batch_size, disable_tqdm=False):
        ''' Sample a batch of images and rescale them to floating point values in [0,1]. '''
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        x = torch.randn(batch_size, *self.image_dim).to(device)
        for t in tqdm(range(self.T - 1, -1, -1), disable=disable_tqdm, desc=f"Sampling {batch_size} images on {device}"):
            if t > 0:
                z = torch.randn(batch_size, *self.image_dim).to(device)
            else:
                z = torch.zeros(size=(batch_size, *self.image_dim)).to(device)
            sigma_t = torch.sqrt(self.beta[t])
            noise_pred = self.model(x, t * torch.ones(batch_size, dtype=torch.long).to(device))
            x = (x - self.beta[t] * noise_pred / torch.sqrt(1 - self.alpha_bar[t])) / torch.sqrt(1 - self.beta[t]) + sigma_t * z
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        return x
