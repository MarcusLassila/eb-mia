from generative_models.agm import AbstractGenerativeModel
from unet.unet import UNet

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from contextlib import nullcontext


class DDPM(AbstractGenerativeModel):

    def __init__(
        self,
        image_dim,
        time_steps=1000,
        beta_schedule="linear",
        base_channels=128,
        channel_mult=(1,1,2,2),
        n_attention_heads=1,
        attention_resolutions=(16,),
        dropout=0.0,
        resample_with_conv=True,
        use_sdpa=True,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.beta_schedule = beta_schedule
        self.time_steps = time_steps
        self.beta = self._create_beta_schedule()
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.n_attention_heads = n_attention_heads
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.use_sdpa = use_sdpa
        self.network = self._create_ddpm_noise_model()

    def _create_beta_schedule(self):
        match self.beta_schedule:
            case "linear":
                return torch.linspace(start=1e-4, end=0.02, steps=self.time_steps)
            case _:
                raise NotImplementedError

    def _create_ddpm_noise_model(self):
        assert self.image_dim[1] == self.image_dim[2], "requires square images"
        return UNet(
            image_size=self.image_dim[1],
            in_channels=self.image_dim[0],
            out_channels=self.image_dim[0],
            base_channels=self.base_channels,
            channel_mult=self.channel_mult,
            n_attention_heads=self.n_attention_heads,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            resample_with_conv=self.resample_with_conv,
            continuous_time=False,
            use_sdpa=self.use_sdpa,
        )

    def move_to(self, device):
        self.beta = self.beta.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        if isinstance(self.network, DDP):
            assert next(iter(self.network.parameters())).device == torch.device(device), "Should not move a DDP wrapped network to another device"
        else:
            self.network.to(device)

    def per_sample_loss(self, x, t, autocast_context=nullcontext(), network_override=None):
        network = network_override if network_override is not None else self.network
        eps = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
        with autocast_context:
            noise_pred = network(z, t)
            loss = F.mse_loss(input=noise_pred, target=eps, reduction="none").mean(dim=(1, 2, 3))
        return loss
    
    def loss(self, x, autocast_context=nullcontext(), network_override=None):
        t = torch.randint(low=0, high=self.time_steps, size=(x.shape[0],), device=x.device)
        return self.per_sample_loss(x, t, autocast_context, network_override=network_override).mean()

    @torch.inference_mode()
    def sample(self, batch_size, disable_tqdm=False):
        ''' Sample a batch of images and rescale them to floating point values in [0,1]. '''
        self.network.eval()
        device = next(iter(self.network.parameters())).device
        x = torch.randn(batch_size, *self.image_dim).to(device)
        for t in tqdm(range(self.time_steps - 1, -1, -1), disable=disable_tqdm, desc=f"Sampling {batch_size} images on {device}"):
            if t > 0:
                z = torch.randn(batch_size, *self.image_dim).to(device)
            else:
                z = torch.zeros(size=(batch_size, *self.image_dim)).to(device)
            sigma_t = torch.sqrt(self.beta[t])
            noise_pred = self.network(x, t * torch.ones(batch_size, dtype=torch.long).to(device))
            x = (x - self.beta[t] * noise_pred / torch.sqrt(1 - self.alpha_bar[t])) / torch.sqrt(1 - self.beta[t]) + sigma_t * z
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        return x
