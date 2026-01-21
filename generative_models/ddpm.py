from generative_models.agm import AbstractGenerativeModel
from unet.unet import UNet

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from contextlib import nullcontext


class DDPM(AbstractGenerativeModel):

    def __init__(
        self,
        image_dim,
        time_steps=1000,
        beta_schedule="linear",
    ):
        super().__init__()
        assert image_dim[1] == image_dim[2], "Only square images are supported"
        self.image_dim = image_dim
        self.beta_schedule = beta_schedule
        self.time_steps = time_steps
        self.beta = self._create_beta_schedule()
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)

    def _create_beta_schedule(self):
        match self.beta_schedule:
            case "linear":
                return torch.linspace(start=1e-4, end=0.02, steps=self.time_steps)
            case _:
                raise NotImplementedError

    def move_to(self, device):
        self.beta = self.beta.to(device)
        self.alpha_bar = self.alpha_bar.to(device)

    def per_sample_loss(self, model, x, t, autocast_context=nullcontext()):
        eps = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
        with autocast_context:
            noise_pred = model(z, t)
            loss = F.mse_loss(input=noise_pred, target=eps, reduction="none").mean(dim=(1, 2, 3))
        return loss
    
    def loss(self, model, x, autocast_context=nullcontext()):
        t = torch.randint(low=0, high=self.time_steps, size=(x.shape[0],), device=x.device)
        return self.per_sample_loss(model, x, t, autocast_context).mean()

    @torch.inference_mode()
    def sample(self, model, batch_size, disable_tqdm=False):
        ''' Sample a batch of images and rescale them to floating point values in [0,1]. '''
        model.eval()
        device = next(iter(model.parameters())).device
        x = torch.randn(batch_size, *self.image_dim).to(device)
        for t in tqdm(range(self.time_steps - 1, -1, -1), disable=disable_tqdm, desc=f"Sampling {batch_size} images on {device}"):
            if t > 0:
                z = torch.randn(batch_size, *self.image_dim).to(device)
            else:
                z = torch.zeros(size=(batch_size, *self.image_dim)).to(device)
            sigma_t = torch.sqrt(self.beta[t])
            noise_pred = model(x, t * torch.ones(batch_size, dtype=torch.long).to(device))
            x = (x - self.beta[t] * noise_pred / torch.sqrt(1 - self.alpha_bar[t])) / torch.sqrt(1 - self.beta[t]) + sigma_t * z
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        return x


def create_ddpm_noise_model(
    image_dim,
    base_channels,
    channel_mult,
    n_attention_heads=1,
    attention_resolutions=(16,),
    dropout=0.0,
    resample_with_conv=True,
    use_sdpa=True,
):
    assert image_dim[1] == image_dim[2], "requires square images"
    return UNet(
        image_size=image_dim[1],
        in_channels=image_dim[0],
        out_channels=image_dim[0],
        base_channels=base_channels,
        channel_mult=channel_mult,
        n_attention_heads=n_attention_heads,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        resample_with_conv=resample_with_conv,
        continuous_time=False,
        use_sdpa=use_sdpa,
    )
