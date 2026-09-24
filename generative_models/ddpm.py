from generative_models.agm import AbstractDiffusionModel
from unet.unet import UNet

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from contextlib import nullcontext


class DDPM(AbstractDiffusionModel):

    def __init__(
        self,
        image_dim,
        time_steps=1000,
        beta_schedule="linear",
        base_channels=128,
        channel_mult=(1,1,2,2),
        n_res_blocks_per_level=2,
        n_attention_heads=1,
        channels_per_head=None,
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
        self.n_res_blocks_per_level = n_res_blocks_per_level
        self.n_attention_heads = n_attention_heads
        self.channels_per_head = channels_per_head
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
            n_res_blocks_per_level=self.n_res_blocks_per_level,
            n_attention_heads=self.n_attention_heads,
            channels_per_head=self.channels_per_head,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            resample_with_conv=self.resample_with_conv,
            continuous_time=False,
            use_sdpa=self.use_sdpa,
        )

    @property
    def image_size(self):
        '''
        Return the model image size.
        Returns:
            int: Spatial size of generated images.
        '''
        return self.image_dim[1]

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

    def fixed_noise_level_per_sample_loss(self, x, noise_level, autocast_context=nullcontext(), network_override=None):
        assert 0.0 <= noise_level <= 1.0
        time_index = round((self.time_steps - 1) * noise_level)
        t = torch.full(size=(x.shape[0],), fill_value=time_index, dtype=torch.long, device=x.device)
        return self.per_sample_loss(x, t, autocast_context=autocast_context, network_override=network_override)

    @torch.inference_mode()
    def denoiser_norm(self, x, noise_level, lp_norm=4):
        '''
        SimA-MC sample score signal from "Score-based Membership Inference on Diffusion Models"
        '''
        assert 0.0 <= noise_level <= 1.0
        time_index = round((self.time_steps - 1) * noise_level)
        t = torch.full(size=(x.shape[0],), fill_value=time_index, dtype=torch.long, device=x.device)
        eps = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
        noise_pred = self.network(z, t)
        return torch.linalg.vector_norm(noise_pred, ord=lp_norm, dim=(1, 2, 3))

    @torch.inference_mode()
    def pia_score(self, x, noise_level, lp_norm=4, normalize=True):
        '''
        PIA score signal from "An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization"
        '''
        assert 0.0 <= noise_level <= 1.0
        t_0 = torch.full(size=(x.shape[0],), fill_value=0, dtype=torch.long, device=x.device)
        eps_0 = self.network(x, t_0)
        if normalize:
            N = x[0].numel()
            eps_0_norm = torch.linalg.vector_norm(eps_0, ord=1, dim=(1, 2, 3)).clamp_min(1e-6)
            normalizer = N * np.sqrt(np.pi * 0.5) / eps_0_norm
            eps_0 = eps_0 * normalizer.view(x.shape[0], 1, 1, 1)
        time_index = round((self.time_steps - 1) * noise_level)
        t = torch.full(size=(x.shape[0],), fill_value=time_index, dtype=torch.long, device=x.device)
        alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps_0
        eps_t = self.network(z, t)
        return torch.linalg.vector_norm(eps_0 - eps_t, ord=lp_norm, dim=(1, 2, 3))

    @torch.inference_mode()
    def t_error(self, x, noise_level, step_length=1):
        '''
        t-error signal from "Are Diffusion Models Vulnerable to Membership Inference Attacks?"
        '''
        assert 0.0 <= noise_level <= 1.0
        time_index = round((self.time_steps - 1) * noise_level)
        if step_length < 1 or time_index < 2 * step_length or time_index % step_length != 0:
            raise ValueError("SecMI requires step_length >= 1 and a selected timestep that is a multiple of step_length and at least 2*step_length.")
        tilde_x_t = x
        for current_index in range(0, time_index - step_length, step_length):
            current_t = torch.full((x.shape[0],), current_index, dtype=torch.long, device=x.device)
            target_t = current_t + step_length
            tilde_x_t = self._tstep(tilde_x_t, current_t, target_t)
        previous_t = torch.full((x.shape[0],), time_index - step_length, dtype=torch.long, device=x.device)
        selected_t = previous_t + step_length
        y = self._tstep(tilde_x_t, previous_t, selected_t)
        y = self._tstep(y, selected_t, previous_t)
        squared_error = (y - tilde_x_t) ** 2
        return squared_error.flatten(1).sum(dim=-1)

    def _fphi(self, x_t, t, eps_t):
        '''Helper for t-error'''
        alpha_bar_t = self.alpha_bar[t].view(x_t.shape[0], 1, 1, 1)
        return (x_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)

    def _tstep(self, x_t, t, target_t):
        '''Helper for t-error'''
        eps_t = self.network(x_t, t)
        f_t = self._fphi(x_t, t, eps_t)
        alpha_bar_tt = self.alpha_bar[target_t].view(x_t.shape[0], 1, 1, 1)
        return torch.sqrt(alpha_bar_tt) * f_t + torch.sqrt(1 - alpha_bar_tt) * eps_t

    @torch.inference_mode()
    def sample(self, batch_size, disable_tqdm=False):
        '''
        Sample a batch of images and rescale them to `[0, 1]`.
        Args:
            batch_size (int): Number of images to sample.
            disable_tqdm (bool): Whether to disable the sampling progress bar.
        Returns:
            torch.Tensor: Sampled image batch.
        '''
        self.network.eval()
        device = next(iter(self.network.parameters())).device
        x = torch.randn(batch_size, *self.image_dim, device=device)
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
