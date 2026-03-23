from generative_models.agm import AbstractGenerativeModel
from unet.unet import UNet

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdiffeq import odeint

from contextlib import nullcontext


class FlowMatching(AbstractGenerativeModel):
    '''
    Flow matching model with optimal transport conditional vector fields.
    '''

    def __init__(
        self,
        image_dim,
        std_min=0.01,
        base_channels=128,
        channel_mult=(1,2,2,2),
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
        self.std_min = std_min
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.n_res_blocks_per_level = n_res_blocks_per_level
        self.n_attention_heads = n_attention_heads
        self.channels_per_head = channels_per_head
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.use_sdpa = use_sdpa
        self.network = self._create_vector_field()

    def _create_vector_field(self):
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
            continuous_time=True,
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
        if isinstance(self.network, DDP):
            assert next(iter(self.network.parameters())).device == torch.device(device), "Should not move a DDP wrapped network to another device"
        else:
            self.network.to(device)

    def per_sample_loss(self, x, t, autocast_context=nullcontext(), network_override=None):
        network = network_override if network_override is not None else self.network
        z = torch.randn_like(x)
        t_b = t.view(t.shape[0], 1, 1, 1)
        phi_t = (1 - (1 - self.std_min) * t_b) * z + t_b * x
        y = x - (1 - self.std_min) * z
        with autocast_context:
            v_t = network(phi_t, t)
            loss = F.mse_loss(input=v_t, target=y, reduction="none").mean(dim=(1, 2, 3))
        return loss
    
    def loss(self, x, autocast_context=nullcontext(), network_override=None):
        t = torch.rand(size=(x.shape[0],), device=x.device)
        return self.per_sample_loss(x, t, autocast_context, network_override=network_override).mean()

    def _arg_swapped_network(self, t, x):
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.network(x, t)

    @torch.inference_mode()
    def sample(self, batch_size, method="dopri5", n_time_steps=50, **kwargs):
        '''
        Sample a batch of images and rescale them to `[0, 1]`.
        Args:
            batch_size (int): Number of images to sample.
            n_steps (int): Number of ODE integration steps.
            kwargs (dict): Ignored extra keyword arguments.
        Returns:
            torch.Tensor: Sampled image batch.
        '''
        self.network.eval()
        device = next(iter(self.network.parameters())).device
        x_0 = torch.randn(size=(batch_size, *self.image_dim), device=device, dtype=torch.float32)
        t = torch.linspace(0, 1, steps=n_time_steps, device=device, dtype=torch.float32)
        x = odeint(
            func=self._arg_swapped_network,
            y0=x_0,
            t=t,
            method=method,
            rtol=1e-5,
            atol=1e-5,
        )
        x_1 = x[-1]
        x_1 = (x_1 + 1.0) / 2.0
        x_1 = torch.clamp(x_1, 0.0, 1.0)
        return x_1
