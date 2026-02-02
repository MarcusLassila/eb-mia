from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch

class AbstractGenerativeModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def move_to(self, device):
        raise NotImplementedError

    @property
    @abstractmethod
    def image_size(self):
        '''Return model image size. Args: None. Returns: int.'''
        raise NotImplementedError

    @abstractmethod
    def per_sample_loss(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def loss(self, x, autocast_context=nullcontext(), network_override=None, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def sample(self, batch_size, **kwargs):
        raise NotImplementedError
