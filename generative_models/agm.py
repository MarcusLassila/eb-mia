from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch

class AbstractGenerativeModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def move_to(self, device):
        raise NotImplementedError

    @abstractmethod
    def per_sample_loss(self, model, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def loss(self, model, x, autocast_context=nullcontext()):
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def sample(self, model, batch_size, **kwargs):
        raise NotImplementedError
