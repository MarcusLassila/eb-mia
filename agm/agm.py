from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch.nn as nn

class AbstractGenerativeModel(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, x, autocast_context=nullcontext(), **kwargs):
        pass

    @abstractmethod
    def sample(self, batch_size, **kwargs):
        pass
