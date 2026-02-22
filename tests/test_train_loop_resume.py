import tempfile
import unittest

import torch
from torch.utils.data import Dataset, Subset

from accelerate.accelerate import AcceleratorLite
from generative_models.agm import AbstractGenerativeModel
from training.train_loop import TrainConfig, TrainLoop
from contextlib import nullcontext
from pathlib import Path


class _TensorDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


class _CompileWrappedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._orig_mod = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self._orig_mod(x)


class _SimpleModel(AbstractGenerativeModel):
    def __init__(self):
        super().__init__()
        self.network = _CompileWrappedLinear()

    def move_to(self, device):
        self.network.to(device)

    def per_sample_loss(self, x, *args, **kwargs):
        loss = self.loss(x, *args, **kwargs)
        return loss.repeat(x.shape[0])

    def loss(self, x, autocast_context=nullcontext(), network_override=None, **kwargs):
        network = network_override if network_override is not None else self.network
        with autocast_context:
            out = network(x)
            return (out ** 2).mean()

    @torch.inference_mode()
    def sample(self, batch_size, **kwargs):
        return torch.zeros(batch_size, 1)

    @property
    def image_size(self):
        '''Return model image size. Args: None. Returns: int.'''
        return 1


class TestTrainLoopResume(unittest.TestCase):
    def test_resume_checkpoint_loads_into_compiled_wrapper(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        model = _SimpleModel()

        expected_weight = torch.tensor([[3.0]])
        checkpoint = {
            "epoch": 5,
            "raw_network_state_dict": {"weight": expected_weight.clone()},
            "network_state_dict": {"weight": expected_weight.clone()},
        }
        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="bfloat16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "resume_test.pth"
            train_loop = TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
                resume_checkpoint=checkpoint,
            )
        self.assertEqual(train_loop.start_epoch, 5)
        self.assertTrue(torch.equal(train_loop.raw_network._orig_mod.weight.detach().cpu(), expected_weight))


if __name__ == "__main__":
    unittest.main()
