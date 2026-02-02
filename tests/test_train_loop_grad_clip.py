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


class _InfGradModel(AbstractGenerativeModel):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(1, 1, bias=False)

    def move_to(self, device):
        self.network.to(device)

    def per_sample_loss(self, x, *args, **kwargs):
        loss = self.loss(x, *args, **kwargs)
        return loss.repeat(x.shape[0])

    def loss(self, x, autocast_context=nullcontext(), network_override=None, **kwargs):
        network = network_override if network_override is not None else self.network
        with autocast_context:
            out = network(x)
            return out.sum() * torch.tensor(float("inf"), device=out.device)

    @torch.inference_mode()
    def sample(self, batch_size, **kwargs):
        return torch.zeros(batch_size, 1)

    @property
    def image_size(self):
        '''Return model image size. Args: None. Returns: int.'''
        return 1


class TestTrainLoopGradClip(unittest.TestCase):
    def test_grad_clip_raises_on_non_finite_norm(self):
        torch.manual_seed(0)
        data = torch.ones(2, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 1))
        val_dataset = Subset(dataset, indices=torch.arange(1, 2))
        model = _InfGradModel()

        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=1.0,
            autocast_dtype="bfloat16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "fail_fast_test.pth"
            with self.assertRaisesRegex(RuntimeError, "Non-finite gradient norm detected"):
                TrainLoop(
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    train_config=train_config,
                    model_config={"name": "inf-grad"},
                    accelerator=accelerator,
                    savepath=savepath,
                ).train()


if __name__ == "__main__":
    unittest.main()
