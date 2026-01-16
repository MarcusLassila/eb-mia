import tempfile
import unittest

import torch
from torch.utils.data import Dataset, Subset

from accelerate.accelerate import AcceleratorLite
from ddpm.ddpm import DDPM
from training.train_loop import TrainConfig, TrainLoop
from pathlib import Path


class _TensorImageDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


class TestDDPMTrain(unittest.TestCase):
    def test_train_runs_two_epochs(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        beta = torch.linspace(start=1e-4, end=0.02, steps=10)
        model_config = {
            "beta": beta,
            "channel_mult": (1,),
            "image_dim": (1, 4, 4),
            "base_channels": 32,
            "dropout": 0.0,
            "resample_with_conv": True,
        }
        model = DDPM(**model_config)

        train_config = TrainConfig(
            batch_size=2,
            simul_batch_size=2,
            epochs=2,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="float16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "ddpm_test.pth"
            TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config=model_config,
                accelerator=accelerator,
                savepath=savepath,
            ).train()

    def test_train_runs_with_scheduler_enabled(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        beta = torch.linspace(start=1e-4, end=0.02, steps=10)
        model_config = {
            "beta": beta,
            "channel_mult": (1,),
            "image_dim": (1, 4, 4),
            "base_channels": 32,
            "dropout": 0.0,
            "resample_with_conv": True,
        }
        model = DDPM(**model_config)

        train_config = TrainConfig(
            batch_size=2,
            simul_batch_size=2,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="float16",
            lr_scheduler="cosine",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "ddpm_test.pth"
            TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config=model_config,
                accelerator=accelerator,
                savepath=savepath,
            ).train()


if __name__ == "__main__":
    unittest.main()
