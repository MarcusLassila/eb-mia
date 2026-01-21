import tempfile
import unittest

import torch
from torch.utils.data import Dataset, Subset

from accelerate.accelerate import AcceleratorLite
from training.train_loop import TrainConfig, TrainLoop
from generative_models.vae import VAE, VAE_Network
from pathlib import Path


class _TensorImageDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


class TestVAETrain(unittest.TestCase):
    def test_train_runs_two_epochs(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 8, 8)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        model_config = {
            "in_ch": 1,
            "in_dim": 8,
            "latent_dim": 4,
        }
        model = VAE_Network(**model_config)
        generative_class = VAE(n_rsamples=1)
        configs = {
            "generative_class_config": {
                "n_rsamples": 1,
            },
            "model_config": model_config,
        }

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
            savepath = Path(tmpdir) / "vae_test.pth"
            TrainLoop(
                model=model,
                generative_class=generative_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                configs=configs,
                accelerator=accelerator,
                savepath=savepath,
            ).train()

    def test_train_runs_with_scheduler_enabled(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 8, 8)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        model_config = {
            "in_ch": 1,
            "in_dim": 8,
            "latent_dim": 4,
        }
        model = VAE_Network(**model_config)
        generative_class = VAE(n_rsamples=1)
        configs = {
            "generative_class_config": {
                "n_rsamples": 1,
            },
            "model_config": model_config,
        }

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
            savepath = Path(tmpdir) / "vae_test.pth"
            TrainLoop(
                model=model,
                generative_class=generative_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                configs=configs,
                accelerator=accelerator,
                savepath=savepath,
            ).train()

    def test_train_runs_with_ema_enabled(self):
        '''Ensure EMA checkpoints are saved and match raw weights when decay is zero.'''
        torch.manual_seed(0)
        data = torch.randn(8, 1, 8, 8)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        model_config = {
            "in_ch": 1,
            "in_dim": 8,
            "latent_dim": 4,
        }
        model = VAE_Network(**model_config)
        generative_class = VAE(n_rsamples=1)
        configs = {
            "generative_class_config": {
                "n_rsamples": 1,
            },
            "model_config": model_config,
        }

        train_config = TrainConfig(
            batch_size=2,
            simul_batch_size=2,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            use_ema=True,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="float16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "vae_test.pth"
            TrainLoop(
                model=model,
                generative_class=generative_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                configs=configs,
                accelerator=accelerator,
                savepath=savepath,
            ).train()
            checkpoint_path = Path(tmpdir) / "vae_test-epoch1.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertIn("ema_model_state_dict", checkpoint)
            raw_state = checkpoint["raw_model_state_dict"]
            ema_state = checkpoint["ema_model_state_dict"]
            model_state = checkpoint["model_state_dict"]
            self.assertEqual(raw_state.keys(), ema_state.keys())
            self.assertEqual(model_state.keys(), ema_state.keys())
            for key, model_tensor in model_state.items():
                self.assertTrue(torch.equal(model_tensor, ema_state[key]))
            for key, raw_tensor in raw_state.items():
                ema_tensor = ema_state[key]
                self.assertTrue(torch.equal(ema_tensor, raw_tensor))


if __name__ == "__main__":
    unittest.main()
