import tempfile
import unittest

import torch
from torch.utils.data import Dataset, Subset

from accelerate_lite.accelerate import AcceleratorLite
from generative_models.ddpm import DDPM
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
    def test_default_n_res_blocks_per_level_is_two(self):
        '''
        Construct DDPM with the default residual block depth.
        Returns:
            None
        '''
        model = DDPM(
            image_dim=(1, 4, 4),
            time_steps=10,
            beta_schedule="linear",
            base_channels=32,
            channel_mult=(1,),
            attention_resolutions=(4,),
            use_sdpa=True,
        )
        self.assertEqual(model.n_res_blocks_per_level, 2)
        self.assertEqual(model.network.n_res_blocks_per_level, 2)

    def test_train_runs_two_epochs(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        image_dim = (1, 4, 4)
        ddpm_config = {
            "image_dim": image_dim,
            "time_steps": 10,
            "beta_schedule": "linear",
            "base_channels": 32,
            "channel_mult": (1,),
            "n_attention_heads": 1,
            "attention_resolutions": (4,),
            "dropout": 0.0,
            "resample_with_conv": True,
            "use_sdpa": True,
        }
        model = DDPM(**ddpm_config)

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
                model_config=ddpm_config,
                accelerator=accelerator,
                savepath=savepath,
            ).train()

    def test_train_runs_with_scheduler_enabled(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        image_dim = (1, 4, 4)
        ddpm_config = {
            "image_dim": image_dim,
            "time_steps": 10,
            "beta_schedule": "linear",
            "base_channels": 32,
            "channel_mult": (1,),
            "n_attention_heads": 1,
            "attention_resolutions": (4,),
            "dropout": 0.0,
            "resample_with_conv": True,
            "use_sdpa": True,
        }
        model = DDPM(**ddpm_config)

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
                model_config=ddpm_config,
                accelerator=accelerator,
                savepath=savepath,
            ).train()

    def test_train_runs_with_ema_enabled(self):
        '''
        Ensure EMA checkpoints are saved and match raw weights when decay is zero.
        Returns:
            None
        '''
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        image_dim = (1, 4, 4)
        ddpm_config = {
            "image_dim": image_dim,
            "time_steps": 10,
            "beta_schedule": "linear",
            "base_channels": 32,
            "channel_mult": (1,),
            "n_attention_heads": 1,
            "attention_resolutions": (4,),
            "dropout": 0.0,
            "resample_with_conv": True,
            "use_sdpa": True,
        }
        model = DDPM(**ddpm_config)

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
            savepath = Path(tmpdir) / "ddpm_test.pth"
            TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config=ddpm_config,
                accelerator=accelerator,
                savepath=savepath,
            ).train()
            checkpoint_path = Path(tmpdir) / "ddpm_test-epoch1.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertIn("ema_network_state_dict", checkpoint)
            raw_state = checkpoint["raw_network_state_dict"]
            ema_state = checkpoint["ema_network_state_dict"]
            model_state = checkpoint["network_state_dict"]
            self.assertEqual(raw_state.keys(), ema_state.keys())
            self.assertEqual(model_state.keys(), ema_state.keys())
            for key, model_tensor in model_state.items():
                self.assertTrue(torch.equal(model_tensor, ema_state[key]))
            for key, raw_tensor in raw_state.items():
                ema_tensor = ema_state[key]
                self.assertTrue(torch.equal(ema_tensor, raw_tensor))

    def test_checkpoint_state_dicts_are_unwrapped_before_save(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        image_dim = (1, 4, 4)
        ddpm_config = {
            "image_dim": image_dim,
            "time_steps": 10,
            "beta_schedule": "linear",
            "base_channels": 32,
            "channel_mult": (1,),
            "n_attention_heads": 1,
            "attention_resolutions": (4,),
            "dropout": 0.0,
            "resample_with_conv": True,
            "use_sdpa": True,
        }
        model = DDPM(**ddpm_config)

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
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "ddpm_test.pth"
            train_loop = TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config=ddpm_config,
                accelerator=accelerator,
                savepath=savepath,
            )
            original_state_dict = train_loop.raw_network.state_dict
            train_loop.raw_network.state_dict = lambda: {f"_orig_mod.{k}": v for k, v in original_state_dict().items()}
            train_loop.train()
            checkpoint_path = Path(tmpdir) / "ddpm_test-epoch1.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertTrue(all(not key.startswith("_orig_mod.") for key in checkpoint["network_state_dict"]))
            self.assertTrue(all(not key.startswith("_orig_mod.") for key in checkpoint["raw_network_state_dict"]))


if __name__ == "__main__":
    unittest.main()
