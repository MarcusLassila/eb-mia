import tempfile
import unittest

import torch
from torch.utils.data import Dataset, Subset

from accelerate.accelerate import AcceleratorLite
from generative_models.flow_matching import FlowMatching
from training.train_loop import TrainConfig, TrainLoop
from pathlib import Path


class _TensorImageDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


class TestFlowMatching(unittest.TestCase):
    def test_sample_returns_unit_interval_images(self):
        '''Sample images and verify shape and range. Args: None. Returns: None.'''
        torch.manual_seed(0)
        model_config = {
            "image_dim": (1, 4, 4),
            "std_min": 0.01,
            "base_channels": 32,
            "channel_mult": (1,),
            "n_res_blocks_per_level": 1,
            "n_attention_heads": 1,
            "attention_resolutions": (4,),
            "dropout": 0.0,
            "resample_with_conv": True,
            "use_sdpa": True,
        }
        model = FlowMatching(**model_config)
        samples = model.sample(batch_size=3, n_steps=2)
        self.assertEqual(samples.shape, (3, 1, 4, 4))
        self.assertTrue(torch.isfinite(samples).all().item())
        self.assertGreaterEqual(float(samples.min()), 0.0)
        self.assertLessEqual(float(samples.max()), 1.0)

    def test_train_runs_one_epoch(self):
        '''Train FlowMatching for one epoch as a smoke test. Args: None. Returns: None.'''
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))
        model_config = {
            "image_dim": (1, 4, 4),
            "std_min": 0.01,
            "base_channels": 32,
            "channel_mult": (1,),
            "n_res_blocks_per_level": 1,
            "n_attention_heads": 1,
            "attention_resolutions": (4,),
            "dropout": 0.0,
            "resample_with_conv": True,
            "use_sdpa": True,
        }
        model = FlowMatching(**model_config)
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
            savepath = Path(tmpdir) / "flow_matching_test.pth"
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
