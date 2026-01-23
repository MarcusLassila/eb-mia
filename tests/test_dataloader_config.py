import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset
import yaml

from accelerate.accelerate import AcceleratorLite


class _TensorImageDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


class TestDataLoaderConfig(unittest.TestCase):
    def test_default_dataloader_config(self):
        data = torch.randn(4, 1, 2, 2)
        dataset = _TensorImageDataset(data)
        model = torch.nn.Identity()
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        _, train_dataloader, _ = accelerator.prepare(model, dataset, dataset, batch_size=2)
        dataloader = train_dataloader.dataloader
        self.assertEqual(dataloader.num_workers, 0)
        self.assertFalse(dataloader.pin_memory)
        self.assertFalse(dataloader.persistent_workers)
        self.assertIsNone(dataloader.prefetch_factor)

    def test_custom_dataloader_config(self):
        data = torch.randn(4, 1, 2, 2)
        dataset = _TensorImageDataset(data)
        model = torch.nn.Identity()
        dataloader_config = {
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0, dataloader_config=dataloader_config)
        _, train_dataloader, _ = accelerator.prepare(model, dataset, dataset, batch_size=2)
        dataloader = train_dataloader.dataloader
        self.assertEqual(dataloader.num_workers, 2)
        self.assertTrue(dataloader.pin_memory)
        self.assertTrue(dataloader.persistent_workers)
        self.assertEqual(dataloader.prefetch_factor, 4)

    def test_training_configs_have_dataloader_config(self):
        root = Path(__file__).resolve().parents[1]
        celeba_config_path = root / "training" / "configs" / "config_train_celeba.yaml"
        vae_config_path = root / "training" / "configs" / "config_train_vae_celeba.yaml"

        with open(celeba_config_path, "r") as file:
            celeba_config = yaml.safe_load(file)
        _, celeba_params = next(iter(celeba_config.items()))
        self.assertIn("dataloader_config", celeba_params)
        self.assertEqual(celeba_params["dataloader_config"]["num_workers"], 0)
        self.assertFalse(celeba_params["dataloader_config"]["pin_memory"])
        self.assertFalse(celeba_params["dataloader_config"]["persistent_workers"])
        self.assertIsNone(celeba_params["dataloader_config"]["prefetch_factor"])

        with open(vae_config_path, "r") as file:
            vae_config = yaml.safe_load(file)
        _, vae_params = next(iter(vae_config.items()))
        self.assertIn("dataloader_config", vae_params)
        self.assertEqual(vae_params["dataloader_config"]["num_workers"], 8)
        self.assertTrue(vae_params["dataloader_config"]["pin_memory"])
        self.assertTrue(vae_params["dataloader_config"]["persistent_workers"])
        self.assertEqual(vae_params["dataloader_config"]["prefetch_factor"], 2)


if __name__ == "__main__":
    unittest.main()
