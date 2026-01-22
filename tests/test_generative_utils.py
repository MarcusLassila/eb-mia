import tempfile
import unittest
from pathlib import Path

import torch

from utils import get_dataset_and_model_from_path, get_train_indices
from generative_models.utils import load_model
from generative_models.ddpm import DDPM
from generative_models.vae import VAE


class TestGenerativeUtils(unittest.TestCase):
    def test_get_dataset_and_model_from_path(self):
        dataset, model = get_dataset_and_model_from_path("/tmp/cifar10-DDPM-123.pth")
        self.assertEqual(dataset, "cifar10")
        self.assertEqual(model, "DDPM")

    def test_get_train_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cifar10-DDPM-test.pth"
            train_indices = torch.tensor([1, 2, 3])
            checkpoint = {"train_indices": train_indices}
            torch.save(checkpoint, path)
            loaded = get_train_indices(path)
            self.assertTrue(torch.equal(loaded, train_indices))

    def test_load_model_ddpm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "Dummy-DDPM-0.pth"
            image_dim = (1, 4, 4)
            model_config = {
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
            model = DDPM(**model_config)
            checkpoint = {
                "model_config": model_config,
                "network_state_dict": model.network.state_dict(),
                "train_indices": torch.tensor([0, 1, 2]),
            }
            torch.save(checkpoint, path)
            loaded_model, train_indices = load_model(path, torch.device("cpu"))
            self.assertIsInstance(loaded_model, DDPM)
            self.assertEqual(loaded_model.time_steps, 10)
            self.assertTrue(torch.equal(train_indices, checkpoint["train_indices"]))
            for key, tensor in model.network.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded_model.network.state_dict()[key]))
            self.assertFalse(loaded_model.network.training)

    def test_load_model_vae(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "Dummy-VAE-0.pth"
            model_config = {
                "in_ch": 1,
                "in_dim": 8,
                "latent_dim": 4,
                "n_rsamples": 2,
            }
            model = VAE(**model_config)
            checkpoint = {
                "model_config": model_config,
                "network_state_dict": model.network.state_dict(),
                "train_indices": torch.tensor([3, 4]),
            }
            torch.save(checkpoint, path)
            loaded_model, train_indices = load_model(path, torch.device("cpu"))
            self.assertIsInstance(loaded_model, VAE)
            self.assertEqual(loaded_model.n_rsamples, 2)
            self.assertTrue(torch.equal(train_indices, checkpoint["train_indices"]))
            for key, tensor in model.network.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded_model.network.state_dict()[key]))
            self.assertFalse(loaded_model.network.training)


if __name__ == "__main__":
    unittest.main()
