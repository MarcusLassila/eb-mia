import tempfile
import unittest
from pathlib import Path

import torch

from utils import get_dataset_and_model_from_path, get_train_indices
from generative_models.utils import load_model
from generative_models.ddpm import DDPM, create_ddpm_noise_model
from generative_models.vae import VAE, VAE_Network


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
                "base_channels": 32,
                "channel_mult": (1,),
                "n_attention_heads": 1,
                "attention_resolutions": (4,),
                "dropout": 0.0,
                "resample_with_conv": True,
                "use_sdpa": True,
            }
            generative_class_config = {
                "image_dim": image_dim,
                "time_steps": 10,
                "beta_schedule": "linear",
            }
            model = create_ddpm_noise_model(**model_config)
            checkpoint = {
                "configs": {
                    "model_config": model_config,
                    "generative_class_config": generative_class_config,
                },
                "model_state_dict": model.state_dict(),
                "train_indices": torch.tensor([0, 1, 2]),
            }
            torch.save(checkpoint, path)
            loaded_model, generative_class, train_indices = load_model(path, torch.device("cpu"))
            self.assertIsInstance(generative_class, DDPM)
            self.assertEqual(generative_class.time_steps, 10)
            self.assertTrue(torch.equal(train_indices, checkpoint["train_indices"]))
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded_model.state_dict()[key]))
            self.assertFalse(loaded_model.training)

    def test_load_model_vae(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "Dummy-VAE-0.pth"
            model_config = {
                "in_ch": 1,
                "in_dim": 8,
                "latent_dim": 4,
            }
            generative_class_config = {"n_rsamples": 2}
            model = VAE_Network(**model_config)
            checkpoint = {
                "configs": {
                    "model_config": model_config,
                    "generative_class_config": generative_class_config,
                },
                "model_state_dict": model.state_dict(),
                "train_indices": torch.tensor([3, 4]),
            }
            torch.save(checkpoint, path)
            loaded_model, generative_class, train_indices = load_model(path, torch.device("cpu"))
            self.assertIsInstance(generative_class, VAE)
            self.assertEqual(generative_class.n_rsamples, 2)
            self.assertTrue(torch.equal(train_indices, checkpoint["train_indices"]))
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded_model.state_dict()[key]))
            self.assertFalse(loaded_model.training)


if __name__ == "__main__":
    unittest.main()
