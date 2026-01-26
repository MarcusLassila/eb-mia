import tempfile
import unittest
from pathlib import Path

import torch

import generate_samples
from generative_models.vae import VAE


class TestGenerateSamples(unittest.TestCase):
    def _write_checkpoint(self, tmpdir):
        model_config = {
            "in_ch": 1,
            "in_dim": 8,
            "latent_dim": 4,
            "n_rsamples": 1,
        }
        model = VAE(**model_config)
        checkpoint = {
            "model_config": model_config,
            "network_state_dict": model.network.state_dict(),
            "train_indices": torch.tensor([0]),
        }
        path = Path(tmpdir) / "Dummy-VAE-0.pth"
        torch.save(checkpoint, path)
        return path, model_config

    def test_generate_samples_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path, model_config = self._write_checkpoint(tmpdir)
            samples = generate_samples.generate_samples_from_checkpoint(path, 3, "cpu")
            self.assertEqual(samples.shape[0], 3)
            self.assertEqual(samples.shape[1:], (model_config["in_ch"], model_config["in_dim"], model_config["in_dim"]))

    def test_save_samples_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path, _ = self._write_checkpoint(tmpdir)
            output_dir = Path(tmpdir) / "outputs"

            png_paths = generate_samples.sample_and_save(path, 2, output_dir)
            self.assertEqual(len(png_paths), 2)
            for saved_path in png_paths:
                self.assertTrue(saved_path.exists())
                self.assertEqual(saved_path.suffix, ".png")

            raw_paths = generate_samples.sample_and_save(
                path,
                2,
                output_dir,
                output_format="raw",
                base_name="raw_samples",
            )
            self.assertEqual(len(raw_paths), 1)
            self.assertTrue(raw_paths[0].exists())
            loaded = torch.load(raw_paths[0])
            self.assertEqual(loaded.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
