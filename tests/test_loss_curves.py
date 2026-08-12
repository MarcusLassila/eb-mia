import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.loss_curves import load_loss_history, plot_loss_curves


class TestLossCurves(unittest.TestCase):
    def test_load_loss_history_requires_matching_nonempty_fields(self):
        with self.assertRaises(AssertionError):
            load_loss_history({"train_losses": [1.0], "val_losses": []})
        with self.assertRaises(AssertionError):
            load_loss_history({"train_losses": [], "val_losses": []})
        with self.assertRaises(AssertionError):
            load_loss_history({})

    def test_plot_loss_curves_saves_available_pngs(self):
        checkpoint = {
            "train_losses": [1.0, 0.8, 0.5],
            "train_eval_losses": [0.95, 0.75, 0.45],
            "val_losses": [1.1, 0.9, 0.7],
            "fixed_noise_train_losses": [0.6, 0.5, 0.4],
            "fixed_noise_val_losses": [0.7, 0.6, 0.5],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model-epoch3.pth"
            output_dir = Path(tmpdir) / "plots"
            torch.save(checkpoint, checkpoint_path)
            saved_paths = plot_loss_curves(checkpoint_path, output_dir)
            expected_paths = [
                output_dir / "model-epoch3_train_vs_val_loss.png",
                output_dir / "model-epoch3_train_eval_vs_val_loss.png",
                output_dir / "model-epoch3_fixed_noise_train_vs_val_loss.png",
            ]
            self.assertEqual(saved_paths, expected_paths)
            for output_path in expected_paths:
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_loss_curves_skips_missing_optional_histories(self):
        checkpoint = {
            "train_losses": [1.0, 0.8, 0.5],
            "val_losses": [1.1, 0.9, 0.7],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model-epoch3.pth"
            output_dir = Path(tmpdir) / "plots"
            torch.save(checkpoint, checkpoint_path)
            saved_paths = plot_loss_curves(checkpoint_path, output_dir)
            self.assertEqual(saved_paths, [output_dir / "model-epoch3_train_vs_val_loss.png"])


if __name__ == "__main__":
    unittest.main()
