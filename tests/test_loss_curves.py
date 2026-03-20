import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.loss_curves import load_loss_history, plot_loss_curves


class TestLossCurves(unittest.TestCase):
    def test_load_loss_history_requires_matching_nonempty_fields(self):
        with self.assertRaises(ValueError):
            load_loss_history({"train_losses": [1.0], "val_losses": []})
        with self.assertRaises(ValueError):
            load_loss_history({"train_losses": [], "val_losses": []})
        with self.assertRaises(ValueError):
            load_loss_history({})

    def test_plot_loss_curves_saves_png(self):
        checkpoint = {
            "train_losses": [1.0, 0.8, 0.5],
            "val_losses": [1.1, 0.9, 0.7],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model-epoch3.pth"
            output_path = Path(tmpdir) / "loss_curve.png"
            torch.save(checkpoint, checkpoint_path)
            saved_path = plot_loss_curves(checkpoint_path, output_path)
            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
