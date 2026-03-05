import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import verify_shadow_split_balance as verify_module


class TestVerifyShadowSplitBalance(unittest.TestCase):
    def test_verify_shadow_split_balance_passes_when_every_sample_is_in_half_of_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir)
            checkpoint_paths = [
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s0-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s1-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s2-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch1.pth",
            ]
            for path in checkpoint_paths:
                path.touch()

            train_indices_by_path = {
                checkpoint_paths[0].name: torch.tensor([0, 1, 2], dtype=torch.long),
                checkpoint_paths[1].name: torch.tensor([0, 3, 4], dtype=torch.long),
                checkpoint_paths[2].name: torch.tensor([1, 3, 5], dtype=torch.long),
                checkpoint_paths[3].name: torch.tensor([2, 4, 5], dtype=torch.long),
            }

            def fake_get_train_indices(path):
                return train_indices_by_path[Path(path).name]

            with (
                patch.object(verify_module, "load_dataset", return_value=list(range(6))) as load_dataset_fn,
                patch.object(verify_module.utils, "get_train_indices", side_effect=fake_get_train_indices),
            ):
                summary = verify_module.verify_shadow_split_balance(
                    checkpoints_dir=checkpoints_dir,
                    data_dir="/tmp/data",
                )

            self.assertTrue(summary["is_balanced"])
            self.assertEqual(summary["mismatched_indices"], [])
            self.assertEqual(summary["expected_inclusions_per_sample"], 2)
            load_dataset_fn.assert_called_once_with("CIFAR10", data_dir="/tmp/data", size=32, grayscale=False)

    def test_verify_shadow_split_balance_reports_mismatched_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir)
            checkpoint_paths = [
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s0-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s1-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s2-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch1.pth",
            ]
            for path in checkpoint_paths:
                path.touch()

            train_indices_by_path = {
                checkpoint_paths[0].name: torch.tensor([0, 1, 2], dtype=torch.long),
                checkpoint_paths[1].name: torch.tensor([0, 3, 4], dtype=torch.long),
                checkpoint_paths[2].name: torch.tensor([1, 3, 5], dtype=torch.long),
                checkpoint_paths[3].name: torch.tensor([2, 3, 4], dtype=torch.long),
            }

            def fake_get_train_indices(path):
                return train_indices_by_path[Path(path).name]

            with (
                patch.object(verify_module, "load_dataset", return_value=list(range(6))),
                patch.object(verify_module.utils, "get_train_indices", side_effect=fake_get_train_indices),
            ):
                summary = verify_module.verify_shadow_split_balance(
                    checkpoints_dir=checkpoints_dir,
                )

            self.assertFalse(summary["is_balanced"])
            self.assertIn(3, summary["mismatched_counts"])
            self.assertEqual(summary["mismatched_counts"][3], 3)
            self.assertIn(5, summary["mismatched_counts"])
            self.assertEqual(summary["mismatched_counts"][5], 1)

    def test_verify_shadow_split_balance_requires_even_number_of_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir)
            checkpoint_paths = [
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s0-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s1-sz32-epoch1.pth",
                checkpoints_dir / "DDPM-cifar10-rand-f0p5-s2-sz32-epoch1.pth",
            ]
            for path in checkpoint_paths:
                path.touch()

            with self.assertRaisesRegex(ValueError, "even number of checkpoints"):
                verify_module.verify_shadow_split_balance(checkpoints_dir=checkpoints_dir)


if __name__ == "__main__":
    unittest.main()
