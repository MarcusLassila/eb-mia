import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

import verify_dataset_metadata


class TestVerifyDatasetMetadata(unittest.TestCase):
    def save_checkpoint(self, path, train_indices):
        '''
        Save a minimal checkpoint with train indices.
        Args:
            path (Path): Output checkpoint path.
            train_indices (list[int]): Training indices to serialize.
        Returns:
            None
        '''
        checkpoint = {"train_indices": torch.tensor(train_indices, dtype=torch.long)}
        torch.save(checkpoint, path)

    def save_metadata(self, path, metadata):
        '''
        Save dataset metadata to a pickle file.
        Args:
            path (Path): Output metadata path.
            metadata (dict): Metadata payload to serialize.
        Returns:
            None
        '''
        with open(path, "wb") as file:
            pickle.dump(metadata, file)

    def test_main_accepts_first_n_sample_split(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            metadata_path = tmpdir_path / "CIFAR10.pkl"
            checkpoint_path = tmpdir_path / "DDPM-CIFAR10-smpl-first4-sz32-epoch4.pth"
            self.save_metadata(
                metadata_path,
                {
                    "version": 1,
                    "dataset": "CIFAR10",
                    "n_samples": 10,
                },
            )
            self.save_checkpoint(checkpoint_path, [0, 1, 2, 3])
            with redirect_stdout(stdout):
                verify_dataset_metadata.main([
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metadata-dir",
                    str(tmpdir_path),
                ])
        output = stdout.getvalue()
        self.assertIn("Training samples: 4", output)
        self.assertIn("Status: OK", output)

    def test_main_accepts_entity_complement_with_hold_out(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            metadata_path = tmpdir_path / "CelebA.pkl"
            checkpoint_path = tmpdir_path / "DDPM-CelebA-ent-f0p5-p0p5-h0p25-s0-comp-sz64-epoch4.pth"
            self.save_metadata(
                metadata_path,
                {
                    "version": 1,
                    "dataset": "CelebA",
                    "n_samples": 16,
                    "entity_ids": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "n_entities": 4,
                },
            )
            self.save_checkpoint(checkpoint_path, [8, 9, 12, 13])
            with redirect_stdout(stdout):
                verify_dataset_metadata.main([
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metadata-dir",
                    str(tmpdir_path),
                ])
        output = stdout.getvalue()
        self.assertIn("Selected entities: 2", output)
        self.assertIn("Hold-out samples excluded: 2", output)
        self.assertIn("Status: OK", output)

    def test_main_rejects_entity_hold_out_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            metadata_path = tmpdir_path / "CelebA.pkl"
            checkpoint_path = tmpdir_path / "DDPM-CelebA-ent-f0p5-p0p5-h0p25-s0-sz64-epoch4.pth"
            self.save_metadata(
                metadata_path,
                {
                    "version": 1,
                    "dataset": "CelebA",
                    "n_samples": 16,
                    "entity_ids": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "n_entities": 4,
                },
            )
            self.save_checkpoint(checkpoint_path, [0, 3, 8, 9])
            with self.assertRaisesRegex(ValueError, "Hold-out mismatch"):
                verify_dataset_metadata.main([
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metadata-dir",
                    str(tmpdir_path),
                ])


if __name__ == "__main__":
    unittest.main()
