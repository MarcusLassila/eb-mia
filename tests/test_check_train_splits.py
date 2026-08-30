import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import check_train_splits
from data import datasets
from training import train_split


class _FakeEntityDataset(datasets.EntityDataset):
    def __init__(self):
        self._entity_ids = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            dtype=torch.long,
        )

    @property
    def entity_ids(self):
        return self._entity_ids

    @property
    def n_entities(self):
        return 4

    def get_entity_index_table(self):
        return {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
            2: [8, 9, 10, 11],
            3: [12, 13, 14, 15],
        }

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self._entity_ids)


class _FakeSparseEntityDataset(datasets.EntityDataset):
    def __init__(self):
        self._entity_ids = torch.tensor(
            [0, 1, 1, 2, 2, 3],
            dtype=torch.long,
        )

    @property
    def entity_ids(self):
        return self._entity_ids

    @property
    def n_entities(self):
        return 4

    def get_entity_index_table(self):
        return {
            0: [0],
            1: [1, 2],
            2: [3, 4],
            3: [5],
        }

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self._entity_ids)


class _FakeSampleDataset:
    def __len__(self):
        return 10


class TestCheckTrainSplits(unittest.TestCase):
    def test_main_reports_sample_overlap_statistics_for_non_entity_dataset(self):
        dataset = _FakeSampleDataset()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_split.save_indices([0, 1, 2, 3, 4], len(dataset), tmpdir, "ImageNet-smpl-f0p5-s0.pkl")
            train_split.save_indices([5, 6, 7, 8, 9], len(dataset), tmpdir, "ImageNet-smpl-f0p5-s0-comp.pkl")
            train_split.save_indices([0, 2, 4, 6, 8], len(dataset), tmpdir, "ImageNet-smpl-f0p5-s1.pkl")
            train_split.save_indices([1, 3, 5, 7, 9], len(dataset), tmpdir, "ImageNet-smpl-f0p5-s1-comp.pkl")
            with patch.object(check_train_splits, "load_dataset", return_value=dataset):
                with redirect_stdout(stdout):
                    check_train_splits.main([
                        "--train-splits-dir",
                        str(tmpdir_path),
                        "--data-dir",
                        "./datasets",
                    ])
        output = stdout.getvalue()
        self.assertIn("Dataset: ImageNet", output)
        self.assertIn("Split mode: sample", output)
        self.assertIn("Complement pairs checked: 2", output)
        self.assertIn("Other pairs checked: 4", output)
        self.assertIn("Mean training samples: 5.000000", output)
        self.assertIn("Mean training fraction (% of dataset): 50.000000", output)
        self.assertIn("Mean datapoint overlap (% of dataset): 25.000000", output)
        self.assertIn("Std datapoint overlap (% of dataset): 5.000000", output)
        self.assertNotIn("Mean entity overlap", output)

    def test_main_reports_overlap_statistics(self):
        dataset = _FakeEntityDataset()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_split.save_indices([0, 1, 4, 5], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-s0.pkl")
            train_split.save_indices([8, 9, 12, 13], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-s0-comp.pkl")
            train_split.save_indices([0, 2, 8, 10], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-s1.pkl")
            train_split.save_indices([4, 6, 12, 14], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-s1-comp.pkl")
            with patch.object(check_train_splits, "load_dataset", return_value=dataset):
                with redirect_stdout(stdout):
                    check_train_splits.main([
                        "--train-splits-dir",
                        str(tmpdir_path),
                        "--data-dir",
                        "./datasets",
                    ])
        output = stdout.getvalue()
        self.assertIn("Dataset: CelebA", output)
        self.assertIn("Complement pairs checked: 2", output)
        self.assertIn("Other pairs checked: 4", output)
        self.assertIn("Mean training samples: 4.000000", output)
        self.assertIn("Mean training fraction (% of dataset): 25.000000", output)
        self.assertIn("Entities with non-empty holdout: 0", output)
        self.assertIn("Mean datapoint overlap (% of dataset): 6.250000", output)
        self.assertIn("Std datapoint overlap (% of dataset): 0.000000", output)
        self.assertIn("Mean entity overlap (% of entities): 25.000000", output)
        self.assertIn("Std entity overlap (% of entities): 0.000000", output)

    def test_main_allows_optional_zero_sample_entities(self):
        dataset = _FakeSparseEntityDataset()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_split.save_indices([1], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-h0p2-s0.pkl")
            train_split.save_indices([3], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-h0p2-s0-comp.pkl")
            with patch.object(check_train_splits, "load_dataset", return_value=dataset):
                with redirect_stdout(stdout):
                    check_train_splits.main([
                        "--train-splits-dir",
                        str(tmpdir_path),
                        "--data-dir",
                        "./datasets",
                    ])
        output = stdout.getvalue()
        self.assertIn("Dataset: CelebA", output)
        self.assertIn("Complement pairs checked: 1", output)
        self.assertIn("Mean training samples: 1.000000", output)

    def test_main_raises_when_non_optional_entity_is_missing(self):
        dataset = _FakeSparseEntityDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_split.save_indices([1], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-h0p2-s0.pkl")
            train_split.save_indices([], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-h0p2-s0-comp.pkl")
            with patch.object(check_train_splits, "load_dataset", return_value=dataset):
                with self.assertRaisesRegex(ValueError, "does not cover all non-optional entities"):
                    check_train_splits.main([
                        "--train-splits-dir",
                        str(tmpdir_path),
                        "--data-dir",
                        "./datasets",
                    ])

    def test_main_raises_for_overlapping_complement_entities(self):
        dataset = _FakeEntityDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_split.save_indices([0, 1, 4, 5], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-s0.pkl")
            train_split.save_indices([2, 3, 12, 13], len(dataset), tmpdir, "CelebA-ent-f0p5-p0p5-s0-comp.pkl")
            with patch.object(check_train_splits, "load_dataset", return_value=dataset):
                with self.assertRaisesRegex(ValueError, "Overlapping entities"):
                    check_train_splits.main([
                        "--train-splits-dir",
                        str(tmpdir_path),
                        "--data-dir",
                        "./datasets",
                    ])


if __name__ == "__main__":
    unittest.main()
