import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import training.train_split as train_split
from data import datasets


class _FakeDataset:
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return None


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


class TestTrainSplit(unittest.TestCase):
    def test_save_and_load_indices(self):
        indices = [5, 1, 3]
        with tempfile.TemporaryDirectory() as tmpdir:
            train_split.save_indices(indices, 10, tmpdir, "indices.pkl")
            loaded = train_split.load_indices(Path(tmpdir) / "indices.pkl", len_dataset=10)
        self.assertEqual(loaded, indices)

    def test_select_fraction_uses_float_fraction_and_remainder_coin_flip(self):
        rng = np.random.default_rng(3)
        selected = train_split.select_fraction([0, 1, 2, 3, 4], 2 / 3, rng)
        self.assertEqual(selected, [4, 2, 1])

    def test_cli_sample_and_complement_create_expected_pickles(self):
        rng_factory = np.random.default_rng
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(train_split, "load_dataset", return_value=_FakeDataset()),
                patch.object(
                    train_split.np.random,
                    "default_rng",
                    side_effect=lambda seed=None: rng_factory(5),
                ),
            ):
                train_split.main([
                    "--dataset",
                    "FakeDataset",
                    "--output-dir",
                    tmpdir,
                    "--seed",
                    "5",
                    "--fraction",
                    "0.5",
                    "--mode",
                    "sample",
                ])
                sample_path = Path(tmpdir) / "FakeDataset-smpl-f0p5-s5.pkl"
                sample_indices = train_split.load_indices(sample_path, len_dataset=10)
                self.assertEqual(sample_indices, [1, 2, 3, 6, 7])

                train_split.main([
                    "--dataset",
                    "FakeDataset",
                    "--output-dir",
                    tmpdir,
                    "--mode",
                    "complement",
                    "--train-split-path",
                    str(sample_path),
                ])
                complement_path = Path(tmpdir) / "FakeDataset-smpl-f0p5-s5-comp.pkl"
                complement_indices = train_split.load_indices(complement_path, len_dataset=10)
                self.assertEqual(complement_indices, [0, 4, 5, 8, 9])
                self.assertEqual(set(sample_indices) | set(complement_indices), set(range(10)))

    def test_cli_entity_mode_creates_expected_indices(self):
        rng_factory = np.random.default_rng
        dataset = _FakeEntityDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(train_split, "load_dataset", return_value=dataset),
                patch.object(
                    train_split.np.random,
                    "default_rng",
                    side_effect=lambda seed=None: rng_factory(7),
                ),
            ):
                train_split.main([
                    "--dataset",
                    "CelebA",
                    "--output-dir",
                    tmpdir,
                    "--seed",
                    "7",
                    "--mode",
                    "entity",
                    "--entity-fraction",
                    "0.5",
                    "--per-entity-fraction",
                    "0.5",
                ])
            path = Path(tmpdir) / "CelebA-ent-f0p5-p0p5-s7.pkl"
            indices = train_split.load_indices(path, len_dataset=len(dataset))
        self.assertEqual(indices, [1, 3, 9, 10])

    def test_cli_entity_mode_includes_hold_out_fraction_in_file_name(self):
        rng_factory = np.random.default_rng
        dataset = _FakeEntityDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(train_split, "load_dataset", return_value=dataset),
                patch.object(
                    train_split.np.random,
                    "default_rng",
                    side_effect=lambda seed=None: rng_factory(7),
                ),
            ):
                train_split.main([
                    "--dataset",
                    "CelebA",
                    "--output-dir",
                    tmpdir,
                    "--seed",
                    "7",
                    "--mode",
                    "entity",
                    "--entity-fraction",
                    "0.5",
                    "--per-entity-fraction",
                    "0.5",
                    "--per-entity-hold-out",
                    "0.25",
                ])
            path = Path(tmpdir) / "CelebA-ent-f0p5-p0p5-h0p25-s7.pkl"
            indices = train_split.load_indices(path, len_dataset=len(dataset))
        self.assertEqual(indices, [1, 2, 9, 10])

    def test_cli_entity_complement_mode_creates_expected_indices(self):
        rng_factory = np.random.default_rng
        dataset = _FakeEntityDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "CelebA-ent-f0p5-p0p5-s7.pkl"
            train_split.save_indices([1, 3, 8, 11], len(dataset), tmpdir, base_path.name)
            with (
                patch.object(train_split, "load_dataset", return_value=dataset),
                patch.object(
                    train_split.np.random,
                    "default_rng",
                    side_effect=lambda seed=None: rng_factory(11),
                ),
            ):
                train_split.main([
                    "--dataset",
                    "CelebA",
                    "--output-dir",
                    tmpdir,
                    "--mode",
                    "entity-complement",
                    "--train-split-path",
                    str(base_path),
                ])
            complement_path = Path(tmpdir) / "CelebA-ent-f0p5-p0p5-s7-comp.pkl"
            indices = train_split.load_indices(complement_path, len_dataset=len(dataset))
        self.assertEqual(indices, [5, 7, 13, 15])


if __name__ == "__main__":
    unittest.main()
