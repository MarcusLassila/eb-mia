import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import mia.train_split as train_split
from data import data as data_module


class TestTrainSplit(unittest.TestCase):
    def test_random_subset_indices_deterministic(self):
        indices_a = train_split.sample_random_fraction_indices(10, 0.3, seed=123)
        indices_b = train_split.sample_random_fraction_indices(10, 0.3, seed=123)
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(len(indices_a), 3)
        self.assertTrue(all(0 <= index < 10 for index in indices_a))

    def test_save_and_load_indices(self):
        indices = [5, 1, 3]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "indices.pkl"
            train_split.save_indices(indices, path)
            loaded = train_split.load_indices(path)
            self.assertEqual(loaded, indices)

    def test_complement_indices(self):
        indices = [0, 2]
        complement = train_split.complement_indices(indices, n_items=5)
        self.assertEqual(complement, [1, 3, 4])

    def test_complement_subset_filename_contains_source(self):
        name = train_split.complement_subset_filename("cifar-random-frac0p5-seed1.pkl")
        self.assertIn("complement", name)
        self.assertIn("cifar-random-frac0p5-seed1", name)

    def test_random_subset_filename_structure(self):
        name = train_split.random_subset_filename("MNIST", 0.25, seed=7)
        self.assertIn("MNIST", name)
        self.assertIn("random", name)
        self.assertIn("frac0p25", name)
        self.assertIn("seed7", name)

    def test_entity_subset_indices_deterministic(self):
        entity_ids = [1, 1, 2, 2, 3, 3, 4, 4]
        indices_a = train_split.sample_entity_fraction_indices(
            entity_ids=entity_ids,
            entity_fraction=0.5,
            per_entity_fraction=0.5,
            seed=42,
        )
        indices_b = train_split.sample_entity_fraction_indices(
            entity_ids=entity_ids,
            entity_fraction=0.5,
            per_entity_fraction=0.5,
            seed=42,
        )
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(len(indices_a), 2)
        self.assertTrue(all(0 <= index < len(entity_ids) for index in indices_a))
        selected_entity_ids = {entity_ids[index] for index in indices_a}
        self.assertEqual(len(selected_entity_ids), 2)

    def test_entity_subset_indices_min_one_per_entity(self):
        entity_ids = [10, 11, 12, 13]
        indices = train_split.sample_entity_fraction_indices(
            entity_ids=entity_ids,
            entity_fraction=1.0,
            per_entity_fraction=0.5,
            seed=99,
        )
        selected_entity_ids = {entity_ids[index] for index in indices}
        self.assertEqual(len(selected_entity_ids), 4)
        self.assertEqual(len(indices), 4)

    def test_entity_subset_filename_structure(self):
        name = train_split.entity_subset_filename("CelebA", 0.5, 0.25, seed=9)
        self.assertIn("CelebA", name)
        self.assertIn("entity", name)
        self.assertIn("frac0p5", name)
        self.assertIn("per0p25", name)
        self.assertIn("seed9", name)

    def test_entity_subset_save_load_matches_dataset(self):
        class FakeSplit:
            def __init__(self, celeb_ids):
                self._data = [{"celeb_id": celeb_id, "image": None} for celeb_id in celeb_ids]

            def __getitem__(self, index):
                return self._data[index]

            def __len__(self):
                return len(self._data)

        class FakeConcatDataset:
            def __init__(self, datasets):
                self._data = []
                for dataset in datasets:
                    self._data.extend(dataset._data)

            def __getitem__(self, index):
                if isinstance(index, str):
                    return [item[index] for item in self._data]
                return self._data[index]

            def __len__(self):
                return len(self._data)

        def fake_load_dataset(*args, **kwargs):
            return {
                "train": FakeSplit([0, 0, 0, 0, 1, 1, 1, 1]),
                "valid": FakeSplit([2, 2, 2, 2]),
                "test": FakeSplit([3, 3, 3, 3]),
            }

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.data.load_dataset", side_effect=fake_load_dataset), patch(
                "data.data.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                dataset = data_module.CelebA(data_dir=tmpdir, transform=None)
                path = train_split.create_entity_subset(
                    dataset_name="CelebA",
                    entity_ids=dataset.entity_ids,
                    entity_fraction=0.5,
                    per_entity_fraction=0.5,
                    seed=123,
                    output_dir=tmpdir,
                )
                indices = train_split.load_indices(path)

        entity_ids = dataset.entity_ids.tolist()
        unique_entity_ids = sorted(set(entity_ids))
        selected_entity_ids = {entity_ids[index] for index in indices}
        expected_entity_count = int(len(unique_entity_ids) * 0.5)
        self.assertEqual(len(selected_entity_ids), expected_entity_count)

        total_counts = Counter(entity_ids)
        selected_counts = Counter(entity_ids[index] for index in indices)
        for entity_id in unique_entity_ids:
            expected = int(total_counts[entity_id] * 0.5)
            if entity_id not in selected_entity_ids:
                expected = 0
            self.assertEqual(selected_counts.get(entity_id, 0), expected)

    def test_cli_random_and_complement(self):
        class FakeDataset:
            def __init__(self, data_dir="./datasets", transform=None):
                self._n = 10

            def __len__(self):
                return self._n

            def __getitem__(self, index):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(data_module, "FakeDataset", FakeDataset, create=True):
                random_path = train_split.main([
                    "--dataset",
                    "FakeDataset",
                    "--output-dir",
                    tmpdir,
                    "--seed",
                    "5",
                    "--fraction",
                    "0.4",
                    "--mode",
                    "random",
                ])
                random_indices = train_split.load_indices(random_path)
                self.assertEqual(len(random_indices), 4)
                complement_path = train_split.main([
                    "--dataset",
                    "FakeDataset",
                    "--output-dir",
                    tmpdir,
                    "--mode",
                    "complement",
                    "--subset-path",
                    str(random_path),
                ])
                complement_indices = train_split.load_indices(complement_path)
                self.assertEqual(len(complement_indices), 6)
                combined = set(random_indices) | set(complement_indices)
                self.assertEqual(combined, set(range(10)))

    def test_cli_entity_mode(self):
        class FakeSplit:
            def __init__(self, celeb_ids):
                self._data = [{"celeb_id": celeb_id, "image": None} for celeb_id in celeb_ids]

            def __getitem__(self, index):
                return self._data[index]

            def __len__(self):
                return len(self._data)

        class FakeConcatDataset:
            def __init__(self, datasets):
                self._data = []
                for dataset in datasets:
                    self._data.extend(dataset._data)

            def __getitem__(self, index):
                if isinstance(index, str):
                    return [item[index] for item in self._data]
                return self._data[index]

            def __len__(self):
                return len(self._data)

        train_ids = [0, 0, 0, 0, 1, 1, 1, 1]
        valid_ids = [2, 2, 2, 2]
        test_ids = [3, 3, 3, 3]
        entity_ids = train_ids + valid_ids + test_ids

        def fake_load_dataset(*args, **kwargs):
            return {
                "train": FakeSplit(train_ids),
                "valid": FakeSplit(valid_ids),
                "test": FakeSplit(test_ids),
            }

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.data.load_dataset", side_effect=fake_load_dataset), patch(
                "data.data.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                path = train_split.main([
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
                indices = train_split.load_indices(path)
                unique_entity_ids = sorted(set(entity_ids))
                selected_entity_ids = {entity_ids[index] for index in indices}
                expected_entity_count = int(len(unique_entity_ids) * 0.5)
                self.assertEqual(len(selected_entity_ids), expected_entity_count)
                total_counts = Counter(entity_ids)
                selected_counts = Counter(entity_ids[index] for index in indices)
                for entity_id in unique_entity_ids:
                    expected = int(total_counts[entity_id] * 0.5)
                    if entity_id not in selected_entity_ids:
                        expected = 0
                    self.assertEqual(selected_counts.get(entity_id, 0), expected)


if __name__ == "__main__":
    unittest.main()
