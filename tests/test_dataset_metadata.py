import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import torch

from data import dataset_metadata as dataset_metadata_module
from data.datasets import EntityDataset


class _FakeEntityDataset(EntityDataset):
    def __init__(self):
        self._entity_ids = torch.tensor([0, 0, 2, 2, 2], dtype=torch.long)

    @property
    def entity_ids(self):
        return self._entity_ids

    @property
    def n_entities(self):
        return 3

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self._entity_ids)


class _FakeSampleDataset:
    def __len__(self):
        return 4


class TestDatasetMetadata(unittest.TestCase):
    def test_main_saves_entity_dataset_metadata(self):
        dataset = _FakeEntityDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "metadata"
            with patch.object(dataset_metadata_module, "load_dataset", return_value=dataset) as load_dataset_fn:
                output_path = dataset_metadata_module.main([
                    "--dataset",
                    "VGGFace2",
                    "--data-dir",
                    tmpdir,
                    "--output-dir",
                    str(output_dir),
                ])

            load_dataset_fn.assert_called_once_with(dataset_name="VGGFace2", data_dir=Path(tmpdir))
            self.assertEqual(output_path, output_dir / "VGGFace2.pkl")
            with open(output_path, "rb") as file:
                payload = pickle.load(file)
            self.assertEqual(
                payload,
                {
                    "version": 1,
                    "dataset": "VGGFace2",
                    "n_samples": 5,
                    "entity_ids": [0, 0, 2, 2, 2],
                    "n_entities": 3,
                },
            )

    def test_main_saves_sample_dataset_metadata_without_entity_fields(self):
        dataset = _FakeSampleDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "metadata"
            with patch.object(dataset_metadata_module, "load_dataset", return_value=dataset) as load_dataset_fn:
                output_path = dataset_metadata_module.main([
                    "--dataset",
                    "CIFAR10",
                    "--data-dir",
                    tmpdir,
                    "--output-dir",
                    str(output_dir),
                ])

            load_dataset_fn.assert_called_once_with(dataset_name="CIFAR10", data_dir=Path(tmpdir))
            self.assertEqual(output_path, output_dir / "CIFAR10.pkl")
            with open(output_path, "rb") as file:
                payload = pickle.load(file)
            self.assertEqual(
                payload,
                {
                    "version": 1,
                    "dataset": "CIFAR10",
                    "n_samples": 4,
                },
            )

    def test_main_without_dataset_saves_all_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "metadata"

            def fake_load_dataset(dataset_name, data_dir):
                self.assertEqual(data_dir, Path(tmpdir))
                if dataset_name == "MNIST":
                    return _FakeSampleDataset()
                if dataset_name == "VGGFace2":
                    return _FakeEntityDataset()
                raise AssertionError(f"Unexpected dataset: {dataset_name}")

            with (
                patch.object(dataset_metadata_module, "dataset_names", return_value=("MNIST", "VGGFace2")),
                patch.object(dataset_metadata_module, "load_dataset", side_effect=fake_load_dataset) as load_dataset_fn,
            ):
                output_paths = dataset_metadata_module.main([
                    "--data-dir",
                    tmpdir,
                    "--output-dir",
                    str(output_dir),
                ])

            self.assertEqual(
                output_paths,
                [
                    output_dir / "MNIST.pkl",
                    output_dir / "VGGFace2.pkl",
                ],
            )
            self.assertEqual(
                load_dataset_fn.call_args_list,
                [
                    call(dataset_name="MNIST", data_dir=Path(tmpdir)),
                    call(dataset_name="VGGFace2", data_dir=Path(tmpdir)),
                ],
            )
            with open(output_dir / "MNIST.pkl", "rb") as file:
                mnist_payload = pickle.load(file)
            with open(output_dir / "VGGFace2.pkl", "rb") as file:
                vggface2_payload = pickle.load(file)
            self.assertEqual(
                mnist_payload,
                {
                    "version": 1,
                    "dataset": "MNIST",
                    "n_samples": 4,
                },
            )
            self.assertEqual(
                vggface2_payload,
                {
                    "version": 1,
                    "dataset": "VGGFace2",
                    "n_samples": 5,
                    "entity_ids": [0, 0, 2, 2, 2],
                    "n_entities": 3,
                },
            )

    def test_load_dataset_metadata_returns_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            metadata_path = metadata_dir / "VGGFace2.pkl"
            with open(metadata_path, "wb") as file:
                pickle.dump(
                    {
                        "version": 1,
                        "dataset": "VGGFace2",
                        "n_samples": 5,
                        "entity_ids": [0, 0, 2, 2, 2],
                        "n_entities": 3,
                    },
                    file,
                )

            payload = dataset_metadata_module.load_dataset_metadata("VGGFace2", metadata_dir=metadata_dir)

            self.assertEqual(payload["dataset"], "VGGFace2")
            self.assertEqual(payload["n_samples"], 5)
            self.assertEqual(payload["entity_ids"], [0, 0, 2, 2, 2])

    def test_load_dataset_metadata_rejects_dataset_name_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            metadata_path = metadata_dir / "VGGFace2.pkl"
            with open(metadata_path, "wb") as file:
                pickle.dump(
                    {
                        "version": 1,
                        "dataset": "CelebA",
                        "n_samples": 5,
                        "entity_ids": [0, 0, 2, 2, 2],
                        "n_entities": 3,
                    },
                    file,
                )

            with self.assertRaisesRegex(ValueError, "Dataset metadata name mismatch"):
                dataset_metadata_module.load_dataset_metadata("VGGFace2", metadata_dir=metadata_dir)

    def test_load_dataset_metadata_rejects_entity_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            metadata_path = metadata_dir / "VGGFace2.pkl"
            with open(metadata_path, "wb") as file:
                pickle.dump(
                    {
                        "version": 1,
                        "dataset": "VGGFace2",
                        "n_samples": 5,
                        "entity_ids": [0, 0, 2],
                        "n_entities": 3,
                    },
                    file,
                )

            with self.assertRaisesRegex(ValueError, "Entity id length mismatch"):
                dataset_metadata_module.load_dataset_metadata("VGGFace2", metadata_dir=metadata_dir)

    def test_entity_index_table_from_entity_ids_preserves_population_order(self):
        entity_index_table = dataset_metadata_module.entity_index_table_from_entity_ids([2, 2, 0, 1, 0])
        self.assertEqual(entity_index_table, {2: [0, 1], 0: [2, 4], 1: [3]})


if __name__ == "__main__":
    unittest.main()
