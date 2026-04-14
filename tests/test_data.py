import os
import tempfile
import unittest
from unittest.mock import patch

from datasets import load_dataset as hf_load_dataset
from PIL import Image

from data import datasets as data_module


class TestData(unittest.TestCase):
    def test_celeba_keeps_singleton_celeb_ids(self):
        class FakeSplit:
            def __init__(self, celeb_ids):
                self._data = [{"celeb_id": celeb_id, "image": None} for celeb_id in celeb_ids]

            def __getitem__(self, index):
                return self._data[index]

            def __len__(self):
                return len(self._data)

        class FakeConcatDataset:
            def __init__(self, datasets=None, data=None):
                if data is not None:
                    self._data = list(data)
                else:
                    self._data = []
                    for dataset in datasets:
                        self._data.extend(dataset._data)

            def __getitem__(self, index):
                if isinstance(index, str):
                    return [item[index] for item in self._data]
                return self._data[index]

            def __len__(self):
                return len(self._data)

            def select(self, indices):
                return FakeConcatDataset(data=[self._data[index] for index in indices])

        def fake_load_dataset(*args, **kwargs):
            return {
                "train": FakeSplit([10, 10, 30]),
                "valid": FakeSplit([40, 40]),
                "test": FakeSplit([50]),
            }

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.datasets.load_dataset", side_effect=fake_load_dataset), patch(
                "data.datasets.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                dataset = data_module.CelebA(
                    data_dir=tmpdir,
                    transform=lambda x: x,
                )
                self.assertEqual(len(dataset), 6)
                self.assertEqual(dataset.entity_ids.tolist(), [0, 0, 1, 2, 2, 3])
                self.assertEqual(dataset.n_entities, 4)

    def test_celeba_normalizes_entity_ids(self):
        class FakeSplit:
            def __init__(self, celeb_ids):
                self._data = [{"celeb_id": celeb_id, "image": None} for celeb_id in celeb_ids]

            def __getitem__(self, index):
                return self._data[index]

            def __len__(self):
                return len(self._data)

        class FakeConcatDataset:
            def __init__(self, datasets=None, data=None):
                if data is not None:
                    self._data = list(data)
                else:
                    self._data = []
                    for dataset in datasets:
                        self._data.extend(dataset._data)

            def __getitem__(self, index):
                if isinstance(index, str):
                    return [item[index] for item in self._data]
                return self._data[index]

            def __len__(self):
                return len(self._data)

            def select(self, indices):
                return FakeConcatDataset(data=[self._data[index] for index in indices])

        def fake_load_dataset(*args, **kwargs):
            return {
                "train": FakeSplit([10, 10, 30]),
                "valid": FakeSplit([50, 50]),
                "test": FakeSplit([70]),
            }

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.datasets.load_dataset", side_effect=fake_load_dataset), patch(
                "data.datasets.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                dataset = data_module.CelebA(
                    data_dir=tmpdir,
                    transform=lambda x: x,
                )
                self.assertEqual(dataset.entity_ids.tolist(), [0, 0, 1, 2, 2, 3])
                self.assertEqual(dataset.n_entities, 4)

    def test_celeba_low_res_grayscale_transform(self):
        class FakeSplit:
            def __init__(self, celeb_ids, image):
                self._data = [{"celeb_id": celeb_id, "image": image} for celeb_id in celeb_ids]

            def __getitem__(self, index):
                return self._data[index]

            def __len__(self):
                return len(self._data)

        class FakeConcatDataset:
            def __init__(self, datasets=None, data=None):
                if data is not None:
                    self._data = list(data)
                else:
                    self._data = []
                    for dataset in datasets:
                        self._data.extend(dataset._data)

            def __getitem__(self, index):
                if isinstance(index, str):
                    return [item[index] for item in self._data]
                return self._data[index]

            def __len__(self):
                return len(self._data)

            def select(self, indices):
                return FakeConcatDataset(data=[self._data[index] for index in indices])

        image = Image.new("RGB", (178, 178), color=(128, 128, 128))

        def fake_load_dataset(*args, **kwargs):
            return {
                "train": FakeSplit([0, 0], image),
                "valid": FakeSplit([1, 1], image),
                "test": FakeSplit([2, 2], image),
            }

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.datasets.load_dataset", side_effect=fake_load_dataset), patch(
                "data.datasets.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                low_res = data_module.CelebA(data_dir=tmpdir, transform=None, size=32, grayscale=True)
                low_res_sample = low_res[0]
                self.assertEqual(tuple(low_res_sample.shape), (1, 32, 32))
                color = data_module.CelebA(data_dir=tmpdir, transform=None, size=64, grayscale=False)
                color_sample = color[0]
                self.assertEqual(tuple(color_sample.shape), (3, 64, 64))

    def test_vggface2_normalizes_entity_ids(self):
        class FakeSplit:
            def __init__(self, class_ids):
                self._data = [{"class_id": class_id, "image": None} for class_id in class_ids]

            def __getitem__(self, index):
                return self._data[index]

            def __len__(self):
                return len(self._data)

        class FakeConcatDataset:
            def __init__(self, datasets=None, data=None):
                if data is not None:
                    self._data = list(data)
                else:
                    self._data = []
                    for dataset in datasets:
                        self._data.extend(dataset._data)

            def __getitem__(self, index):
                if isinstance(index, str):
                    return [item[index] for item in self._data]
                return self._data[index]

            def __len__(self):
                return len(self._data)

        def fake_load_dataset(name, config_name, split, cache_dir):
            self.assertEqual(name, "logasja/VGGFace2")
            self.assertEqual(config_name, "256")
            if split == "train":
                return FakeSplit([10, 10, 30])
            if split == "test":
                return FakeSplit([50, 70])
            raise AssertionError(f"Unexpected split: {split}")

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.datasets.load_dataset", side_effect=fake_load_dataset), patch(
                "data.datasets.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                dataset = data_module.VGGFace2(
                    data_dir=tmpdir,
                    transform=lambda x: x,
                )
                self.assertEqual(len(dataset), 5)
                self.assertEqual(dataset.entity_ids.tolist(), [0, 0, 1, 2, 3])
                self.assertEqual(dataset.n_entities, 4)

    def test_vggface2_grayscale_transform(self):
        class FakeSplit:
            def __init__(self, class_ids, image):
                self._data = [{"class_id": class_id, "image": image} for class_id in class_ids]

            def __getitem__(self, index):
                return self._data[index]

            def __len__(self):
                return len(self._data)

        class FakeConcatDataset:
            def __init__(self, datasets=None):
                self._data = []
                for dataset in datasets:
                    self._data.extend(dataset._data)

            def __getitem__(self, index):
                if isinstance(index, str):
                    return [item[index] for item in self._data]
                return self._data[index]

            def __len__(self):
                return len(self._data)

        image = Image.new("RGB", (256, 256), color=(128, 128, 128))

        def fake_load_dataset(name, config_name, split, cache_dir):
            self.assertEqual(name, "logasja/VGGFace2")
            self.assertEqual(config_name, "256")
            return FakeSplit([0, 1], image)

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.datasets.load_dataset", side_effect=fake_load_dataset), patch(
                "data.datasets.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                gray_dataset = data_module.VGGFace2(data_dir=tmpdir, transform=None, size=32, grayscale=True)
                gray_sample = gray_dataset[0]
                self.assertEqual(tuple(gray_sample.shape), (1, 32, 32))
                color_dataset = data_module.VGGFace2(data_dir=tmpdir, transform=None, size=64, grayscale=False)
                color_sample = color_dataset[0]
                self.assertEqual(tuple(color_sample.shape), (3, 64, 64))

    def test_vggface2_huggingface_tiny_split(self):
        if os.environ.get("RUN_REMOTE_DATASET_TESTS") != "1":
            self.skipTest("Set RUN_REMOTE_DATASET_TESTS=1 to run remote dataset checks.")
        try:
            train_dataset = hf_load_dataset("logasja/VGGFace2", "256", split="train[:1]")
            test_dataset = hf_load_dataset("logasja/VGGFace2", "256", split="test[:1]")
        except Exception as error:
            self.skipTest(f"Unable to fetch tiny VGGFace2 subset from Hugging Face: {error}")
        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(len(test_dataset), 1)
        self.assertIn("image", train_dataset.column_names)
        self.assertIn("class_id", train_dataset.column_names)


if __name__ == "__main__":
    unittest.main()
