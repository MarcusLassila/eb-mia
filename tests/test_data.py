import os
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from datasets import load_dataset as hf_load_dataset
from PIL import Image
import torch

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
            self.assertTrue(name.endswith("logasja___VGGFace2"))
            self.assertEqual(config_name, "256")
            if split == "train":
                return FakeSplit([10, 10, 30])
            if split == "test":
                return FakeSplit([50, 70])
            raise AssertionError(f"Unexpected split: {split}")

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_repo_dir = str(Path(tmpdir) / "logasja___VGGFace2")
            with patch("data.datasets.snapshot_download", return_value=local_repo_dir) as snapshot_fn, patch(
                "data.datasets.load_dataset",
                side_effect=fake_load_dataset,
            ), patch("data.datasets.concatenate_datasets", side_effect=fake_concatenate_datasets):
                dataset = data_module.VGGFace2(
                    data_dir=tmpdir,
                    transform=lambda x: x,
                )
                snapshot_fn.assert_called_once_with(
                    repo_id="logasja/VGGFace2",
                    repo_type="dataset",
                    local_dir=local_repo_dir,
                )
                self.assertEqual(len(dataset), 5)
                self.assertEqual(dataset.entity_ids.tolist(), [0, 0, 1, 2, 3])
                self.assertEqual(dataset.n_entities, 4)

    def test_imagenet_combines_train_and_val_and_discards_labels(self):
        class FakeImageNet:
            def __init__(self, root, split, transform):
                self.root = root
                self.split = split
                self.transform = transform
                self.image = Image.new("RGB", (32, 32), color=(128, 128, 128))

            def __getitem__(self, index):
                return self.transform(self.image), 123

            def __len__(self):
                return 2 if self.split == "train" else 1

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.datasets.datasets.ImageNet", side_effect=FakeImageNet) as imagenet_fn:
                dataset = data_module.ImageNet(data_dir=tmpdir, transform=lambda image: image)

                self.assertEqual(len(dataset), 3)
                self.assertIsInstance(dataset[0], Image.Image)
                self.assertEqual(imagenet_fn.call_count, 2)

    def test_imagenet_grayscale_transform(self):
        class FakeImageNet:
            def __init__(self, root, split, transform):
                self.transform = transform
                self.image = Image.new("RGB", (256, 256), color=(128, 128, 128))

            def __getitem__(self, index):
                return self.transform(self.image), 0

            def __len__(self):
                return 1

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.datasets.datasets.ImageNet", side_effect=FakeImageNet):
                dataset = data_module.ImageNet(data_dir=tmpdir, transform=None, size=32, grayscale=True)

                sample = dataset[0]

        self.assertEqual(tuple(sample.shape), (1, 32, 32))

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
            self.assertTrue(name.endswith("logasja___VGGFace2"))
            self.assertEqual(config_name, "256")
            return FakeSplit([0, 1], image)

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_repo_dir = str(Path(tmpdir) / "logasja___VGGFace2")
            with patch("data.datasets.snapshot_download", return_value=local_repo_dir) as snapshot_fn, patch(
                "data.datasets.load_dataset",
                side_effect=fake_load_dataset,
            ), patch("data.datasets.concatenate_datasets", side_effect=fake_concatenate_datasets):
                gray_dataset = data_module.VGGFace2(data_dir=tmpdir, transform=None, size=32, grayscale=True)
                gray_sample = gray_dataset[0]
                self.assertEqual(tuple(gray_sample.shape), (1, 32, 32))
                color_dataset = data_module.VGGFace2(data_dir=tmpdir, transform=None, size=64, grayscale=False)
                color_sample = color_dataset[0]
                self.assertEqual(tuple(color_sample.shape), (3, 64, 64))
                self.assertEqual(snapshot_fn.call_count, 2)

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

    def test_msmt17_extracts_archive_and_normalizes_entity_ids(self):
        train_image = Image.new("RGB", (16, 24), color=(64, 128, 192))
        test_image = Image.new("RGB", (16, 24), color=(192, 128, 64))
        train_buffer = BytesIO()
        test_buffer = BytesIO()
        train_image.save(train_buffer, format="JPEG")
        test_image.save(test_buffer, format="JPEG")
        train_bytes = train_buffer.getvalue()
        test_bytes = test_buffer.getvalue()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_repo_dir = Path(tmpdir) / "xianpeijie___MSMT17_V1"
            local_repo_dir.mkdir()
            archive_path = local_repo_dir / "MSMT17_V1.zip"
            with zipfile.ZipFile(archive_path, mode="w") as archive_file:
                archive_file.writestr(
                    "MSMT17_V1/list_train.txt",
                    "train_a.jpg 10\ntrain_b.jpg 10\n",
                )
                archive_file.writestr(
                    "MSMT17_V1/list_val.txt",
                    "val_a.jpg 30\n",
                )
                archive_file.writestr(
                    "MSMT17_V1/list_query.txt",
                    "query_a.jpg 50\n",
                )
                archive_file.writestr(
                    "MSMT17_V1/list_gallery.txt",
                    "gallery_a.jpg 70\n",
                )
                archive_file.writestr("MSMT17_V1/train/train_a.jpg", train_bytes)
                archive_file.writestr("MSMT17_V1/train/train_b.jpg", train_bytes)
                archive_file.writestr("MSMT17_V1/train/val_a.jpg", train_bytes)
                archive_file.writestr("MSMT17_V1/test/query_a.jpg", test_bytes)
                archive_file.writestr("MSMT17_V1/test/gallery_a.jpg", test_bytes)
            with patch("data.datasets.snapshot_download", return_value=str(local_repo_dir)) as snapshot_fn:
                dataset = data_module.MSMT17(
                    data_dir=tmpdir,
                    transform=lambda x: x,
                )
            snapshot_fn.assert_called_once_with(
                repo_id="xianpeijie/MSMT17_V1",
                repo_type="dataset",
                local_dir=str(local_repo_dir),
            )
            extracted_root = local_repo_dir / "MSMT17_V1"
            self.assertTrue((extracted_root / "list_train.txt").exists())
            self.assertEqual(len(dataset), 5)
            self.assertEqual(dataset.entity_ids.tolist(), [0, 0, 1, 2, 3])
            self.assertEqual(dataset.n_entities, 4)

    def test_msmt17_grayscale_transform(self):
        image = Image.new("RGB", (20, 30), color=(100, 110, 120))
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_repo_dir = Path(tmpdir) / "xianpeijie___MSMT17_V1"
            extracted_root = local_repo_dir / "MSMT17_V1"
            train_dir = extracted_root / "train"
            test_dir = extracted_root / "test"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            (extracted_root / "list_train.txt").write_text("train_a.jpg 0\n")
            (extracted_root / "list_val.txt").write_text("")
            (extracted_root / "list_query.txt").write_text("query_a.jpg 1\n")
            (extracted_root / "list_gallery.txt").write_text("")
            (train_dir / "train_a.jpg").write_bytes(image_bytes)
            (test_dir / "query_a.jpg").write_bytes(image_bytes)
            with patch("data.datasets.snapshot_download", return_value=str(local_repo_dir)) as snapshot_fn:
                gray_dataset = data_module.MSMT17(data_dir=tmpdir, transform=None, size=32, grayscale=True)
                gray_sample = gray_dataset[0]
                color_dataset = data_module.MSMT17(data_dir=tmpdir, transform=None, size=64, grayscale=False)
                color_sample = color_dataset[0]
            self.assertEqual(tuple(gray_sample.shape), (1, 32, 32))
            self.assertEqual(tuple(color_sample.shape), (3, 64, 64))
            self.assertEqual(snapshot_fn.call_count, 2)

    def test_msmt17_default_transform_center_pads_before_resize(self):
        image = Image.new("RGB", (20, 30), color=(255, 255, 255))
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_repo_dir = Path(tmpdir) / "xianpeijie___MSMT17_V1"
            extracted_root = local_repo_dir / "MSMT17_V1"
            train_dir = extracted_root / "train"
            test_dir = extracted_root / "test"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            (extracted_root / "list_train.txt").write_text("train_a.jpg 0\n")
            (extracted_root / "list_val.txt").write_text("")
            (extracted_root / "list_query.txt").write_text("")
            (extracted_root / "list_gallery.txt").write_text("")
            (train_dir / "train_a.jpg").write_bytes(image_bytes)
            with patch("data.datasets.snapshot_download", return_value=str(local_repo_dir)):
                dataset = data_module.MSMT17(
                    data_dir=tmpdir,
                    transform=None,
                    size=32,
                    grayscale=False,
                    random_horizontal_flip=False,
                )
            sample = dataset[0]
            left_padding = sample[:, :, :5]
            center_content = sample[:, :, 16]
            right_padding = sample[:, :, -5:]
            self.assertEqual(tuple(sample.shape), (3, 32, 32))
            self.assertTrue(torch.equal(left_padding, torch.full_like(left_padding, -1.0)))
            self.assertTrue(torch.equal(right_padding, torch.full_like(right_padding, -1.0)))
            self.assertTrue(torch.all(center_content > 0.9))

    def test_msmt17_keeps_train_and_test_identity_spaces_disjoint(self):
        image = Image.new("RGB", (12, 18), color=(80, 90, 100))
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_repo_dir = Path(tmpdir) / "xianpeijie___MSMT17_V1"
            extracted_root = local_repo_dir / "MSMT17_V1"
            train_dir = extracted_root / "train"
            test_dir = extracted_root / "test"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            (extracted_root / "list_train.txt").write_text("train_a.jpg 0\n")
            (extracted_root / "list_val.txt").write_text("")
            (extracted_root / "list_query.txt").write_text("query_a.jpg 0\n")
            (extracted_root / "list_gallery.txt").write_text("")
            (train_dir / "train_a.jpg").write_bytes(image_bytes)
            (test_dir / "query_a.jpg").write_bytes(image_bytes)
            with patch("data.datasets.snapshot_download", return_value=str(local_repo_dir)):
                dataset = data_module.MSMT17(
                    data_dir=tmpdir,
                    transform=lambda x: x,
                )
            self.assertEqual(dataset.entity_ids.tolist(), [0, 1])
            self.assertEqual(dataset.n_entities, 2)


if __name__ == "__main__":
    unittest.main()
