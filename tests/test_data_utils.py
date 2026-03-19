import unittest
from unittest.mock import patch

from data import utils as utils_module


class FakeDataset:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeMNIST(FakeDataset):
    pass


class FakeCIFAR10(FakeDataset):
    pass


class FakeCelebA(FakeDataset):
    pass


class FakeCelebA2(FakeDataset):
    pass


class FakeCelebAHQ(FakeDataset):
    pass


class FakeFlowers(FakeDataset):
    pass


class TestLoadDataset(unittest.TestCase):
    def test_load_dataset_supported_names(self):
        with patch.multiple(
            utils_module,
            MNIST=FakeMNIST,
            CIFAR10=FakeCIFAR10,
            CelebA=FakeCelebA,
            CelebA2=FakeCelebA2,
            CelebAHQ=FakeCelebAHQ,
            Flowers=FakeFlowers,
        ):
            dataset = utils_module.load_dataset("MNIST", data_dir="root", transform="t")
            self.assertIsInstance(dataset, FakeMNIST)
            self.assertEqual(dataset.kwargs, {"data_dir": "root", "transform": "t"})

            dataset = utils_module.load_dataset("CIFAR10", data_dir="root", transform=None)
            self.assertIsInstance(dataset, FakeCIFAR10)
            self.assertEqual(dataset.kwargs, {"data_dir": "root", "transform": None})

            dataset = utils_module.load_dataset(
                "CelebA",
                data_dir="root",
                transform="t",
                size=64,
                grayscale=True,
                random_horizontal_flip=False,
            )
            self.assertIsInstance(dataset, FakeCelebA)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "size": 64,
                    "grayscale": True,
                    "random_horizontal_flip": False,
                },
            )

            dataset = utils_module.load_dataset("CelebA2", data_dir="root", transform="t")
            self.assertIsInstance(dataset, FakeCelebA2)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "size": 128,
                    "grayscale": False,
                },
            )

            dataset = utils_module.load_dataset("CelebAHQ", data_dir="root", transform="t")
            self.assertIsInstance(dataset, FakeCelebAHQ)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "random_horizontal_flip": True,
                },
            )

            dataset = utils_module.load_dataset(
                "CelebAHQ",
                data_dir="root",
                transform="t",
                random_horizontal_flip=False,
            )
            self.assertIsInstance(dataset, FakeCelebAHQ)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "random_horizontal_flip": False,
                },
            )

            dataset = utils_module.load_dataset(
                "Flowers",
                data_dir="root",
                transform="t",
                random_horizontal_flip=False,
            )
            self.assertIsInstance(dataset, FakeFlowers)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "random_horizontal_flip": False,
                },
            )

    def test_load_dataset_unknown_name(self):
        with self.assertRaises(ValueError):
            utils_module.load_dataset("Unknown")


if __name__ == "__main__":
    unittest.main()
