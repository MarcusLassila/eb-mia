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


class FakeCelebAHQ(FakeDataset):
    pass


class FakeVGGFace2(FakeDataset):
    pass


class FakeMSMT17(FakeDataset):
    pass


class FakeFlowers(FakeDataset):
    pass


class TestLoadDataset(unittest.TestCase):
    def test_infer_dataset_name_from_lowercase(self):
        self.assertEqual(utils_module.infer_dataset_name("cifar10"), "CIFAR10")
        self.assertEqual(utils_module.infer_dataset_name("celeba"), "CelebA")
        self.assertEqual(utils_module.infer_dataset_name("celebahq"), "CelebAHQ")
        self.assertEqual(utils_module.infer_dataset_name("vggface2"), "VGGFace2")

    def test_infer_dataset_name_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset: unavailable"):
            utils_module.infer_dataset_name("unavailable")

    def test_load_dataset_supported_names(self):
        with patch.multiple(
            utils_module,
            MNIST=FakeMNIST,
            CIFAR10=FakeCIFAR10,
            CelebA=FakeCelebA,
            VGGFace2=FakeVGGFace2,
            MSMT17=FakeMSMT17,
            CelebAHQ=FakeCelebAHQ,
            Flowers=FakeFlowers,
        ):
            dataset = utils_module.load_dataset("MNIST", data_dir="root", transform="t")
            self.assertIsInstance(dataset, FakeMNIST)
            self.assertEqual(dataset.kwargs, {"data_dir": "root", "transform": "t"})

            dataset = utils_module.load_dataset("CIFAR10", data_dir="root", transform=None)
            self.assertIsInstance(dataset, FakeCIFAR10)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": None,
                    "random_horizontal_flip": True,
                },
            )

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

            dataset = utils_module.load_dataset("CelebA", data_dir="root", transform="t")
            self.assertIsInstance(dataset, FakeCelebA)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "size": 128,
                    "grayscale": False,
                    "random_horizontal_flip": True,
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
                "VGGFace2",
                data_dir="root",
                transform="t",
                size=64,
                grayscale=True,
                random_horizontal_flip=False,
            )
            self.assertIsInstance(dataset, FakeVGGFace2)
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

            dataset = utils_module.load_dataset("VGGFace2", data_dir="root", transform="t")
            self.assertIsInstance(dataset, FakeVGGFace2)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "size": 128,
                    "grayscale": False,
                    "random_horizontal_flip": True,
                },
            )

            dataset = utils_module.load_dataset(
                "MSMT17",
                data_dir="root",
                transform="t",
                size=64,
                grayscale=True,
                random_horizontal_flip=False,
            )
            self.assertIsInstance(dataset, FakeMSMT17)
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

            dataset = utils_module.load_dataset("MSMT17", data_dir="root", transform="t")
            self.assertIsInstance(dataset, FakeMSMT17)
            self.assertEqual(
                dataset.kwargs,
                {
                    "data_dir": "root",
                    "transform": "t",
                    "size": 128,
                    "grayscale": False,
                    "random_horizontal_flip": True,
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
