import tempfile
import unittest
from unittest.mock import patch

from data import data as data_module


class TestData(unittest.TestCase):
    def test_celeba_filters_singleton_celeb_ids(self):
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
                "train": FakeSplit([0, 0, 1]),
                "valid": FakeSplit([2, 2]),
                "test": FakeSplit([3]),
            }

        def fake_concatenate_datasets(datasets):
            return FakeConcatDataset(datasets=datasets)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("data.data.load_dataset", side_effect=fake_load_dataset), patch(
                "data.data.concatenate_datasets",
                side_effect=fake_concatenate_datasets,
            ):
                dataset = data_module.CelebA(
                    data_dir=tmpdir,
                    transform=lambda x: x,
                    min_celeb_samples=2,
                )
                self.assertEqual(len(dataset), 4)
                self.assertEqual(dataset.celeb_ids.tolist(), [0, 0, 2, 2])
                self.assertEqual(dataset.entity_ids.tolist(), [0, 0, 2, 2])


if __name__ == "__main__":
    unittest.main()
