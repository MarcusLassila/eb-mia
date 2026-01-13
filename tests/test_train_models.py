import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import Dataset

import mia.train_models as train_models
import mia.utils as mia_utils


class _DummyDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return torch.zeros(1, 2, 2)


def _dummy_dataset_loader(data_dir=None, transform=None):
    return _DummyDataset()


class TestTrainModels(unittest.TestCase):
    def test_train_model_from_indices_file_uses_mask(self):
        indices = [0, 2, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_indices.pkl"
            with open(path, "wb") as file:
                pickle.dump(indices, file)
            config = types.SimpleNamespace(dataset="Dummy", data_dir=".", val_frac=0.2, model="VAE")
            dummy_data = types.SimpleNamespace(Dummy=_dummy_dataset_loader)
            with mock.patch.object(train_models, "data", dummy_data):
                with mock.patch.object(train_models, "train_model") as mock_train_model:
                    train_models.train_model_from_indices_file(
                        accelerator=object(),
                        config=config,
                        savedir=Path(tmpdir),
                        train_indices_path=path,
                        id_=3,
                    )
                    args, _ = mock_train_model.call_args
                    train_mask = args[4]
                    expected_mask = mia_utils.index_to_mask(
                        torch.tensor(indices, dtype=torch.long),
                        6,
                    )
                    self.assertTrue(torch.equal(train_mask, expected_mask))


if __name__ == "__main__":
    unittest.main()
