import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import Dataset

import training.train_model as train_model
import utils


class _DummyDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return torch.zeros(1, 2, 2)


def _dummy_dataset_loader(dataset_name=None, data_dir=None, transform=None, **kwargs):
    return _DummyDataset()


class TestTrainModels(unittest.TestCase):
    def test_train_model_from_indices_file_uses_mask(self):
        indices = [0, 2, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_indices.pkl"
            with open(path, "wb") as file:
                pickle.dump(indices, file)
            config = types.SimpleNamespace(
                dataset="Dummy",
                data_dir=".",
                val_frac=0.2,
                model="VAE",
                image_resolution=2,
            )
            with mock.patch.object(train_model, "load_dataset", side_effect=_dummy_dataset_loader):
                with mock.patch.object(train_model, "train_model") as mock_train_model:
                    train_model.train_model_from_indices_file(
                        accelerator=object(),
                        config=config,
                        savedir=Path(tmpdir),
                        train_indices_path=path,
                    )
                    args, _ = mock_train_model.call_args
                    train_mask = args[4]
                    expected_mask = utils.index_to_mask(
                        torch.tensor(indices, dtype=torch.long),
                        6,
                    )
                    self.assertTrue(torch.equal(train_mask, expected_mask))

    def test_main_overrides_train_indices_path_from_cli_and_prints_it(self):
        config_text = (
            "save_dir: checkpoints\n"
            "torch_compile: false\n"
            "train_indices_path: from_config.pkl\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "training" / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "test_config.yaml").write_text(config_text)
            with mock.patch.object(train_model.utils, "get_root", return_value=str(root)):
                with mock.patch.object(train_model, "AcceleratorLite", return_value=object()):
                    with mock.patch.object(train_model, "train_model_from_indices_file") as mock_train:
                        with mock.patch("builtins.print") as mock_print:
                            train_model.main(
                                config_file="test_config",
                                suffix="",
                                train_indices_path="from_cli.pkl",
                            )
            train_call_kwargs = mock_train.call_args.kwargs
            self.assertEqual(train_call_kwargs["train_indices_path"], root / "from_cli.pkl")
            printed_config = mock_print.call_args.args[0]
            self.assertIn("train_indices_path: from_cli.pkl", printed_config)
            self.assertNotIn("train_indices_path: from_config.pkl", printed_config)

    def test_main_uses_config_train_indices_path_when_cli_not_given(self):
        config_text = (
            "save_dir: checkpoints\n"
            "torch_compile: false\n"
            "train_indices_path: from_config.pkl\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "training" / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "test_config.yaml").write_text(config_text)
            with mock.patch.object(train_model.utils, "get_root", return_value=str(root)):
                with mock.patch.object(train_model, "AcceleratorLite", return_value=object()):
                    with mock.patch.object(train_model, "train_model_from_indices_file") as mock_train:
                        with mock.patch("builtins.print") as mock_print:
                            train_model.main(
                                config_file="test_config",
                                suffix="",
                            )
            train_call_kwargs = mock_train.call_args.kwargs
            self.assertEqual(train_call_kwargs["train_indices_path"], root / "from_config.pkl")
            printed_config = mock_print.call_args.args[0]
            self.assertIn("train_indices_path: from_config.pkl", printed_config)


if __name__ == "__main__":
    unittest.main()
