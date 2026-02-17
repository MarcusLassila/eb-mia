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
    def test_train_model_uses_checkpoint_configs_except_epochs(self):
        train_mask = torch.tensor([True, True, True, False, False, False], dtype=torch.bool)
        checkpoint = {
            "train_config": {
                "batch_size": 1,
                "simul_batch_size": 1,
                "epochs": 50,
                "epochs_per_checkpoint": 7,
                "lr": 5e-4,
                "weight_decay": 1e-2,
                "use_ema": False,
                "ema_decay": 0.0,
                "grad_clip": 0.0,
                "autocast_dtype": "float16",
                "lr_scheduler": "none",
            },
            "model_config": {
                "image_dim": (1, 2, 2),
                "time_steps": 10,
                "beta_schedule": "linear",
                "base_channels": 32,
                "channel_mult": (1,),
                "n_attention_heads": 1,
                "channels_per_head": 32,
                "attention_resolutions": (2,),
                "dropout": 0.0,
                "resample_with_conv": True,
                "use_sdpa": True,
            },
        }
        config = types.SimpleNamespace(
            val_frac=0.2,
            model="DDPM",
            image_resolution=2,
            base_channels=16,
            channel_mult=(1,),
            n_attention_heads=1,
            channels_per_head=32,
            attention_resolutions=(2,),
            dropout=0.0,
            use_sdpa=True,
            latent_dim=4,
            batch_size=2,
            simul_batch_size=2,
            epochs=3,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="float16",
            lr_scheduler="cosine",
            suffix="",
            resume_checkpoint_path=Path("/tmp/checkpoint.pth"),
            use_checkpoint_configs=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(train_model.torch, "load", return_value=checkpoint):
                model = types.SimpleNamespace(network=torch.nn.Linear(1, 1))
                with mock.patch.object(train_model, "DDPM", return_value=model):
                    with mock.patch.object(train_model, "TrainLoop") as mock_train_loop:
                        train_model.train_model(
                            accelerator=types.SimpleNamespace(print=mock.Mock()),
                            config=config,
                            savedir=Path(tmpdir),
                            dataset=_DummyDataset(),
                            train_mask=train_mask,
                            split_stem="split",
                        )
                        kwargs = mock_train_loop.call_args.kwargs
                        self.assertEqual(kwargs["train_config"].epochs, 3)
                        self.assertEqual(kwargs["train_config"].batch_size, 1)
                        self.assertEqual(kwargs["model_config"], checkpoint["model_config"])
                        self.assertEqual(kwargs["resume_checkpoint_path"], config.resume_checkpoint_path)

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
                with mock.patch.object(train_model, "AcceleratorLite", return_value=types.SimpleNamespace(print=mock.Mock())):
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
                with mock.patch.object(train_model, "AcceleratorLite", return_value=types.SimpleNamespace(print=mock.Mock())):
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

    def test_main_overrides_resume_checkpoint_path_from_cli_and_prints_it(self):
        config_text = (
            "save_dir: checkpoints\n"
            "torch_compile: false\n"
            "train_indices_path: from_config.pkl\n"
            "resume_checkpoint_path: from_config_checkpoint.pth\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "training" / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "test_config.yaml").write_text(config_text)
            with mock.patch.object(train_model.utils, "get_root", return_value=str(root)):
                with mock.patch.object(train_model, "AcceleratorLite", return_value=types.SimpleNamespace(print=mock.Mock())):
                    with mock.patch.object(train_model, "train_model_from_indices_file") as mock_train:
                        with mock.patch("builtins.print") as mock_print:
                            train_model.main(
                                config_file="test_config",
                                suffix="",
                                resume_checkpoint_path="from_cli_checkpoint.pth",
                            )
            train_call_kwargs = mock_train.call_args.kwargs
            self.assertEqual(train_call_kwargs["config"].resume_checkpoint_path, root / "from_cli_checkpoint.pth")
            printed_config = mock_print.call_args.args[0]
            self.assertIn("resume_checkpoint_path: from_cli_checkpoint.pth", printed_config)
            self.assertNotIn("resume_checkpoint_path: from_config_checkpoint.pth", printed_config)


if __name__ == "__main__":
    unittest.main()
