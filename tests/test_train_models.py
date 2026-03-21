import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import Dataset

import training.train_model as train_model


class _DummyDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return torch.zeros(1, 2, 2)


def _dummy_dataset_loader(dataset_name=None, data_dir=None, transform=None, **kwargs):
    return _DummyDataset()


class _DummyAccelerator:
    def print(self, *args, **kwargs):
        return None


class _DummyModel:
    def __init__(self):
        self.network = torch.nn.Linear(1, 1)


class TestTrainModels(unittest.TestCase):
    def test_train_model_from_scratch_passes_ddpm_res_block_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_indices.pkl"
            with open(path, "wb") as file:
                pickle.dump({"indices": [0, 2, 4], "len_dataset": 6}, file)
            config = types.SimpleNamespace(
                dataset="Dummy",
                val_frac=0.2,
                model="DDPM",
                image_resolution=2,
                suffix="",
                batch_size=2,
                simul_batch_size=2,
                epochs=1,
                epochs_per_checkpoint=1,
                lr=1e-3,
                weight_decay=0.0,
                ema_decay=0.0,
                grad_clip=0.0,
                autocast_dtype="float16",
                lr_scheduler="none",
                base_channels=32,
                channel_mult=(1,),
                n_res_blocks_per_level=2,
                n_attention_heads=1,
                channels_per_head=None,
                attention_resolutions=(2,),
                dropout=0.0,
                use_sdpa=True,
            )
            with mock.patch.object(train_model, "load_dataset", side_effect=_dummy_dataset_loader):
                with mock.patch.object(train_model, "DDPM", return_value=_DummyModel()) as mock_ddpm:
                    with mock.patch.object(train_model, "TrainLoop") as mock_loop:
                        mock_loop.return_value.train.return_value = None
                        train_model.train_model_from_scratch(
                            accelerator=_DummyAccelerator(),
                            config=config,
                            data_dir=Path(tmpdir),
                            savedir=Path(tmpdir),
                            train_indices_path=path,
                        )
            self.assertEqual(mock_ddpm.call_args.kwargs["n_res_blocks_per_level"], 2)

    def test_train_model_from_scratch_uses_indices_file(self):
        indices = [0, 2, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_indices.pkl"
            with open(path, "wb") as file:
                pickle.dump({"indices": indices, "len_dataset": 6}, file)
            config = types.SimpleNamespace(
                dataset="Dummy",
                val_frac=0.2,
                model="VAE",
                image_resolution=2,
                latent_dim=2,
                suffix="",
                batch_size=2,
                simul_batch_size=2,
                epochs=1,
                epochs_per_checkpoint=1,
                lr=1e-3,
                weight_decay=0.0,
                ema_decay=0.0,
                grad_clip=0.0,
                autocast_dtype="float16",
                lr_scheduler="none",
            )
            with mock.patch.object(train_model, "load_dataset", side_effect=_dummy_dataset_loader):
                with mock.patch.object(train_model, "VAE", return_value=_DummyModel()):
                    with mock.patch.object(train_model, "TrainLoop") as mock_loop:
                        mock_loop.return_value.train.return_value = None
                        train_model.train_model_from_scratch(
                            accelerator=_DummyAccelerator(),
                            config=config,
                            data_dir=Path(tmpdir),
                            savedir=Path(tmpdir),
                            train_indices_path=path,
                        )
                    train_kwargs = mock_loop.call_args.kwargs
                    train_dataset = train_kwargs["train_dataset"]
                    val_dataset = train_kwargs["val_dataset"]
                    self.assertTrue(torch.equal(train_dataset.indices, torch.tensor(indices, dtype=torch.long)))
                    self.assertTrue(torch.equal(val_dataset.indices, torch.tensor([1], dtype=torch.long)))

    def test_train_model_from_scratch_passes_lr_scheduler_params_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_indices.pkl"
            with open(path, "wb") as file:
                pickle.dump({"indices": [0, 2, 4], "len_dataset": 6}, file)
            config = types.SimpleNamespace(
                dataset="Dummy",
                val_frac=0.2,
                model="VAE",
                image_resolution=2,
                latent_dim=2,
                suffix="",
                batch_size=2,
                simul_batch_size=2,
                epochs=1,
                epochs_per_checkpoint=1,
                lr=1e-3,
                weight_decay=0.0,
                ema_decay=0.0,
                grad_clip=0.0,
                autocast_dtype="float16",
                lr_scheduler="linear",
                lr_scheduler_params={"warmup_steps": 5, "min_lr": 1e-4},
            )
            with mock.patch.object(train_model, "load_dataset", side_effect=_dummy_dataset_loader):
                with mock.patch.object(train_model, "VAE", return_value=_DummyModel()):
                    with mock.patch.object(train_model, "TrainLoop") as mock_loop:
                        mock_loop.return_value.train.return_value = None
                        train_model.train_model_from_scratch(
                            accelerator=_DummyAccelerator(),
                            config=config,
                            data_dir=Path(tmpdir),
                            savedir=Path(tmpdir),
                            train_indices_path=path,
                        )
            train_config = mock_loop.call_args.kwargs["train_config"]
            self.assertEqual(train_config.lr_scheduler, "linear")
            self.assertEqual(train_config.lr_scheduler_params, {"warmup_steps": 5, "min_lr": 1e-4})

    def test_train_model_from_checkpoint_overrides_train_indices_from_cli(self):
        checkpoint = {
            "train_indices": [0, 1],
            "val_indices": [2, 3],
            "train_config": {
                "batch_size": 2,
                "simul_batch_size": 2,
                "epochs": 1,
                "epochs_per_checkpoint": 1,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "use_ema": False,
                "ema_decay": 0.0,
                "grad_clip": 0.0,
                "autocast_dtype": "float16",
                "lr_scheduler": "none",
            },
            "model_config": {
                "in_ch": 1,
                "in_dim": 2,
                "latent_dim": 2,
                "n_rsamples": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "VAE-dummy-smpl-f0p5-s0-sz2-gray-epoch10.pth"
            train_indices_path = Path(tmpdir) / "override.pkl"
            with open(train_indices_path, "wb") as file:
                pickle.dump({"indices": [0, 2, 4], "len_dataset": 6}, file)
            with mock.patch.object(train_model, "load_dataset", side_effect=_dummy_dataset_loader):
                with mock.patch.object(train_model, "VAE", return_value=_DummyModel()):
                    with mock.patch.object(train_model, "TrainLoop") as mock_loop:
                        mock_loop.return_value.train.return_value = None
                        train_model.train_model_from_checkpoint(
                            accelerator=_DummyAccelerator(),
                            config=None,
                            savedir=Path(tmpdir),
                            data_dir=Path(tmpdir),
                            checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            train_indices_path=train_indices_path,
                        )
                    train_kwargs = mock_loop.call_args.kwargs
                    self.assertEqual(train_kwargs["savepath"], Path(tmpdir) / "VAE-dummy-smpl-f0p5-s0-sz2-gray.pth")
                    self.assertEqual(list(train_kwargs["train_dataset"].indices), [0, 2, 4])
                    self.assertTrue(torch.equal(train_kwargs["val_dataset"].indices, torch.tensor([1, 3], dtype=torch.long)))

    def test_main_passes_cli_train_indices_to_scratch_training(self):
        config_text = (
            "torch_compile: false\n"
            "dataset: Dummy\n"
            "image_resolution: 2\n"
            "val_frac: 0.2\n"
            "model: VAE\n"
            "latent_dim: 2\n"
            "batch_size: 2\n"
            "simul_batch_size: 2\n"
            "epochs: 1\n"
            "epochs_per_checkpoint: 1\n"
            "lr: 0.001\n"
            "weight_decay: 0.0\n"
            "ema_decay: 0.0\n"
            "grad_clip: 0.0\n"
            "autocast_dtype: float16\n"
            "lr_scheduler: none\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "training" / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "test_config.yaml").write_text(config_text)
            with mock.patch.object(train_model.utils, "get_root", return_value=str(root)):
                with mock.patch.object(train_model, "AcceleratorLite", return_value=_DummyAccelerator()):
                    with mock.patch.object(train_model, "train_model_from_scratch") as mock_train:
                        train_model.main(
                            config_file="test_config",
                            suffix="",
                            train_indices_path="from_cli.pkl",
                            data_dir="datasets",
                            save_dir="checkpoints",
                        )
            train_call_kwargs = mock_train.call_args.kwargs
            self.assertEqual(train_call_kwargs["train_indices_path"], root / "from_cli.pkl")
            self.assertEqual(train_call_kwargs["data_dir"], root / "datasets")
            self.assertEqual(train_call_kwargs["savedir"], root / "checkpoints")

    def test_main_resume_without_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = {"train_config": {}, "model_config": {}}
            with mock.patch.object(train_model.utils, "get_root", return_value=str(root)):
                with mock.patch.object(train_model.utils, "load_checkpoint", return_value=checkpoint):
                    with mock.patch.object(train_model, "AcceleratorLite", return_value=_DummyAccelerator()):
                        with mock.patch.object(train_model, "train_model_from_checkpoint") as mock_train:
                            train_model.main(
                                checkpoint_path="checkpoint.pth",
                                data_dir="datasets",
                                save_dir="checkpoints",
                            )
            train_call_kwargs = mock_train.call_args.kwargs
            self.assertIsNone(train_call_kwargs["config"])
            self.assertEqual(train_call_kwargs["checkpoint_path"], root / "checkpoint.pth")
            self.assertEqual(train_call_kwargs["data_dir"], root / "datasets")
            self.assertEqual(train_call_kwargs["savedir"], root / "checkpoints")

    def test_main_resume_without_config_allows_cli_torch_compile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = {"train_config": {}, "model_config": {}}
            with mock.patch.object(train_model.utils, "get_root", return_value=str(root)):
                with mock.patch.object(train_model.utils, "load_checkpoint", return_value=checkpoint):
                    with mock.patch.object(train_model, "AcceleratorLite", return_value=_DummyAccelerator()) as mock_accelerator:
                        with mock.patch.object(train_model, "train_model_from_checkpoint"):
                            train_model.main(
                                checkpoint_path="checkpoint.pth",
                                data_dir="datasets",
                                save_dir="checkpoints",
                                torch_compile=True,
                            )
            self.assertTrue(mock_accelerator.call_args.kwargs["torch_compile"])

    def test_main_resume_with_config_and_cli_train_indices(self):
        config_text = (
            "torch_compile: false\n"
            "dataset: Dummy\n"
            "image_resolution: 2\n"
            "val_frac: 0.2\n"
            "model: VAE\n"
            "latent_dim: 2\n"
            "batch_size: 2\n"
            "simul_batch_size: 2\n"
            "epochs: 1\n"
            "epochs_per_checkpoint: 1\n"
            "lr: 0.001\n"
            "weight_decay: 0.0\n"
            "ema_decay: 0.0\n"
            "grad_clip: 0.0\n"
            "autocast_dtype: float16\n"
            "lr_scheduler: none\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "training" / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "test_config.yaml").write_text(config_text)
            checkpoint = {"train_config": {}, "model_config": {}}
            with mock.patch.object(train_model.utils, "get_root", return_value=str(root)):
                with mock.patch.object(train_model.utils, "load_checkpoint", return_value=checkpoint):
                    with mock.patch.object(train_model, "AcceleratorLite", return_value=_DummyAccelerator()):
                        with mock.patch.object(train_model, "train_model_from_checkpoint") as mock_train:
                            train_model.main(
                                config_file="test_config",
                                train_indices_path="override.pkl",
                                checkpoint_path="checkpoint.pth",
                                data_dir="datasets",
                                save_dir="checkpoints",
                            )
            train_call_kwargs = mock_train.call_args.kwargs
            self.assertEqual(train_call_kwargs["train_indices_path"], root / "override.pkl")
            self.assertEqual(train_call_kwargs["checkpoint_path"], root / "checkpoint.pth")
            self.assertIsNotNone(train_call_kwargs["config"])


if __name__ == "__main__":
    unittest.main()
