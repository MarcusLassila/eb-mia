import tempfile
import unittest

import torch
from torch.utils.data import Dataset, Subset

from accelerate_lite.accelerate import AcceleratorLite
from generative_models.agm import AbstractGenerativeModel
from training.train_loop import TrainConfig, TrainLoop
from contextlib import nullcontext
from pathlib import Path


class _TensorDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


class _CompileWrappedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._orig_mod = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self._orig_mod(x)


class _SimpleModel(AbstractGenerativeModel):
    def __init__(self):
        super().__init__()
        self.network = _CompileWrappedLinear()

    def move_to(self, device):
        self.network.to(device)

    def per_sample_loss(self, x, *args, **kwargs):
        loss = self.loss(x, *args, **kwargs)
        return loss.repeat(x.shape[0])

    def loss(self, x, autocast_context=nullcontext(), network_override=None, **kwargs):
        network = network_override if network_override is not None else self.network
        with autocast_context:
            out = network(x)
            return (out ** 2).mean()

    @torch.inference_mode()
    def sample(self, batch_size, **kwargs):
        return torch.zeros(batch_size, 1)

    @property
    def image_size(self):
        '''
        Return the model image size.
        Returns:
            int: Spatial size of generated images.
        '''
        return 1


class TestTrainLoopResume(unittest.TestCase):
    def test_train_config_uses_independent_scheduler_param_dicts(self):
        config_a = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
        )
        config_b = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
        )

        config_a.lr_scheduler_params["warmup_steps"] = 10
        self.assertEqual(config_a.lr_scheduler_params, {"warmup_steps": 10})
        self.assertEqual(config_b.lr_scheduler_params, {})

    def test_checkpoint_saves_loss_history(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        model = _SimpleModel()

        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=2,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="bfloat16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "loss_history_test.pth"
            TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
            ).train()
            checkpoint = torch.load(Path(tmpdir) / "loss_history_test-epoch2.pth", map_location="cpu", weights_only=False)
        self.assertIn("train_losses", checkpoint)
        self.assertIn("val_losses", checkpoint)
        self.assertEqual(len(checkpoint["train_losses"]), 2)
        self.assertEqual(len(checkpoint["val_losses"]), 2)
        self.assertTrue(all(isinstance(loss, float) for loss in checkpoint["train_losses"]))
        self.assertTrue(all(isinstance(loss, float) for loss in checkpoint["val_losses"]))

    def test_linear_scheduler_uses_configured_dict_params(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        model = _SimpleModel()

        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=2,
            epochs_per_checkpoint=1,
            lr=1e-3,
            lr_scheduler="linear",
            lr_scheduler_params={"warmup_steps": 2, "min_lr": 1e-4},
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "poly_scheduler_test.pth"
            train_loop = TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
            )

        lr_multiplier = train_loop.scheduler.lr_lambdas[0]
        self.assertAlmostEqual(lr_multiplier(0), 0.1)
        self.assertAlmostEqual(lr_multiplier(1), 0.55)

    def test_linear_scheduler_steps_per_optimizer_step(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        model = _SimpleModel()

        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            lr_scheduler="linear",
            lr_scheduler_params={"warmup_steps": 0, "min_lr": 1e-4},
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "poly_scheduler_steps_test.pth"
            TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
            ).train()
            checkpoint = torch.load(Path(tmpdir) / "poly_scheduler_steps_test-epoch1.pth", map_location="cpu", weights_only=False)

        self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], 2)

    def test_linear_scheduler_asserts_when_warmup_matches_total_steps(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        model = _SimpleModel()

        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            lr_scheduler="linear",
            lr_scheduler_params={"warmup_steps": 2},
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "poly_scheduler_assert_test.pth"
            with self.assertRaises(AssertionError):
                TrainLoop(
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    train_config=train_config,
                    model_config={"name": "simple"},
                    accelerator=accelerator,
                    savepath=savepath,
                )

    def test_resume_checkpoint_restores_optimizer_lr_from_scheduler_state(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            lr_scheduler="linear",
            lr_scheduler_params={"warmup_steps": 0, "min_lr": 1e-4},
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "resume_scheduler_lr_test.pth"
            initial_loop = TrainLoop(
                model=_SimpleModel(),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
            )
            for _ in range(2):
                initial_loop.optimizer.step()
                initial_loop.scheduler.step()
            checkpoint = {
                "epoch": 1,
                "raw_network_state_dict": initial_loop.raw_network.state_dict(),
                "network_state_dict": initial_loop.raw_network.state_dict(),
                "optimizer_state_dict": initial_loop.optimizer.state_dict(),
                "scheduler_state_dict": initial_loop.scheduler.state_dict(),
                "train_config": train_config,
            }

            resumed_loop = TrainLoop(
                model=_SimpleModel(),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
                resume_checkpoint=checkpoint,
            )

        self.assertAlmostEqual(
            resumed_loop.optimizer.param_groups[0]["lr"],
            resumed_loop.scheduler.get_last_lr()[0],
        )

    def test_resume_checkpoint_loads_into_compiled_wrapper(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        model = _SimpleModel()

        expected_weight = torch.tensor([[3.0]])
        checkpoint = {
            "epoch": 5,
            "raw_network_state_dict": {"weight": expected_weight.clone()},
            "network_state_dict": {"weight": expected_weight.clone()},
        }
        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="bfloat16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "resume_test.pth"
            train_loop = TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
                resume_checkpoint=checkpoint,
            )
        self.assertEqual(train_loop.start_epoch, 5)
        self.assertTrue(torch.equal(train_loop.raw_network._orig_mod.weight.detach().cpu(), expected_weight))

    def test_resume_checkpoint_preserves_existing_loss_history(self):
        torch.manual_seed(0)
        data = torch.randn(4, 1)
        dataset = _TensorDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 2))
        val_dataset = Subset(dataset, indices=torch.arange(2, 4))
        model = _SimpleModel()

        train_config = TrainConfig(
            batch_size=1,
            simul_batch_size=1,
            epochs=1,
            epochs_per_checkpoint=1,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="bfloat16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "resume_loss_history.pth"
            first_loop = TrainLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
            )
            first_loop.train()
            checkpoint = torch.load(Path(tmpdir) / "resume_loss_history-epoch1.pth", map_location="cpu", weights_only=False)
            resumed_loop = TrainLoop(
                model=_SimpleModel(),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_config=train_config,
                model_config={"name": "simple"},
                accelerator=accelerator,
                savepath=savepath,
                resume_checkpoint=checkpoint,
            )
            resumed_loop.train()
            resumed_checkpoint = torch.load(Path(tmpdir) / "resume_loss_history-epoch2.pth", map_location="cpu", weights_only=False)
        self.assertEqual(len(resumed_checkpoint["train_losses"]), 2)
        self.assertEqual(len(resumed_checkpoint["val_losses"]), 2)
        self.assertAlmostEqual(resumed_checkpoint["train_losses"][0], checkpoint["train_losses"][0])
        self.assertAlmostEqual(resumed_checkpoint["val_losses"][0], checkpoint["val_losses"][0])


if __name__ == "__main__":
    unittest.main()
