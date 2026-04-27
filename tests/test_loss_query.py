import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from generative_models.flow_matching import FlowMatching
from mia import loss_query as loss_query_module
from mia import path_utils


class TestLossQuery(unittest.TestCase):
    def test_compute_averaged_loss_supports_flow_matching(self):
        '''
        Average FlowMatching losses at the configured fixed query time.
        Returns:
            None
        '''
        loss_query = loss_query_module.LossQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_loss_samples=2,
            noise_level=0.25,
        )
        model = FlowMatching(
            image_dim=(1, 4, 4),
            std_min=0.01,
            base_channels=32,
            channel_mult=(1,),
            n_res_blocks_per_level=1,
            n_attention_heads=1,
            attention_resolutions=(4,),
            dropout=0.0,
            resample_with_conv=True,
            use_sdpa=True,
        )
        samples = torch.randn(3, 1, 4, 4)
        first_loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        second_loss = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)

        with patch.object(model, "per_sample_loss", side_effect=[first_loss, second_loss]) as per_sample_loss_fn:
            avg_loss = loss_query.compute_averaged_loss(model, samples)

        expected_loss = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        expected_t = torch.full(size=(3,), fill_value=0.25, dtype=torch.float32)
        self.assertTrue(torch.allclose(avg_loss, expected_loss))
        self.assertEqual(per_sample_loss_fn.call_count, 2)
        for call_args in per_sample_loss_fn.call_args_list:
            called_samples = call_args.args[0]
            called_t = call_args.args[1]
            self.assertTrue(torch.equal(called_samples, samples))
            self.assertTrue(torch.allclose(called_t, expected_t))
            self.assertEqual(called_t.device, samples.device)
            self.assertEqual(called_t.dtype, torch.float32)

    def test_compute_averaged_loss_supports_ddpm_noise_level(self):
        '''
        Query DDPM losses at the configured noise-level-derived step.
        Returns:
            None
        '''
        class DDPM:
            time_steps = 1000

            def per_sample_loss(self, samples, t):
                return torch.ones(samples.shape[0], dtype=torch.float32)

        loss_query = loss_query_module.LossQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_loss_samples=1,
            noise_level=0.25,
        )
        model = DDPM()
        samples = torch.randn(3, 1, 4, 4)

        with patch.object(model, "per_sample_loss", wraps=model.per_sample_loss) as per_sample_loss_fn:
            avg_loss = loss_query.compute_averaged_loss(model, samples)

        expected_loss = torch.ones(3, dtype=torch.float32)
        expected_t = torch.full(size=(3,), fill_value=250, dtype=torch.long)
        called_t = per_sample_loss_fn.call_args.args[1]
        self.assertTrue(torch.allclose(avg_loss, expected_loss))
        self.assertTrue(torch.equal(called_t, expected_t))
        self.assertEqual(called_t.device, samples.device)
        self.assertEqual(called_t.dtype, torch.long)

        endpoint_query = loss_query_module.LossQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_loss_samples=1,
            noise_level=1.0,
        )
        with patch.object(model, "per_sample_loss", wraps=model.per_sample_loss) as per_sample_loss_fn:
            endpoint_query.compute_averaged_loss(model, samples)
        called_t = per_sample_loss_fn.call_args.args[1]
        endpoint_t = torch.full(size=(3,), fill_value=999, dtype=torch.long)
        self.assertTrue(torch.equal(called_t, endpoint_t))

    def test_run_loss_query_saves_loss_signals_for_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            config = loss_query_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "res_dir": tmpdir,
                "n_loss_samples": 10,
                "checkpoint_paths": [target_path],
            })
            dataset = list(range(4))

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(
                    loss_query_module.LossQuery,
                    "query_loss",
                    return_value=(
                        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
                        torch.tensor([0, 1, 0, 1], dtype=torch.bool),
                    ),
                ) as query_loss_fn,
            ):
                saved_paths = loss_query_module.run_loss_query(config=config, device=torch.device("cpu"), noise_level=0.25)

            load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
            query_loss_fn.assert_called_once()
            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 10, 0.25)
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            self.assertTrue(torch.allclose(torch.tensor(payload["loss_sigs"]), torch.tensor([0.1, 0.2, 0.3, 0.4])))
            self.assertEqual(payload["train_mask"], [False, True, False, True])

    def test_run_loss_query_requires_n_loss_samples(self):
        config = loss_query_module.utils.Config({"checkpoint_paths": ["/tmp/model.pth"]})
        with self.assertRaisesRegex(ValueError, "n_loss_samples"):
            loss_query_module.run_loss_query(config=config, device=torch.device("cpu"))

    def test_run_loss_query_rejects_invalid_noise_level(self):
        '''
        Reject loss-query noise levels outside the supported range.
        Returns:
            None
        '''
        config = loss_query_module.utils.Config({"n_loss_samples": 1})
        with self.assertRaisesRegex(ValueError, "noise_level"):
            loss_query_module.run_loss_query(config=config, device=torch.device("cpu"), noise_level=1.1)

    def test_parse_args_accepts_noise_level(self):
        '''
        Parse the loss-query CLI noise-level override.
        Returns:
            None
        '''
        args = loss_query_module.parse_args(["--noise-level", "0.25"])
        self.assertEqual(args.noise_level, 0.25)

    def test_main_prints_effective_settings_after_overrides(self):
        '''
        Print only effective loss-query settings after CLI overrides.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            configured_path = "/tmp/DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth"
            override_path = "/tmp/DDPM-cifar10-smpl-f0p5-s1-sz32-epoch4.pth"
            with open(config_path, "w") as file:
                file.write("\n".join([
                    'dataset: "cifar10"',
                    f'data_dir: "{tmpdir}"',
                    "batch_size: 2",
                    f'res_dir: "{tmpdir}"',
                    "n_loss_samples: 3",
                    "checkpoint_paths:",
                    f'  - "{configured_path}"',
                ]))

            with (
                patch.object(loss_query_module, "run_loss_query", return_value=[]) as run_loss_query_fn,
                patch("builtins.print") as print_fn,
            ):
                loss_query_module.main([
                    "--config",
                    str(config_path),
                    "--checkpoint-paths",
                    override_path,
                    "--noise-level",
                    "0.25",
                ])

            run_loss_query_fn.assert_called_once()
            printed_lines = [args.args[0] for args in print_fn.call_args_list]
            self.assertIn("Loss query settings", printed_lines)
            self.assertIn(f"config_path: {config_path}", printed_lines)
            self.assertIn("device: cpu", printed_lines)
            self.assertIn("dataset: cifar10", printed_lines)
            self.assertIn(f"data_dir: {tmpdir}", printed_lines)
            self.assertIn("batch_size: 2", printed_lines)
            self.assertIn(f"res_dir: {tmpdir}", printed_lines)
            self.assertIn("n_loss_samples: 3", printed_lines)
            self.assertIn("noise_level: 0.25", printed_lines)
            self.assertIn("checkpoint_paths:", printed_lines)
            self.assertIn(f"  - {override_path}", printed_lines)
            self.assertNotIn(f"  - {configured_path}", printed_lines)


if __name__ == "__main__":
    unittest.main()
