import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from generative_models.flow_matching import FlowMatching
from mia import loss_query as loss_query_module
from mia import path_utils
from mia import utils as mia_utils


class TestLossQuery(unittest.TestCase):
    def test_compute_loss_samples_supports_flow_matching(self):
        '''
        Query and sort FlowMatching loss samples at the configured fixed query time.
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
        first_loss = torch.tensor([3.0, 2.0, 5.0], dtype=torch.float32)
        second_loss = torch.tensor([1.0, 4.0, 3.0], dtype=torch.float32)

        with patch.object(model, "per_sample_loss", side_effect=[first_loss, second_loss]) as per_sample_loss_fn:
            loss_samples = loss_query.compute_loss_samples(model, samples)

        expected_loss_samples = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]], dtype=torch.float32)
        expected_t = torch.full(size=(3,), fill_value=0.75, dtype=torch.float32)
        self.assertTrue(torch.allclose(loss_samples, expected_loss_samples))
        self.assertEqual(per_sample_loss_fn.call_count, 2)
        for call_args in per_sample_loss_fn.call_args_list:
            called_samples = call_args.args[0]
            called_t = call_args.args[1]
            self.assertTrue(torch.equal(called_samples, samples))
            self.assertTrue(torch.allclose(called_t, expected_t))
            self.assertEqual(called_t.device, samples.device)
            self.assertEqual(called_t.dtype, torch.float32)

    def test_compute_loss_samples_supports_ddpm_noise_level(self):
        '''
        Query and sort DDPM losses at the configured noise-level-derived step.
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
            n_loss_samples=2,
            noise_level=0.25,
        )
        model = DDPM()
        samples = torch.randn(3, 1, 4, 4)
        first_loss = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32)
        second_loss = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32)

        with patch.object(model, "per_sample_loss", side_effect=[first_loss, second_loss]) as per_sample_loss_fn:
            loss_samples = loss_query.compute_loss_samples(model, samples)

        expected_loss_samples = torch.tensor([[1.0, 2.0], [1.0, 3.0], [2.0, 3.0]], dtype=torch.float32)
        expected_t = torch.full(size=(3,), fill_value=250, dtype=torch.long)
        called_t = per_sample_loss_fn.call_args.args[1]
        self.assertTrue(torch.allclose(loss_samples, expected_loss_samples))
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
            endpoint_query.compute_loss_samples(model, samples)
        called_t = per_sample_loss_fn.call_args.args[1]
        endpoint_t = torch.full(size=(3,), fill_value=999, dtype=torch.long)
        self.assertTrue(torch.equal(called_t, endpoint_t))

    def test_compute_loss_samples_supports_vae(self):
        '''
        Query VAE losses as individual samples without keeping VAE internal averaging.
        Returns:
            None
        '''
        class VAE:
            def __init__(self):
                self.n_rsamples = 5
                self.n_rsamples_seen = []
                self.losses = [
                    torch.tensor([4.0, 1.0, 3.0], dtype=torch.float32),
                    torch.tensor([2.0, 5.0, 2.0], dtype=torch.float32),
                ]

            def per_sample_loss(self, samples):
                self.n_rsamples_seen.append(self.n_rsamples)
                return self.losses.pop(0)

        loss_query = loss_query_module.LossQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_loss_samples=2,
            noise_level=0.25,
        )
        model = VAE()
        model.n_rsamples = 1
        samples = torch.randn(3, 1, 4, 4)

        loss_samples = loss_query.compute_loss_samples(model, samples)

        expected_loss_samples = torch.tensor([[2.0, 4.0], [1.0, 5.0], [2.0, 3.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(loss_samples, expected_loss_samples))
        self.assertEqual(model.n_rsamples_seen, [1, 1])
        self.assertEqual(model.n_rsamples, 1)

    def test_run_loss_query_saves_loss_signals_for_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(4))

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(
                    loss_query_module.LossQuery,
                    "query_loss",
                    return_value=(
                        torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5], [0.4, 0.6]], dtype=torch.float32),
                        torch.tensor([0, 1, 0, 1], dtype=torch.bool),
                    ),
                ) as query_loss_fn,
            ):
                saved_paths = loss_query_module.run_loss_query(
                    checkpoint_paths=[target_path],
                    checkpoint_properties=loss_query_module.utils.parse_properties_from_checkpoint_path(target_path),
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_loss_samples=10,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                )

            load_dataset_fn.assert_called_once_with("CIFAR10", data_dir=tmpdir, size=32)
            query_loss_fn.assert_called_once()
            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 10, 0.25)
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            expected_loss_sigs = torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5], [0.4, 0.6]])
            self.assertTrue(torch.allclose(torch.tensor(payload["loss_sigs"]), expected_loss_sigs))
            self.assertEqual(payload["train_mask"], [False, True, False, True])

    def test_run_loss_query_infers_dataset_from_checkpoint_path(self):
        '''
        Use the checkpoint dataset token when no explicit dataset is provided.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(4))

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(
                    loss_query_module.LossQuery,
                    "query_loss",
                    return_value=(
                        torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5], [0.4, 0.6]], dtype=torch.float32),
                        torch.tensor([0, 1, 0, 1], dtype=torch.bool),
                    ),
                ),
            ):
                loss_query_module.run_loss_query(
                    checkpoint_paths=[target_path],
                    checkpoint_properties=loss_query_module.utils.parse_properties_from_checkpoint_path(target_path),
                    dataset=None,
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_loss_samples=10,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                )

            load_dataset_fn.assert_called_once_with("CIFAR10", data_dir=tmpdir, size=32)

    def test_load_loss_signals_averages_saved_loss_samples(self):
        '''
        Average 2D saved loss samples when loading signals for existing audits.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            loss_path = Path(tmpdir) / "loss_signals.pkl"
            with open(loss_path, "wb") as file:
                pickle.dump(
                    {
                        "loss_sigs": [[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]],
                        "train_mask": [True, False, True],
                    },
                    file,
                )

            loss_sigs, train_mask = mia_utils.load_loss_signals(loss_path)

            self.assertTrue(torch.allclose(loss_sigs, torch.tensor([2.0, 3.0, 6.0])))
            self.assertTrue(torch.equal(train_mask, torch.tensor([True, False, True])))

    def test_run_loss_query_returns_no_paths_for_empty_checkpoint_paths(self):
        checkpoint_properties = {
            "dataset": "cifar10",
            "size": 32,
        }

        with patch.object(loss_query_module, "load_dataset", return_value=[]):
            saved_paths = loss_query_module.run_loss_query(
                checkpoint_paths=[],
                checkpoint_properties=checkpoint_properties,
                dataset="CIFAR10",
                data_dir="/tmp/data",
                batch_size=2,
                res_dir="/tmp/results",
                n_loss_samples=2,
                device=torch.device("cpu"),
                noise_level=0.1,
            )

        self.assertEqual(saved_paths, [])

    def test_parse_args_requires_n_loss_samples(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            loss_query_module.parse_args([
                "--checkpoint-paths",
                "/tmp/DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth",
                "--batch-size",
                "2",
                "--noise-level",
                "0.25",
            ])

    def test_parse_args_accepts_noise_level(self):
        '''
        Parse the loss-query CLI noise-level override.
        Returns:
            None
        '''
        args = loss_query_module.parse_args([
            "--checkpoint-paths",
            "/tmp/DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth",
            "--dataset",
            "CIFAR10",
            "--batch-size",
            "2",
            "--n-loss-samples",
            "3",
            "--noise-level",
            "0.25",
        ])
        self.assertEqual(args.noise_level, 0.25)

    def test_main_prints_effective_settings(self):
        '''
        Print effective loss-query settings from CLI arguments.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            override_res_dir = str(Path(tmpdir) / "override")
            override_path = "/tmp/DDPM-cifar10-smpl-f0p5-s1-sz32-epoch4.pth"

            with (
                patch.object(loss_query_module, "run_loss_query", return_value=[]) as run_loss_query_fn,
                patch("builtins.print") as print_fn,
            ):
                loss_query_module.main([
                    "--checkpoint-paths",
                    override_path,
                    "--dataset",
                    "cifar10",
                    "--data-dir",
                    tmpdir,
                    "--batch-size",
                    "2",
                    "--n-loss-samples",
                    "3",
                    "--noise-level",
                    "0.25",
                    "--res-dir",
                    override_res_dir,
                ])

            run_loss_query_fn.assert_called_once()
            printed_lines = [args.args[0] for args in print_fn.call_args_list]
            self.assertIn("Loss query settings", printed_lines)
            self.assertIn("device: cpu", printed_lines)
            self.assertIn("dataset: CIFAR10", printed_lines)
            self.assertIn(f"data_dir: {tmpdir}", printed_lines)
            self.assertIn("batch_size: 2", printed_lines)
            self.assertIn(f"res_dir: {override_res_dir}", printed_lines)
            self.assertIn("n_loss_samples: 3", printed_lines)
            self.assertIn("noise_level: 0.25", printed_lines)
            self.assertIn("checkpoint_paths:", printed_lines)
            self.assertIn(f"  - {override_path}", printed_lines)
            kwargs = run_loss_query_fn.call_args.kwargs
            self.assertEqual(kwargs["checkpoint_paths"], [Path(override_path)])
            self.assertEqual(kwargs["dataset"], "CIFAR10")
            self.assertEqual(kwargs["data_dir"], Path(tmpdir))
            self.assertEqual(kwargs["batch_size"], 2)
            self.assertEqual(kwargs["res_dir"], Path(override_res_dir))
            self.assertEqual(kwargs["n_loss_samples"], 3)
            self.assertEqual(kwargs["noise_level"], 0.25)

    def test_main_allows_omitted_dataset(self):
        '''
        Allow CLI loss querying without an explicit dataset argument.
        Returns:
            None
        '''
        override_path = "/tmp/DDPM-cifar10-smpl-f0p5-s1-sz32-epoch4.pth"

        with (
            patch.object(loss_query_module, "run_loss_query", return_value=[]) as run_loss_query_fn,
            patch("builtins.print") as print_fn,
        ):
            loss_query_module.main([
                "--checkpoint-paths",
                override_path,
                "--batch-size",
                "2",
                "--n-loss-samples",
                "3",
                "--noise-level",
                "0.25",
            ])

        kwargs = run_loss_query_fn.call_args.kwargs
        printed_lines = [args.args[0] for args in print_fn.call_args_list]
        self.assertIn("dataset: CIFAR10", printed_lines)
        self.assertEqual(kwargs["checkpoint_paths"], [Path(override_path)])
        self.assertEqual(kwargs["dataset"], "CIFAR10")


if __name__ == "__main__":
    unittest.main()
