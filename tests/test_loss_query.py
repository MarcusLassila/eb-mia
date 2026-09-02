import io
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

torchdiffeq_stub = types.ModuleType("torchdiffeq")

def odeint_stub(func, y0, t, *args, **kwargs):
    return torch.stack([y0 for _ in range(t.shape[0])], dim=0)

torchdiffeq_stub.odeint = odeint_stub
sys.modules.setdefault("torchdiffeq", torchdiffeq_stub)

from generative_models.ddpm import DDPM as DDPMModel
from generative_models.flow_matching import FlowMatching
from mia import loss_query as loss_query_module
from mia import path_utils
from mia import utils as mia_utils


class TestLossQuery(unittest.TestCase):
    def test_loss_signal_supports_flow_matching(self):
        '''
        Query FlowMatching loss samples at the configured fixed query time.
        Returns:
            None
        '''
        loss_query = loss_query_module.LossQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_samples=2,
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
            loss_samples = loss_query.loss_signal(model, samples)

        expected_loss_samples = torch.tensor([[3.0, 1.0], [2.0, 4.0], [5.0, 3.0]], dtype=torch.float32)
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

    def test_loss_signal_supports_ddpm_noise_level(self):
        '''
        Query DDPM losses at the configured noise-level-derived step.
        Returns:
            None
        '''
        class DDPM(loss_query_module.AbstractDiffusionModel):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.values = [
                    torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32),
                    torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32),
                ]

            @property
            def image_size(self):
                return 4

            def move_to(self, device):
                return None

            def per_sample_loss(self, samples, *args, **kwargs):
                return torch.ones(samples.shape[0], dtype=torch.float32)

            def loss(self, samples, *args, **kwargs):
                return torch.tensor(0.0)

            def fixed_noise_level_per_sample_loss(self, samples, noise_level):
                self.calls.append((samples, noise_level))
                return self.values.pop(0)

            def sample(self, batch_size, **kwargs):
                return torch.zeros(batch_size, 1, 4, 4)

        loss_query = loss_query_module.LossQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_samples=2,
            noise_level=0.25,
        )
        model = DDPM()
        samples = torch.randn(3, 1, 4, 4)

        loss_samples = loss_query.loss_signal(model, samples)

        expected_loss_samples = torch.tensor([[2.0, 1.0], [1.0, 3.0], [3.0, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(loss_samples, expected_loss_samples))
        self.assertEqual(len(model.calls), 2)
        for called_samples, noise_level in model.calls:
            self.assertTrue(torch.equal(called_samples, samples))
            self.assertEqual(noise_level, 0.25)

        endpoint_query = loss_query_module.LossQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_samples=1,
            noise_level=1.0,
        )
        endpoint_model = DDPM()
        endpoint_query.loss_signal(endpoint_model, samples)
        self.assertEqual(endpoint_model.calls[0][1], 1.0)

    def test_loss_signal_supports_vae(self):
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
            n_samples=2,
            noise_level=0.25,
        )
        model = VAE()
        model.n_rsamples = 1
        samples = torch.randn(3, 1, 4, 4)

        loss_samples = loss_query.loss_signal(model, samples)

        expected_loss_samples = torch.tensor([[4.0, 2.0], [1.0, 5.0], [3.0, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(loss_samples, expected_loss_samples))
        self.assertEqual(model.n_rsamples_seen, [1, 1])
        self.assertEqual(model.n_rsamples, 1)

    def test_l4_norm_signal_supports_ddpm(self):
        '''
        Query DDPM denoiser L4 norm samples without sorting Monte Carlo draws.
        Returns:
            None
        '''
        class DDPM:
            def __init__(self):
                self.calls = []
                self.values = [
                    torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32),
                    torch.tensor([5.0, 4.0, 6.0], dtype=torch.float32),
                ]

            def denoiser_norm(self, x, noise_level, lp_norm=4):
                self.calls.append((x, noise_level, lp_norm))
                return self.values.pop(0)

        l4_query = loss_query_module.L4NormQuery(
            batch_size=2,
            device=torch.device("cpu"),
            n_samples=2,
            noise_level=0.25,
        )
        model = DDPM()
        samples = torch.randn(3, 1, 4, 4)

        l4_samples = l4_query.l4_norm_signal(model, samples)

        expected_l4_samples = torch.tensor([[2.0, 5.0], [1.0, 4.0], [3.0, 6.0]])
        self.assertTrue(torch.allclose(l4_samples, expected_l4_samples))
        self.assertEqual(len(model.calls), 2)
        for called_samples, noise_level, lp_norm in model.calls:
            self.assertTrue(torch.equal(called_samples, samples))
            self.assertEqual(noise_level, 0.25)
            self.assertEqual(lp_norm, 4)

    def test_run_loss_query_saves_loss_signals_for_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(4))

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(
                    loss_query_module.LossQuery,
                    "query",
                    return_value=(
                        torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5], [0.4, 0.6]], dtype=torch.float32),
                        torch.tensor([0, 1, 0, 1], dtype=torch.bool),
                    ),
                ) as query_fn,
            ):
                saved_paths = loss_query_module.run_loss_query(
                    checkpoint_paths=[target_path],
                    checkpoint_properties=loss_query_module.utils.parse_properties_from_checkpoint_path(target_path),
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_samples=10,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                )

            load_dataset_fn.assert_called_once_with("CIFAR10", data_dir=tmpdir, size=32)
            query_fn.assert_called_once()
            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 10, 0.25)
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            expected_loss_sigs = torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5], [0.4, 0.6]])
            self.assertTrue(torch.allclose(torch.tensor(payload["loss_sigs"]), expected_loss_sigs))
            self.assertEqual(payload["train_mask"], [False, True, False, True])

    def test_query_loss_can_use_first_n_data_points(self):
        '''
        Query only the first n dataset rows and slice the membership mask.
        Returns:
            None
        '''
        class DummyModel:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(5))
            loss_query = loss_query_module.LossQuery(
                batch_size=2,
                device=torch.device("cpu"),
                n_samples=2,
                noise_level=0.25,
            )

            with (
                patch.object(loss_query, "load_model", return_value=(DummyModel(), torch.tensor([0, 2, 4]))),
                patch.object(
                    loss_query,
                    "loss_signal",
                    side_effect=[
                        torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32),
                        torch.tensor([[4.0, 5.0]], dtype=torch.float32),
                    ],
                ) as loss_signal_fn,
            ):
                loss_sig, train_mask = loss_query.query(dataset, target_path, n_data_points=3)

            self.assertTrue(torch.allclose(loss_sig, torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])))
            self.assertTrue(torch.equal(train_mask, torch.tensor([True, False, True])))
            called_batches = [call_args.args[1] for call_args in loss_signal_fn.call_args_list]
            self.assertTrue(torch.equal(called_batches[0], torch.tensor([0, 1])))
            self.assertTrue(torch.equal(called_batches[1], torch.tensor([2])))

    def test_query_l4_norm_can_use_first_n_data_points(self):
        '''
        Query only the first n dataset rows and slice the membership mask for L4 norms.
        Returns:
            None
        '''
        class DummyModel:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(5))
            l4_query = loss_query_module.L4NormQuery(
                batch_size=2,
                device=torch.device("cpu"),
                n_samples=2,
                noise_level=0.25,
            )

            with (
                patch.object(l4_query, "load_model", return_value=(DummyModel(), torch.tensor([0, 2, 4]))),
                patch.object(
                    l4_query,
                    "l4_norm_signal",
                    side_effect=[
                        torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32),
                        torch.tensor([[4.0, 5.0]], dtype=torch.float32),
                    ],
                ) as l4_norm_signal_fn,
            ):
                signal, train_mask = l4_query.query(dataset, target_path, n_data_points=3)

            self.assertTrue(torch.allclose(signal, torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])))
            self.assertTrue(torch.equal(train_mask, torch.tensor([True, False, True])))
            called_batches = [call_args.args[1] for call_args in l4_norm_signal_fn.call_args_list]
            self.assertTrue(torch.equal(called_batches[0], torch.tensor([0, 1])))
            self.assertTrue(torch.equal(called_batches[1], torch.tensor([2])))

    def test_pia_signal_passes_normalization_choice(self):
        '''
        Forward the PIA normalization flag into the DDPM query.
        Returns:
            None
        '''
        class DDPM:
            def __init__(self):
                self.calls = []

            def pia_score(self, x, noise_level, lp_norm=4, normalize=True):
                self.calls.append((x, noise_level, lp_norm, normalize))
                return torch.tensor([1.0, 2.0], dtype=torch.float32)

        x = torch.randn(2, 1, 4, 4)
        model = DDPM()
        pia_query = loss_query_module.PIAQuery(
            batch_size=2,
            device=torch.device("cpu"),
            noise_level=0.2,
            normalize=False,
        )

        signal = pia_query.pia_signal(model, x)

        self.assertTrue(torch.allclose(signal, torch.tensor([[1.0], [2.0]])))
        self.assertEqual(model.calls, [(x, 0.2, 4, False)])

    def test_t_error_signal_supports_ddpm(self):
        '''
        Query deterministic t-error scores for a DDPM batch.
        Returns:
            None
        '''
        class DDPM:
            def __init__(self):
                self.calls = []

            def t_error(self, x, noise_level):
                self.calls.append((x, noise_level))
                return torch.tensor([1.5, 2.5], dtype=torch.float32)

        x = torch.randn(2, 1, 4, 4)
        model = DDPM()
        t_error_query = loss_query_module.TErrorQuery(
            batch_size=2,
            device=torch.device("cpu"),
            noise_level=0.2,
        )

        signal = t_error_query.t_error(model, x)

        self.assertTrue(torch.allclose(signal, torch.tensor([[1.5], [2.5]])))
        self.assertEqual(model.calls, [(x, 0.2)])

    def test_query_t_error_can_use_first_n_data_points(self):
        '''
        Query only the first n dataset rows and slice the membership mask for t-error.
        Returns:
            None
        '''
        class DummyModel:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(5))
            t_error_query = loss_query_module.TErrorQuery(
                batch_size=2,
                device=torch.device("cpu"),
                noise_level=0.25,
            )

            with (
                patch.object(t_error_query, "load_model", return_value=(DummyModel(), torch.tensor([0, 2, 4]))),
                patch.object(
                    t_error_query,
                    "t_error",
                    side_effect=[
                        torch.tensor([[0.0], [2.0]], dtype=torch.float32),
                        torch.tensor([[4.0]], dtype=torch.float32),
                    ],
                ) as t_error_fn,
            ):
                signal, train_mask = t_error_query.query(dataset, target_path, n_data_points=3)

            self.assertTrue(torch.allclose(signal, torch.tensor([[0.0], [2.0], [4.0]])))
            self.assertTrue(torch.equal(train_mask, torch.tensor([True, False, True])))
            called_batches = [call_args.args[1] for call_args in t_error_fn.call_args_list]
            self.assertTrue(torch.equal(called_batches[0], torch.tensor([0, 1])))
            self.assertTrue(torch.equal(called_batches[1], torch.tensor([2])))

    def test_ddpm_pian_normalization_is_batchwise_and_finite_for_zero_norm(self):
        '''
        Normalize PIAN epsilon per batch item without broadcasting over image axes.
        Returns:
            None
        '''
        class RecordingNetwork:
            def __init__(self):
                self.calls = []
                self.eps_0 = torch.tensor(
                    [
                        [[[1.0, 1.0], [1.0, 1.0]]],
                        [[[0.0, 0.0], [0.0, 0.0]]],
                    ],
                    dtype=torch.float32,
                )
                self.eps_t = torch.zeros_like(self.eps_0)

            def __call__(self, x, t):
                self.calls.append((x.clone(), t.clone()))
                if len(self.calls) == 1:
                    return self.eps_0
                return self.eps_t

        model = DDPMModel.__new__(DDPMModel)
        model.time_steps = 3
        model.alpha_bar = torch.tensor([1.0, 0.25, 0.0], dtype=torch.float32)
        model.network = RecordingNetwork()
        x = torch.zeros(2, 1, 2, 2, dtype=torch.float32)

        score = model.pia_score(x, noise_level=0.5, lp_norm=2, normalize=True)

        expected_scale = torch.sqrt(torch.tensor(torch.pi * 0.5, dtype=torch.float32))
        expected_score = torch.tensor([2.0 * expected_scale, 0.0], dtype=torch.float32)
        self.assertTrue(torch.all(torch.isfinite(score)))
        self.assertTrue(torch.allclose(score, expected_score))
        first_t = model.network.calls[0][1]
        second_t = model.network.calls[1][1]
        self.assertTrue(torch.equal(first_t, torch.tensor([0, 0], dtype=torch.long)))
        self.assertTrue(torch.equal(second_t, torch.tensor([1, 1], dtype=torch.long)))

    def test_run_loss_query_saves_first_n_loss_signals(self):
        '''
        Save compact first-n loss signals with a subset-aware filename.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(4))

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset),
                patch.object(
                    loss_query_module.LossQuery,
                    "query",
                    return_value=(
                        torch.tensor([[0.1, 0.3], [0.2, 0.4]], dtype=torch.float32),
                        torch.tensor([0, 1], dtype=torch.bool),
                    ),
                ) as query_fn,
            ):
                saved_paths = loss_query_module.run_loss_query(
                    checkpoint_paths=[target_path],
                    checkpoint_properties=loss_query_module.utils.parse_properties_from_checkpoint_path(target_path),
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_samples=10,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                    n_data_points=2,
                )

            query_fn.assert_called_once()
            self.assertEqual(query_fn.call_args.args[2], 2)
            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 10, 0.25, 2)
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            self.assertEqual(sorted(payload.keys()), ["loss_sigs", "train_mask"])
            self.assertTrue(torch.allclose(torch.tensor(payload["loss_sigs"]), torch.tensor([[0.1, 0.3], [0.2, 0.4]])))
            self.assertEqual(payload["train_mask"], [False, True])

    def test_run_loss_query_saves_l4_norm_signals_for_checkpoints(self):
        '''
        Route run_loss_query through L4NormQuery and save signal-typed files.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            dataset = list(range(4))

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset),
                patch.object(
                    loss_query_module.L4NormQuery,
                    "query",
                    return_value=(
                        torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32),
                        torch.tensor([0, 1], dtype=torch.bool),
                    ),
                ) as query_fn,
            ):
                saved_paths = loss_query_module.run_loss_query(
                    checkpoint_paths=[target_path],
                    checkpoint_properties=loss_query_module.utils.parse_properties_from_checkpoint_path(target_path),
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_samples=2,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                    n_data_points=2,
                    signal_type="l4_norm",
                )

            query_fn.assert_called_once()
            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 2, 0.25, 2, "l4_norm")
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            expected_signal = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
            self.assertEqual(sorted(payload.keys()), ["loss_sigs", "train_mask"])
            self.assertTrue(torch.allclose(torch.tensor(payload["loss_sigs"]), expected_signal))
            self.assertEqual(payload["train_mask"], [False, True])

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
                    "query",
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
                    n_samples=10,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                )

            load_dataset_fn.assert_called_once_with("CIFAR10", data_dir=tmpdir, size=32)

    def test_run_loss_query_routes_pia_and_pian_signals(self):
        '''
        Route PIA and PIAN signal types with the expected normalization flag.
        Returns:
            None
        '''
        class FakePIAQuery:
            instances = []

            def __init__(self, batch_size, device, noise_level, normalize=True):
                self.batch_size = batch_size
                self.device = device
                self.noise_level = noise_level
                self.normalize = normalize
                self.instances.append(self)

            def query(self, loaded_dataset, model_path, n_data_points=None):
                return (
                    torch.tensor([[0.3], [0.4]], dtype=torch.float32),
                    torch.tensor([True, False], dtype=torch.bool),
                )

        checkpoint_properties = {
            "dataset": "cifar10",
            "size": 32,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            with (
                patch.object(loss_query_module, "load_dataset", return_value=[0, 1]),
                patch.object(loss_query_module, "PIAQuery", new=FakePIAQuery),
            ):
                for signal_type in ("pia_score", "pian_score"):
                    loss_query_module.run_loss_query(
                        checkpoint_paths=[target_path],
                        checkpoint_properties=checkpoint_properties,
                        dataset="cifar10",
                        data_dir=tmpdir,
                        batch_size=2,
                        res_dir=tmpdir,
                        n_samples=1,
                        device=torch.device("cpu"),
                        noise_level=0.25,
                        signal_type=signal_type,
                    )

        self.assertFalse(FakePIAQuery.instances[0].normalize)
        self.assertTrue(FakePIAQuery.instances[1].normalize)

    def test_run_loss_query_rejects_multiple_pia_samples(self):
        '''
        Reject repeated PIA queries because the signal is deterministic.
        Returns:
            None
        '''
        checkpoint_properties = {
            "dataset": "cifar10",
            "size": 32,
        }
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(loss_query_module, "load_dataset", return_value=[0, 1]),
        ):
            with self.assertRaises(AssertionError):
                loss_query_module.run_loss_query(
                    checkpoint_paths=[],
                    checkpoint_properties=checkpoint_properties,
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_samples=2,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                    signal_type="pia_score",
                )

    def test_run_loss_query_routes_t_error_signal(self):
        '''
        Route run_loss_query through TErrorQuery and save signal-typed files.
        Returns:
            None
        '''
        class FakeTErrorQuery:
            instances = []

            def __init__(self, batch_size, device, noise_level):
                self.batch_size = batch_size
                self.device = device
                self.noise_level = noise_level
                self.instances.append(self)

            def query(self, loaded_dataset, model_path, n_data_points=None):
                return (
                    torch.tensor([[0.3], [0.4]], dtype=torch.float32),
                    torch.tensor([True, False], dtype=torch.bool),
                )

        checkpoint_properties = {
            "dataset": "cifar10",
            "size": 32,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            with (
                patch.object(loss_query_module, "load_dataset", return_value=[0, 1]),
                patch.object(loss_query_module, "TErrorQuery", new=FakeTErrorQuery),
            ):
                saved_paths = loss_query_module.run_loss_query(
                    checkpoint_paths=[target_path],
                    checkpoint_properties=checkpoint_properties,
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_samples=1,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                    signal_type="t_error",
                )

            self.assertEqual(len(FakeTErrorQuery.instances), 1)
            self.assertEqual(FakeTErrorQuery.instances[0].noise_level, 0.25)
            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 1, 0.25, None, "t_error")
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            self.assertEqual(payload["loss_sigs"], [[0.30000001192092896], [0.4000000059604645]])
            self.assertEqual(payload["train_mask"], [True, False])

    def test_run_loss_query_rejects_multiple_t_error_samples(self):
        '''
        Reject repeated t-error queries because the signal is deterministic.
        Returns:
            None
        '''
        checkpoint_properties = {
            "dataset": "cifar10",
            "size": 32,
        }
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(loss_query_module, "load_dataset", return_value=[0, 1]),
        ):
            with self.assertRaises(AssertionError):
                loss_query_module.run_loss_query(
                    checkpoint_paths=[],
                    checkpoint_properties=checkpoint_properties,
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_samples=2,
                    device=torch.device("cpu"),
                    noise_level=0.25,
                    signal_type="t_error",
                )

    def test_load_loss_signals_preserves_saved_loss_samples(self):
        '''
        Preserve 2D saved loss samples when loading signals for sample-aware audits.
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

            expected_loss_sigs = torch.tensor([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]])
            self.assertTrue(torch.allclose(loss_sigs, expected_loss_sigs))
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
                n_samples=2,
                device=torch.device("cpu"),
                noise_level=0.1,
            )

        self.assertEqual(saved_paths, [])

    def test_parse_args_defaults_n_samples(self):
        args = loss_query_module.parse_args([
            "--checkpoint-paths",
            "/tmp/DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth",
            "--batch-size",
            "2",
            "--noise-level",
            "0.25",
        ])
        self.assertEqual(args.n_samples, 1)

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
            "--n-samples",
            "3",
            "--noise-level",
            "0.25",
            "--n-data-points",
            "100",
            "--signal-type",
            "l4_norm",
        ])
        self.assertEqual(args.noise_level, 0.25)
        self.assertEqual(args.n_data_points, 100)
        self.assertEqual(args.signal_type, "l4_norm")

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
                    "--n-samples",
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
            self.assertIn("n_samples: 3", printed_lines)
            self.assertIn("noise_level: 0.25", printed_lines)
            self.assertIn("n_data_points: None", printed_lines)
            self.assertIn("signal_type: loss", printed_lines)
            self.assertIn("checkpoint_paths:", printed_lines)
            self.assertIn(f"  - {override_path}", printed_lines)
            kwargs = run_loss_query_fn.call_args.kwargs
            self.assertEqual(kwargs["checkpoint_paths"], [Path(override_path)])
            self.assertEqual(kwargs["dataset"], "CIFAR10")
            self.assertEqual(kwargs["data_dir"], Path(tmpdir))
            self.assertEqual(kwargs["batch_size"], 2)
            self.assertEqual(kwargs["res_dir"], Path(override_res_dir))
            self.assertEqual(kwargs["n_samples"], 3)
            self.assertEqual(kwargs["noise_level"], 0.25)
            self.assertIsNone(kwargs["n_data_points"])
            self.assertEqual(kwargs["signal_type"], "loss")

    def test_main_forwards_n_data_points(self):
        '''
        Forward the first-n data point option from the CLI.
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
                "--n-samples",
                "3",
                "--noise-level",
                "0.25",
                "--n-data-points",
                "100",
            ])

        kwargs = run_loss_query_fn.call_args.kwargs
        printed_lines = [args.args[0] for args in print_fn.call_args_list]
        self.assertIn("n_data_points: 100", printed_lines)
        self.assertEqual(kwargs["n_data_points"], 100)

    def test_main_forwards_l4_norm_signal_type(self):
        '''
        Forward the selected signal type from the CLI.
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
                "--n-samples",
                "3",
                "--noise-level",
                "0.25",
                "--signal-type",
                "l4_norm",
            ])

        kwargs = run_loss_query_fn.call_args.kwargs
        printed_lines = [args.args[0] for args in print_fn.call_args_list]
        self.assertIn("signal_type: l4_norm", printed_lines)
        self.assertEqual(kwargs["signal_type"], "l4_norm")

    def test_main_forwards_t_error_signal_type(self):
        '''
        Forward the selected t-error signal type from the CLI.
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
                "--n-samples",
                "1",
                "--noise-level",
                "0.25",
                "--signal-type",
                "t_error",
            ])

        kwargs = run_loss_query_fn.call_args.kwargs
        printed_lines = [args.args[0] for args in print_fn.call_args_list]
        self.assertIn("signal_type: t_error", printed_lines)
        self.assertEqual(kwargs["signal_type"], "t_error")

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
                "--n-samples",
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
