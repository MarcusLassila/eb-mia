import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mia import path_utils


class TestMiaPathUtils(unittest.TestCase):
    def test_current_loss_signal_name_round_trips(self):
        target_path = Path("DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth")
        filename = path_utils.loss_signals_pickle_name(
            target_path,
            128,
            noise_level=0.33,
            n_data_points=100,
            signal_type="l4_norm",
        )

        metadata = path_utils.parse_loss_signal_path(filename)

        self.assertEqual(
            filename,
            "l4-norm-signals-DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4-n128-nl0p33-dp100.pkl",
        )
        self.assertEqual(metadata["target_stem"], target_path.stem)
        self.assertEqual(metadata["model"], "DDPM")
        self.assertEqual(metadata["dataset"], "CIFAR10")
        self.assertEqual(metadata["n_loss_samples"], 128)
        self.assertEqual(metadata["noise_level"], 0.33)
        self.assertEqual(metadata["n_data_points"], 100)
        self.assertEqual(metadata["signal_type"], "l4_norm")
        self.assertEqual(metadata["t_error_step_size"], 1)

    def test_t_error_filename_round_trips_ddim_interval(self):
        '''Keep SecMI results from different DDIM intervals distinct.'''
        target_path = Path("DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth")
        filename = path_utils.loss_signals_pickle_name(
            target_path,
            1,
            noise_level=0.1,
            signal_type="t_error",
            t_error_step_size=10,
        )

        metadata = path_utils.parse_loss_signal_path(filename)

        self.assertEqual(filename, "t-error-signals-DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4-n1-nl0p1-t-error-step-size10.pkl")
        self.assertEqual(metadata["signal_type"], "t_error")
        self.assertEqual(metadata["t_error_step_size"], 10)

    def test_legacy_loss_signal_name_remains_readable(self):
        filename = (
            "loss_signals-FlowMatching-VGGFace2-ent-f0p5-p0p5-h0p25-s0-sz32-epoch200"
            "-ls32-nl0p33.pkl"
        )

        metadata = path_utils.parse_loss_signal_path(filename)

        self.assertEqual(metadata["model"], "FlowMatching")
        self.assertEqual(metadata["dataset"], "VGGFace2")
        self.assertEqual(metadata["n_loss_samples"], 32)
        self.assertEqual(metadata["signal_type"], "loss")
        self.assertEqual(metadata["t_error_step_size"], 1)

    def test_resolve_audit_loss_signal_paths_accepts_files_and_directories(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            repo_root = Path(temporary_dir)
            loss_dir = repo_root / "loss-signals"
            loss_dir.mkdir()
            filename = path_utils.loss_signals_pickle_name(
                "DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth",
                5,
            )
            loss_path = loss_dir / filename
            loss_path.touch()
            config = SimpleNamespace(target_loss_paths=["loss-signals"])

            with patch.object(path_utils.utils, "get_root", return_value=str(repo_root)):
                resolved_paths = path_utils.resolve_audit_loss_signal_paths(
                    config,
                    "target_loss_paths",
                )

            self.assertEqual(resolved_paths, [loss_path.resolve()])

    def test_audit_results_use_a_named_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_dir = path_utils.audit_results_dir("sample_core_offline", temporary_dir)

            self.assertEqual(result_dir, (Path(temporary_dir) / "sample_core_offline").resolve())
            metadata = {
                "target_stem": "DDPM-CIFAR10-sz32", "signal_type": "loss", "noise_level": 0.1,
            }
            filename = path_utils.metrics_pickle_name(metadata, "LiRA", "sample", 64, "abc123")
            self.assertEqual(filename, "metrics_attack-LiRA_target-DDPM-CIFAR10-sz32_sample_loss-n64-nl0p1_cfg-abc123.pkl")
            with self.assertRaises(ValueError):
                path_utils.audit_results_dir("nested/name", temporary_dir)

    def test_audit_seed_tracks_target_identity_instead_of_iteration_order(self):
        filename = "loss-signals-DDPM-CelebA-smpl-f0p5-s0-sz64-epoch400-n64-nl0p1.pkl"
        seed = path_utils.audit_seed(Path("first") / filename, 7)
        self.assertEqual(seed, path_utils.audit_seed(Path("other") / filename, 7))
        self.assertEqual(seed, path_utils.audit_seed(filename.replace("nl0p1", "nl0p2"), 7))
        self.assertNotEqual(seed, path_utils.audit_seed(filename, 8))
        self.assertNotEqual(seed, path_utils.audit_seed(filename.replace("-s0-", "-s1-"), 7))


if __name__ == "__main__":
    unittest.main()
