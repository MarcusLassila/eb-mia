import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from mia import path_utils
from mia import run_audit as run_audit_module


class TestMiaPathUtils(unittest.TestCase):
    def test_audit_result_name_matches_task_format(self):
        target_path = "/tmp/DDPM-CelebA-ent-f0p5-p0p5-s3-sz64-epoch1000.pth"
        result_name = path_utils.audit_result_name(target_path)
        self.assertEqual(result_name, "DDPM-CelebA-ent-f0p5-p0p5-sz64-epoch1000")

    def test_audit_result_name_preserves_hold_out_token(self):
        target_path = "/tmp/DDPM-CelebA-ent-f0p5-p0p5-h0p25-s3-sz64-epoch1000.pth"
        result_name = path_utils.audit_result_name(target_path)
        self.assertEqual(result_name, "DDPM-CelebA-ent-f0p5-p0p5-h0p25-sz64-epoch1000")

    def test_loss_signal_path_helpers_and_metrics_names(self):
        target_path = Path("/tmp/DDPM-CelebA-ent-f0p5-p0p5-s0-sz64-epoch1000.pth")
        loss_path = path_utils.loss_signals_dir("/results", target_path) / path_utils.loss_signals_pickle_name(target_path, 10)
        self.assertEqual(
            loss_path,
            Path("/results/DDPM-CelebA-ent-f0p5-p0p5-sz64-epoch1000/loss-signals-DDPM-CelebA-ent-f0p5-p0p5-s0-sz64-epoch1000-n10-nl0p1.pkl"),
        )
        self.assertEqual(path_utils.target_stem_from_loss_signals_pickle_path(loss_path), target_path.stem)
        self.assertEqual(path_utils.n_loss_samples_from_loss_signals_pickle_path(loss_path), 10)
        self.assertEqual(path_utils.noise_level_from_loss_signals_pickle_path(loss_path), 0.1)

        result_dir = path_utils.metrics_dir_from_target("/results", target_path, "BASE", "entity", entity_audit_mode="all")
        self.assertEqual(
            result_dir,
            Path("/results/DDPM-CelebA-ent-f0p5-p0p5-sz64-epoch1000/BASE-entity-all"),
        )
        result_dir_with_n = path_utils.metrics_dir_from_target(
            "/results",
            target_path,
            "BASE",
            "entity",
            entity_audit_mode="max_one_train",
            min_samples_per_entity=10,
            max_samples_per_entity=10,
        )
        self.assertEqual(
            result_dir_with_n,
            Path("/results/DDPM-CelebA-ent-f0p5-p0p5-sz64-epoch1000/BASE-entity-max_one_train-n10"),
        )
        filename = path_utils.metrics_pickle_name_from_target(target_path, "BASE", "entity", min_samples_per_entity=1, max_samples_per_entity=2)
        self.assertEqual(
            filename,
            "metrics_attack-BASE_target-DDPM-CelebA-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-entity_min-1_max-2.pkl",
        )

    def test_loss_signal_pickle_name_accepts_data_point_count(self):
        '''
        Add an optional data-point count token to loss-signal filenames.
        Returns:
            None
        '''
        target_path = Path("/tmp/DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth")
        loss_path = path_utils.loss_signals_pickle_name(target_path, 128, noise_level=0.33, n_data_points=100)

        self.assertEqual(
            loss_path,
            "loss-signals-DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4-n128-nl0p33-dp100.pkl",
        )
        self.assertEqual(path_utils.target_stem_from_loss_signals_pickle_path(loss_path), target_path.stem)
        self.assertEqual(path_utils.n_loss_samples_from_loss_signals_pickle_path(loss_path), 128)
        self.assertEqual(path_utils.noise_level_from_loss_signals_pickle_path(loss_path), 0.33)

    def test_loss_signal_pickle_name_accepts_signal_type(self):
        '''
        Prefix signal filenames with the filename-safe signal type.
        Returns:
            None
        '''
        target_path = Path("/tmp/DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth")
        loss_path = path_utils.loss_signals_pickle_name(target_path, 30, noise_level=0.03, signal_type="l4_norm")

        self.assertEqual(
            loss_path,
            "l4-norm-signals-DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4-n30-nl0p03.pkl",
        )
        self.assertEqual(path_utils.target_stem_from_loss_signals_pickle_path(loss_path), target_path.stem)
        self.assertEqual(path_utils.n_loss_samples_from_loss_signals_pickle_path(loss_path), 30)
        self.assertEqual(path_utils.noise_level_from_loss_signals_pickle_path(loss_path), 0.03)

    def test_old_signal_typed_loss_signal_pickle_names_still_parse(self):
        '''
        Parse existing signal-typed loss-signal files created before prefix naming.
        Returns:
            None
        '''
        target_path = Path("/tmp/DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth")
        loss_path = "loss_signals-DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4-signal-l4_norm-ls30-nl0p03.pkl"

        self.assertEqual(path_utils.target_stem_from_loss_signals_pickle_path(loss_path), target_path.stem)
        self.assertEqual(path_utils.n_loss_samples_from_loss_signals_pickle_path(loss_path), 30)
        self.assertEqual(path_utils.noise_level_from_loss_signals_pickle_path(loss_path), 0.03)

    def test_old_loss_signal_pickle_names_parse_as_loss(self):
        '''
        Preserve parsing for existing loss-signal files without signal-type tokens.
        Returns:
            None
        '''
        target_path = Path("/tmp/DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth")
        loss_path = "loss_signals-DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4-ls128-nl0p33-dp100.pkl"

        self.assertEqual(path_utils.target_stem_from_loss_signals_pickle_path(loss_path), target_path.stem)
        self.assertEqual(path_utils.n_loss_samples_from_loss_signals_pickle_path(loss_path), 128)
        self.assertEqual(path_utils.noise_level_from_loss_signals_pickle_path(loss_path), 0.33)

    def test_resolve_audit_loss_signal_paths_accepts_file_and_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth"
            loss_file = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 5)
            loss_file.parent.mkdir(parents=True, exist_ok=True)
            loss_file.touch()
            config = run_audit_module.utils.Config({"res_dir": tmpdir, "target_loss_paths": [str(loss_file.parent)]})
            resolved = path_utils.resolve_audit_loss_signal_paths(config, "target_loss_paths")
            self.assertEqual(resolved, [loss_file.resolve()])

    def test_resolve_audit_loss_signal_paths_accepts_l4_norm_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth"
            loss_file = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 5, signal_type="l4_norm")
            loss_file.parent.mkdir(parents=True, exist_ok=True)
            loss_file.touch()
            config = run_audit_module.utils.Config({"res_dir": tmpdir, "target_loss_paths": [str(loss_file.parent)]})
            resolved = path_utils.resolve_audit_loss_signal_paths(config, "target_loss_paths")
            self.assertEqual(resolved, [loss_file.resolve()])

    def test_resolve_audit_loss_signal_paths_uses_repo_root_for_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            target_path = repo_root / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth"
            loss_dir = repo_root / "loss-signals"
            loss_file = loss_dir / path_utils.loss_signals_pickle_name(target_path, 5)
            loss_file.parent.mkdir(parents=True, exist_ok=True)
            loss_file.touch()
            config = run_audit_module.utils.Config({
                "res_dir": str(repo_root / "mia" / "results"),
                "target_loss_paths": ["loss-signals"],
            })

            with patch.object(path_utils.utils, "get_root", return_value=str(repo_root)):
                resolved = path_utils.resolve_audit_loss_signal_paths(config, "target_loss_paths")

            self.assertEqual(resolved, [loss_file.resolve()])


if __name__ == "__main__":
    unittest.main()
