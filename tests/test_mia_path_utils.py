import unittest
import tempfile
from pathlib import Path

from mia import path_utils
from mia import run_audit as run_audit_module


class TestMiaPathUtils(unittest.TestCase):
    def test_audit_result_name_matches_task_format(self):
        target_path = "/tmp/DDPM-CelebA2-ent-f0p5-p0p5-s3-sz64-epoch1000.pth"
        result_name = path_utils.audit_result_name(target_path)
        self.assertEqual(result_name, "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000")

    def test_loss_signal_path_helpers_and_metrics_names(self):
        target_path = Path("/tmp/DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000.pth")
        loss_path = path_utils.loss_signals_dir("/results", target_path) / path_utils.loss_signals_pickle_name(target_path, 10)
        self.assertEqual(
            loss_path,
            Path("/results/DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000/loss_signals/loss_signals-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000-ls10.pkl"),
        )
        self.assertEqual(path_utils.target_stem_from_loss_signals_pickle_path(loss_path), target_path.stem)
        self.assertEqual(path_utils.n_loss_samples_from_loss_signals_pickle_path(loss_path), 10)

        result_dir = path_utils.metrics_dir_from_target("/results", target_path, "BASE", "entity", entity_audit_mode="all")
        self.assertEqual(
            result_dir,
            Path("/results/DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000/BASE-entity-all"),
        )
        result_dir_with_n = path_utils.metrics_dir_from_target(
            "/results",
            target_path,
            "BASE",
            "entity",
            entity_audit_mode="max_one_train",
            n_audit_samples_per_entity=10,
        )
        self.assertEqual(
            result_dir_with_n,
            Path("/results/DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000/BASE-entity-max_one_train-n10"),
        )
        filename = path_utils.metrics_pickle_name_from_target(target_path, "BASE", "entity", min_samples_per_entity=1, max_samples_per_entity=2)
        self.assertEqual(
            filename,
            "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-entity_min-1_max-2.pkl",
        )

    def test_resolve_audit_loss_signal_paths_accepts_file_and_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth"
            loss_file = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 5)
            loss_file.parent.mkdir(parents=True, exist_ok=True)
            loss_file.touch()
            config = run_audit_module.utils.Config({"res_dir": tmpdir, "target_loss_paths": [str(loss_file.parent)]})
            resolved = path_utils.resolve_audit_loss_signal_paths(config, "target_loss_paths")
            self.assertEqual(resolved, [loss_file.resolve()])


if __name__ == "__main__":
    unittest.main()
