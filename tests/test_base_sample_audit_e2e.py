import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import attacks
from mia import evaluation as evaluation_module
from mia import path_utils
from mia import run_audit as run_audit_module


class TestBaseSampleAuditEndToEnd(unittest.TestCase):
    def _write_loss_file(self, res_dir, target_path, loss_sigs, train_mask):
        loss_path = path_utils.loss_signals_dir(res_dir, target_path) / path_utils.loss_signals_pickle_name(target_path, 1)
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_path, "wb") as file:
            pickle.dump({"loss_sigs": loss_sigs, "train_mask": train_mask}, file)
        return loss_path

    def test_base_sample_audit_and_evaluation_with_explicit_shadows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_path = tmpdir_path / "DDPM-cifar10-rand-f0p5-s0-sz32-epoch4.pth"
            shadow_path_a = tmpdir_path / "DDPM-cifar10-rand-f0p5-s1-sz32-epoch4.pth"
            shadow_path_b = tmpdir_path / "DDPM-cifar10-rand-f0p5-s2-sz32-epoch4.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 0])
            shadow_loss_path_a = self._write_loss_file(tmpdir, shadow_path_a, [1.0, 0.3, 1.2, 0.4], [1, 1, 0, 0])
            shadow_loss_path_b = self._write_loss_file(tmpdir, shadow_path_b, [1.1, 0.2, 1.3, 0.5], [0, 0, 1, 1])
            config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "res_dir": tmpdir,
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "round_robin": False,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5},
            })
            dataset = list(range(4))

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.arange(4, dtype=torch.long)),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_sample_audit(config=config)

            metrics_dir = path_utils.metrics_dir_from_target(tmpdir, target_path, "BASE-off", "sample")
            metrics_path = metrics_dir / path_utils.metrics_pickle_name_from_target(target_path, "BASE-off", "sample")
            self.assertTrue(metrics_path.exists())
            summaries = evaluation_module.run_evaluation(
                config=run_audit_module.utils.Config({"res_dir": tmpdir, "metrics_folders": [str(metrics_dir)]}),
            )
            self.assertEqual(len(summaries), 1)
            self.assertTrue((tmpdir_path / f"average_roc_curves_{metrics_dir.stem}.png").exists())

    def test_base_sample_audit_round_robin_uses_only_target_paths_and_excludes_complements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_paths = [
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s0-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s0-comp-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s1-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s1-comp-sz32-epoch4.pth",
            ]
            target_loss_paths = [
                self._write_loss_file(tmpdir, target_paths[0], [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 0]),
                self._write_loss_file(tmpdir, target_paths[1], [0.9, 0.9, 0.9, 0.9], [0, 1, 0, 1]),
                self._write_loss_file(tmpdir, target_paths[2], [1.0, 0.3, 1.2, 0.4], [1, 1, 0, 0]),
                self._write_loss_file(tmpdir, target_paths[3], [1.1, 0.2, 1.3, 0.5], [0, 0, 1, 1]),
            ]
            config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "res_dir": tmpdir,
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "round_robin": True,
                "target_loss_paths": [str(path) for path in target_loss_paths],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5},
            })
            dataset = list(range(4))
            captured_scores = []

            def fake_evaluate(score, ground_truth):
                captured_scores.append(score.clone())
                return {"AUC": 0.5, "TPR@1%FPR": 0.25, "TPR@0.1%FPR": 0.1, "n_audit_points": len(score)}

            expected_first_target_scores = attacks.BASE(
                shadow_loss_sigs=torch.tensor([[1.0, 0.3, 1.2, 0.4], [1.1, 0.2, 1.3, 0.5]], dtype=torch.float32),
                shadow_train_mask=torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool),
                offline=True,
                prior=0.5,
            ).run_attack(torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32))

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.arange(4, dtype=torch.long)),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_sample_audit(config=config)

            self.assertEqual(len(captured_scores), 4)
            self.assertTrue(torch.allclose(captured_scores[0], expected_first_target_scores, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
