import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import evaluation as evaluation_module
from mia import path_utils
from mia import result_store
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
            target_path = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth"
            shadow_path_a = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s1-sz32-epoch4.pth"
            shadow_path_b = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s2-sz32-epoch4.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 0])
            shadow_loss_path_a = self._write_loss_file(tmpdir, shadow_path_a, [1.0, 0.3, 1.2, 0.4], [1, 1, 0, 0])
            shadow_loss_path_b = self._write_loss_file(tmpdir, shadow_path_b, [1.1, 0.2, 1.3, 0.5], [0, 0, 1, 1])
            config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "results_root": tmpdir,
                "results_dir_name": "test_base_sample",
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "round_robin": False,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5},
            })

            with (
                patch.object(run_audit_module, "select_sample_audit_indices", return_value=torch.arange(4, dtype=torch.long)),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_sample_audit(config=config)

            manifest_path = Path(tmpdir) / "test_base_sample" / "audit_manifest.json"
            metrics_path, = result_store.load_manifest_paths(manifest_path)
            self.assertTrue(metrics_path.exists())
            grouped_metrics = evaluation_module.collect_grouped_metrics(metrics_path.parent)
            self.assertEqual(len(grouped_metrics), 1)

    def test_base_sample_audit_round_robin_uses_only_target_paths_and_excludes_complements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_paths = [
                tmpdir_path / "DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-smpl-f0p5-s1-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-smpl-f0p5-s0-comp-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-smpl-f0p5-s1-comp-sz32-epoch4.pth",
            ]
            target_loss_paths = [
                self._write_loss_file(tmpdir, target_paths[0], [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 0]),
                self._write_loss_file(tmpdir, target_paths[1], [1.0, 0.3, 1.2, 0.4], [1, 1, 0, 0]),
                self._write_loss_file(tmpdir, target_paths[2], [0.9, 0.9, 0.9, 0.9], [0, 1, 0, 1]),
                self._write_loss_file(tmpdir, target_paths[3], [1.1, 0.2, 1.3, 0.5], [0, 0, 1, 1]),
            ]
            config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "results_root": tmpdir,
                "results_dir_name": "test_round_robin",
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "round_robin": True,
                "target_loss_paths": [str(path) for path in target_loss_paths],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5},
            })
            captured_scores = []

            def fake_evaluate(score, ground_truth):
                captured_scores.append(score.clone())
                return {"AUC": 0.5, "pAUC@1%FPR": 0.45, "TPR@1%FPR": 0.25, "TPR@0.1%FPR": 0.1, "TPR@0.01%FPR": 0.01, "n_audit_points": len(score)}

            with (
                patch.object(run_audit_module, "select_sample_audit_indices", return_value=torch.arange(4, dtype=torch.long)),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_sample_audit(config=config)

            self.assertEqual(len(captured_scores), 4)
            self.assertTrue(all(score.shape == (4,) for score in captured_scores))

    def test_explicit_shadow_paths_exclude_target_and_complement_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_path = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth"
            same_shadow_path = tmpdir_path / "FlowMatching-cifar10-smpl-f0p5-s0-sz32-epoch4.pth"
            comp_shadow_path = tmpdir_path / "FlowMatching-cifar10-smpl-f0p5-s0-comp-sz32-epoch4.pth"
            kept_shadow_path = tmpdir_path / "FlowMatching-cifar10-smpl-f0p5-s1-sz32-epoch4.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 0])
            same_shadow_loss_path = self._write_loss_file(tmpdir, same_shadow_path, [1.0, 1.1, 1.2, 1.3], [1, 0, 1, 0])
            comp_shadow_loss_path = self._write_loss_file(tmpdir, comp_shadow_path, [1.4, 1.5, 1.6, 1.7], [0, 1, 0, 1])
            kept_shadow_loss_path = self._write_loss_file(tmpdir, kept_shadow_path, [1.8, 1.9, 2.0, 2.1], [1, 1, 0, 0])
            config = run_audit_module.utils.Config({
                "audit_mode": "sample",
                "round_robin": False,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [
                    str(same_shadow_loss_path),
                    str(comp_shadow_loss_path),
                    str(kept_shadow_loss_path),
                ],
            })

            _, shadow_path_groups = run_audit_module.resolve_target_shadow_paths(config)

            self.assertEqual(shadow_path_groups, [[kept_shadow_loss_path.resolve()]])

    def test_entity_shadow_filter_uses_exact_entity_membership_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            masks = [
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
            ]
            paths = []
            for seed, mask in enumerate(masks):
                checkpoint = Path(directory) / f"DDPM-CelebA-ent-f0p5-p0p5-s{seed}-sz64-epoch10.pth"
                path = self._write_loss_file(directory, checkpoint, [1.] * 8, mask)
                paths.append(path.resolve())
            metadata = {"entity_ids": [0, 0, 1, 1, 2, 2, 3, 3]}
            for round_robin in (False, True):
                config = run_audit_module.utils.Config({
                    "audit_mode": "entity", "dataset": "CelebA", "round_robin": round_robin,
                    "target_loss_paths": paths if round_robin else paths[:1],
                    "shadow_loss_paths": paths,
                })
                with patch.object(run_audit_module, "load_dataset_metadata", return_value=metadata):
                    targets, groups = run_audit_module.resolve_target_shadow_paths(config)
                target_index = targets.index(paths[0])
                self.assertEqual(groups[target_index], paths[4:])
            config.audit_mode = "sample"
            config.round_robin = False
            config.target_loss_paths = paths[:1]
            _, groups = run_audit_module.resolve_target_shadow_paths(config)
            self.assertEqual(groups[0], paths[1:])


if __name__ == "__main__":
    unittest.main()
