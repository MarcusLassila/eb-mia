import pickle
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch, call

import torch

from mia import attacks
from mia import path_utils
from mia import run_audit as run_audit_module


class DummyEntityDataset:

    def __init__(self, entity_ids):
        self.entity_ids = torch.tensor(entity_ids, dtype=torch.long)

    @property
    def n_entities(self):
        return int(torch.unique(self.entity_ids).numel())

    def get_entity_index_table(self):
        table = defaultdict(list)
        for idx, entity_id in enumerate(self.entity_ids.tolist()):
            table[entity_id].append(idx)
        return table

    def __getitem__(self, index):
        return int(index)

    def __len__(self):
        return len(self.entity_ids)


class TestRunAuditMetrics(unittest.TestCase):
    def _write_loss_file(self, res_dir, target_path, n_loss_samples, loss_sigs, train_mask):
        loss_path = path_utils.loss_signals_dir(res_dir, target_path) / path_utils.loss_signals_pickle_name(target_path, n_loss_samples)
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_path, "wb") as file:
            pickle.dump({"loss_sigs": loss_sigs, "train_mask": train_mask}, file)
        return loss_path

    def _entity_metadata(self, entity_ids):
        return {
            "n_samples": len(entity_ids),
            "entity_ids": entity_ids,
            "n_entities": len(set(entity_ids)),
        }

    def test_run_sample_audit_computes_scores_from_loss_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s4-sz32-epoch4.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s5-sz32-epoch4.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, 5, [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 0])
            shadow_loss_path_a = self._write_loss_file(tmpdir, shadow_path_a, 5, [1.0, 0.3, 1.2, 0.4], [1, 1, 0, 0])
            shadow_loss_path_b = self._write_loss_file(tmpdir, shadow_path_b, 5, [1.1, 0.2, 1.3, 0.5], [0, 0, 1, 1])
            config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5},
            })

            expected_scores = attacks.BASE(
                shadow_loss_sigs=torch.tensor([[1.0, 0.3, 1.2, 0.4], [1.1, 0.2, 1.3, 0.5]], dtype=torch.float32),
                shadow_train_mask=torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool),
                offline=True,
                prior=0.5,
            ).run_attack(torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32))

            with (
                patch.object(run_audit_module, "select_sample_audit_indices", return_value=torch.tensor([0, 1, 2, 3], dtype=torch.long)),
                patch.object(
                    run_audit_module.evaluation,
                    "evaluate_MIA",
                    return_value={"AUC": 0.5, "TPR@1%FPR": 0.25, "TPR@0.1%FPR": 0.1, "n_audit_points": 4},
                ) as eval_fn,
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
                patch("builtins.print") as print_fn,
            ):
                run_audit_module.run_sample_audit(config=config)

            eval_kwargs = eval_fn.call_args.kwargs
            self.assertTrue(torch.allclose(eval_kwargs["score"], expected_scores, atol=1e-6))
            self.assertTrue(torch.equal(eval_kwargs["ground_truth"], torch.tensor([1, 0, 1, 0], dtype=torch.long)))
            metrics_dir = path_utils.metrics_dir_from_target(tmpdir, target_path, "BASE-off", "sample")
            metrics_path = metrics_dir / path_utils.metrics_pickle_name_from_target(target_path, "BASE-off", "sample")
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "rb") as file:
                metrics = pickle.load(file)
            self.assertIn("audit_config", metrics)
            print_fn.assert_has_calls([
                call(""),
                call("Audit summary (BASE-off)"),
                call(f"{'Metric':<16} {'Mean':>10}"),
                call(f"{'-' * 16} {'-' * 10}"),
                call(f"{'AUC':<16} {0.5:>10.4f}"),
                call(f"{'TPR@1%FPR':<16} {0.25:>10.4f}"),
                call(f"{'TPR@0.1%FPR':<16} {0.1:>10.4f}"),
                call(f"{'n_audit_points':<16} {4.0:>10.4f}"),
            ])

    def test_run_entity_audit_composes_base_scores_and_saves_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p5-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p5-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p5-s2-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, 3, [0.2, 0.5, 0.1, 0.1], [1, 0, 0, 0])
            shadow_loss_path_a = self._write_loss_file(tmpdir, shadow_path_a, 3, [0.9, 0.2, 1.1, 1.2], [1, 0, 0, 0])
            shadow_loss_path_b = self._write_loss_file(tmpdir, shadow_path_b, 3, [1.0, 0.3, 0.8, 0.9], [0, 0, 1, 0])
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "hold_out",
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 1,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "CompositeBASE-off", "attack": "CompositeBASE", "offline": True, "prior": 0.5},
            })
            captured = {}

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 0.6, "TPR@1%FPR": 0.3, "TPR@0.1%FPR": 0.2, "n_audit_points": 2}

            attack_config = run_audit_module.utils.Config({
                "offline": True,
                "prior": 0.5,
            })
            expected_entity_scores = attacks.CompositeBASE(
                attack_config=attack_config,
                shadow_loss_sigs=torch.tensor([[0.9, 0.2, 1.1, 1.2], [1.0, 0.3, 0.8, 0.9]], dtype=torch.float32),
                shadow_train_mask=torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.bool),
            ).run_attack(
                audit_table={
                    0: torch.tensor([1], dtype=torch.long),
                    1: torch.tensor([3], dtype=torch.long),
                },
                target_loss_sigs=torch.tensor([0.2, 0.5, 0.1, 0.1], dtype=torch.float32),
            )

            with (
                patch.object(run_audit_module, "load_dataset_metadata", return_value=self._entity_metadata([0, 0, 1, 1])),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            expected_score_tensor = torch.stack([expected_entity_scores[0], expected_entity_scores[1]]).to(dtype=torch.float32)
            self.assertTrue(torch.allclose(captured["score"], expected_score_tensor, atol=1e-6))
            metrics_path = (
                path_utils.metrics_dir_from_target(
                    tmpdir,
                    target_path,
                    "CompositeBASE-off",
                    "entity",
                    entity_audit_mode="hold_out",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
                / path_utils.metrics_pickle_name_from_target(
                    target_path,
                    "CompositeBASE-off",
                    "entity",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
            )
            self.assertTrue(metrics_path.exists())

    def test_run_entity_audit_uses_joint_xgb_with_updated_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s2-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, 3, [0.2, 0.5, 0.1, 0.1], [1, 0, 0, 0])
            shadow_loss_path_a = self._write_loss_file(tmpdir, shadow_path_a, 3, [0.9, 0.2, 1.1, 1.2], [1, 1, 0, 0])
            shadow_loss_path_b = self._write_loss_file(tmpdir, shadow_path_b, 3, [1.0, 0.3, 0.8, 0.9], [0, 0, 1, 1])
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "exclude_train",
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 1,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "JointXGB", "attack": "JointXGB", "offline": True, "prior": 0.5},
            })
            captured = {}

            class FakeJointXGB:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def run_attack(self, audit_table, target_loss_sigs):
                    return {
                        0: torch.tensor(0.7, dtype=torch.float32),
                        1: torch.tensor(0.2, dtype=torch.float32),
                    }

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 0.8, "TPR@1%FPR": 0.4, "TPR@0.1%FPR": 0.3, "n_audit_points": 2}

            with (
                patch.object(run_audit_module, "load_dataset_metadata", return_value=self._entity_metadata([0, 0, 1, 1])),
                patch.object(run_audit_module.attacks, "JointXGB", FakeJointXGB),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            self.assertTrue(torch.allclose(captured["score"], torch.tensor([0.7, 0.2], dtype=torch.float32), atol=1e-6))
            metrics_path = (
                path_utils.metrics_dir_from_target(
                    tmpdir,
                    target_path,
                    "JointXGB",
                    "entity",
                    entity_audit_mode="exclude_train",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
                / path_utils.metrics_pickle_name_from_target(
                    target_path,
                    "JointXGB",
                    "entity",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
            )
            self.assertTrue(metrics_path.exists())

    def test_run_entity_audit_accepts_hold_out_mode_with_fixed_entity_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p5-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p5-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p5-s2-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, 3, [0.2, 0.5, 0.1, 0.1], [1, 0, 0, 0])
            shadow_loss_path_a = self._write_loss_file(tmpdir, shadow_path_a, 3, [0.9, 0.2, 1.1, 1.2], [1, 0, 0, 0])
            shadow_loss_path_b = self._write_loss_file(tmpdir, shadow_path_b, 3, [1.0, 0.3, 0.8, 0.9], [0, 0, 1, 0])
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "hold_out",
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 1,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "CompositeBASE", "attack": "CompositeBASE", "offline": True, "prior": 0.5},
            })
            with (
                patch.object(run_audit_module, "load_dataset_metadata", return_value=self._entity_metadata([0, 0, 1, 1])),
                patch.object(
                    run_audit_module.evaluation,
                    "evaluate_MIA",
                    return_value={"AUC": 0.6, "TPR@1%FPR": 0.3, "TPR@0.1%FPR": 0.2, "n_audit_points": 2},
                ),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            metrics_path = (
                path_utils.metrics_dir_from_target(
                    tmpdir,
                    target_path,
                    "CompositeBASE",
                    "entity",
                    entity_audit_mode="hold_out",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
                / path_utils.metrics_pickle_name_from_target(
                    target_path,
                    "CompositeBASE",
                    "entity",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
            )
            self.assertTrue(metrics_path.exists())

    def test_run_entity_audit_accepts_globally_unused_hold_out_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s2-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(
                tmpdir,
                target_path,
                3,
                [0.2, 0.3, 0.4, 0.1, 1.1, 1.2, 1.3, 1.4],
                [1, 1, 1, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_a = self._write_loss_file(
                tmpdir,
                shadow_path_a,
                3,
                [0.5, 0.6, 0.7, 0.2, 1.0, 1.1, 1.2, 1.3],
                [1, 1, 0, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_b = self._write_loss_file(
                tmpdir,
                shadow_path_b,
                3,
                [0.8, 0.9, 1.0, 0.3, 0.4, 0.5, 0.6, 1.5],
                [0, 0, 0, 0, 1, 1, 1, 0],
            )
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "hold_out",
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 1,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "CompositeBASE", "attack": "CompositeBASE", "offline": True, "prior": 0.5},
            })
            captured = {}

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 0.7, "TPR@1%FPR": 0.4, "TPR@0.1%FPR": 0.3, "n_audit_points": 2}

            with (
                patch.object(run_audit_module, "load_dataset_metadata", return_value=self._entity_metadata([0, 0, 0, 0, 1, 1, 1, 1])),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            self.assertEqual(captured["score"].shape, (2,))
            metrics_path = (
                path_utils.metrics_dir_from_target(
                    tmpdir,
                    target_path,
                    "CompositeBASE",
                    "entity",
                    entity_audit_mode="hold_out",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
                / path_utils.metrics_pickle_name_from_target(
                    target_path,
                    "CompositeBASE",
                    "entity",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
            )
            self.assertTrue(metrics_path.exists())

    def test_run_entity_audit_composite_lira_supports_hold_out_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s2-sz64-epoch10.pth"
            shadow_path_c = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s3-sz64-epoch10.pth"
            shadow_path_d = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s4-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(
                tmpdir,
                target_path,
                3,
                [0.2, 0.3, 0.4, 0.1, 1.1, 1.2, 1.3, 1.4],
                [1, 1, 1, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_a = self._write_loss_file(
                tmpdir,
                shadow_path_a,
                3,
                [0.5, 0.6, 0.7, 0.2, 1.0, 1.1, 1.2, 1.3],
                [1, 1, 0, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_b = self._write_loss_file(
                tmpdir,
                shadow_path_b,
                3,
                [0.8, 0.9, 1.0, 0.3, 0.4, 0.5, 0.6, 1.5],
                [0, 0, 0, 0, 1, 1, 0, 0],
            )
            shadow_loss_path_c = self._write_loss_file(
                tmpdir,
                shadow_path_c,
                3,
                [0.4, 0.5, 0.6, 0.4, 1.2, 1.3, 1.4, 1.6],
                [1, 0, 1, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_d = self._write_loss_file(
                tmpdir,
                shadow_path_d,
                3,
                [0.9, 1.0, 1.1, 0.5, 0.3, 0.4, 0.5, 1.7],
                [0, 0, 0, 0, 1, 0, 1, 0],
            )
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "hold_out",
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 1,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [
                    str(shadow_loss_path_a),
                    str(shadow_loss_path_b),
                    str(shadow_loss_path_c),
                    str(shadow_loss_path_d),
                ],
                "attack": {"name": "CompositeLiRA", "attack": "CompositeLiRA", "offline": False},
            })
            captured = {}

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 0.7, "TPR@1%FPR": 0.4, "TPR@0.1%FPR": 0.3, "n_audit_points": 2}

            with (
                patch.object(run_audit_module, "load_dataset_metadata", return_value=self._entity_metadata([0, 0, 0, 0, 1, 1, 1, 1])),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            self.assertEqual(captured["score"].shape, (2,))
            self.assertTrue(torch.isfinite(captured["score"]).all())
            metrics_path = (
                path_utils.metrics_dir_from_target(
                    tmpdir,
                    target_path,
                    "CompositeLiRA",
                    "entity",
                    entity_audit_mode="hold_out",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
                / path_utils.metrics_pickle_name_from_target(
                    target_path,
                    "CompositeLiRA",
                    "entity",
                    min_samples_per_entity=1,
                    max_samples_per_entity=1,
                )
            )
            self.assertTrue(metrics_path.exists())

    def test_run_entity_audit_passes_current_kwargs_to_composite_lira(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s2-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(
                tmpdir,
                target_path,
                3,
                [0.2, 0.3, 0.4, 0.1, 1.1, 1.2, 1.3, 1.4],
                [1, 1, 1, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_a = self._write_loss_file(
                tmpdir,
                shadow_path_a,
                3,
                [0.5, 0.6, 0.7, 0.2, 1.0, 1.1, 1.2, 1.3],
                [1, 1, 0, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_b = self._write_loss_file(
                tmpdir,
                shadow_path_b,
                3,
                [0.8, 0.9, 1.0, 0.3, 0.4, 0.5, 0.6, 1.5],
                [0, 0, 0, 0, 1, 1, 0, 0],
            )
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "hold_out",
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 1,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [
                    str(shadow_loss_path_a),
                    str(shadow_loss_path_b),
                ],
                "attack": {"name": "CompositeLiRA", "attack": "CompositeLiRA", "offline": False},
            })
            captured = {}

            class FakeCompositeLiRA:
                def __init__(self, **kwargs):
                    captured["kwargs"] = kwargs

                def run_attack(self, *args, **kwargs):
                    return {
                        0: torch.tensor(0.1, dtype=torch.float32),
                        1: torch.tensor(0.2, dtype=torch.float32),
                    }

            with (
                patch.object(run_audit_module, "load_dataset_metadata", return_value=self._entity_metadata([0, 0, 0, 0, 1, 1, 1, 1])),
                patch.object(run_audit_module.attacks, "CompositeLiRA", FakeCompositeLiRA),
                patch.object(
                    run_audit_module.evaluation,
                    "evaluate_MIA",
                    return_value={"AUC": 0.7, "TPR@1%FPR": 0.4, "TPR@0.1%FPR": 0.3, "n_audit_points": 2},
                ),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertEqual(captured["kwargs"]["audit_table"], {0: [3], 1: [7]})
            self.assertIn("shadow_loss_sigs", captured["kwargs"])
            self.assertIn("shadow_entity_mask", captured["kwargs"])
            self.assertFalse(captured["kwargs"]["offline"])
            self.assertNotIn("hold_out_frac", captured["kwargs"])
            self.assertNotIn("entity_index_table", captured["kwargs"])

    def test_run_entity_audit_rejects_hold_out_indices_used_by_any_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-h0p25-s2-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(
                tmpdir,
                target_path,
                3,
                [0.2, 0.3, 0.4, 0.1, 1.1, 1.2, 1.3, 1.4],
                [1, 1, 1, 0, 0, 0, 0, 0],
            )
            shadow_loss_path_a = self._write_loss_file(
                tmpdir,
                shadow_path_a,
                3,
                [0.5, 0.6, 0.7, 0.2, 1.0, 1.1, 1.2, 1.3],
                [1, 1, 0, 1, 0, 0, 0, 0],
            )
            shadow_loss_path_b = self._write_loss_file(
                tmpdir,
                shadow_path_b,
                3,
                [0.8, 0.9, 1.0, 0.3, 0.4, 0.5, 0.6, 1.5],
                [0, 0, 0, 0, 1, 1, 1, 0],
            )
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "hold_out",
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 1,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "CompositeBASE", "attack": "CompositeBASE", "offline": True, "prior": 0.5},
            })
            with (
                patch.object(run_audit_module, "load_dataset_metadata", return_value=self._entity_metadata([0, 0, 0, 0, 1, 1, 1, 1])),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    r"Entity hold-out indices must be unused by all models\..*Entity 0 hold-out indices \[3\]",
                ):
                    run_audit_module.run_entity_audit(config=config)


if __name__ == "__main__":
    unittest.main()
