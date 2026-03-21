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
            dataset = list(range(4))

            expected_scores = attacks.BASE(
                shadow_loss_sigs=torch.tensor([[1.0, 0.3, 1.2, 0.4], [1.1, 0.2, 1.3, 0.5]], dtype=torch.float32),
                shadow_train_mask=torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool),
                offline=True,
                prior=0.5,
            ).run_attack(torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32))

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.tensor([0, 1, 2, 3], dtype=torch.long)),
                patch.object(
                    run_audit_module.evaluation,
                    "evaluate_MIA",
                    return_value={"AUC": 0.5, "TPR@1%FPR": 0.25, "TPR@0.1%FPR": 0.1, "n_audit_points": 4},
                ) as eval_fn,
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
                patch("builtins.print") as print_fn,
            ):
                run_audit_module.run_sample_audit(config=config)

            load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
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
                "mode": "all",
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5},
            })
            dataset = DummyEntityDataset([0, 0, 1, 1])
            captured = {}

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 0.6, "TPR@1%FPR": 0.3, "TPR@0.1%FPR": 0.2, "n_audit_points": 2}

            sample_scores = attacks.BASE(
                shadow_loss_sigs=torch.tensor([[0.9, 0.2, 1.1, 1.2], [1.0, 0.3, 0.8, 0.9]], dtype=torch.float32),
                shadow_train_mask=torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool),
                offline=True,
                prior=0.5,
            ).run_attack(torch.tensor([0.2, 0.5, 0.1, 0.1], dtype=torch.float32))
            expected_entity_scores = attacks.composite_BASE({
                0: sample_scores[torch.tensor([0, 1])],
                1: sample_scores[torch.tensor([2, 3])],
            })

            with (
                patch.object(run_audit_module, "EntityDataset", DummyEntityDataset),
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            expected_score_tensor = torch.stack([expected_entity_scores[0], expected_entity_scores[1]]).to(dtype=torch.float32)
            self.assertTrue(torch.allclose(captured["score"], expected_score_tensor, atol=1e-6))
            metrics_path = (
                path_utils.metrics_dir_from_target(tmpdir, target_path, "BASE-off", "entity", entity_audit_mode="all")
                / path_utils.metrics_pickle_name_from_target(target_path, "BASE-off", "entity")
            )
            self.assertTrue(metrics_path.exists())

    def test_run_entity_audit_rejects_all_mode_with_n_audit_samples_per_entity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s0-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, 3, [0.2, 0.5, 0.1, 0.1], [1, 0, 0, 0])
            shadow_loss_path = self._write_loss_file(tmpdir, Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s1-sz64-epoch10.pth", 3, [0.9, 0.2, 1.1, 1.2], [1, 1, 0, 0])
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "all",
                "n_audit_samples_per_entity": 2,
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path)],
                "attack": {"name": "BASE", "attack": "BASE", "offline": True, "prior": 0.5},
            })
            with self.assertRaisesRegex(ValueError, "mode='all'.*n_audit_samples_per_entity"):
                run_audit_module.run_entity_audit(config=config)


if __name__ == "__main__":
    unittest.main()
