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
from mia import run_mia as run_mia_module


class _MockModel:

    def __init__(self, path):
        self.path = Path(path)


class TestBaseSampleAuditEndToEnd(unittest.TestCase):
    def _run_base_scores_feed_sample_audit_and_evaluation_outputs(
        self,
        dataset_name,
        target_split_spec,
        shadow_split_and_indices,
        expect_exact_half_inclusion,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_path = tmpdir_path / f"DDPM-{dataset_name}-{target_split_spec}-s0-sz32-epoch4.pth"
            shadow_paths = [
                tmpdir_path / f"DDPM-{dataset_name}-{shadow_split_spec}-sz32-epoch4.pth"
                for shadow_split_spec, _ in shadow_split_and_indices
            ]
            dataset = list(range(10))

            train_indices_by_path = {
                target_path: torch.tensor([0, 2, 4, 6, 8], dtype=torch.long),
            }
            for (_, train_indices), shadow_path in zip(shadow_split_and_indices, shadow_paths):
                train_indices_by_path[shadow_path] = torch.tensor(train_indices, dtype=torch.long)
            target_loss_signal = torch.linspace(0.2, 1.1, steps=len(dataset), dtype=torch.float32)
            shadow_loss_signal = torch.linspace(1.0, 1.9, steps=len(dataset), dtype=torch.float32)
            loss_signal_by_path = {
                target_path: target_loss_signal,
            }
            for shadow_index, shadow_path in enumerate(shadow_paths):
                signal = shadow_loss_signal.clone()
                signal[shadow_index] = 0.1 + 0.1 * shadow_index
                loss_signal_by_path[shadow_path] = signal
            n_in_per_sample = torch.zeros(len(dataset), dtype=torch.long)
            for path in shadow_paths:
                n_in_per_sample[train_indices_by_path[path]] += 1
            expected_n_in = len(shadow_paths) // 2
            if expect_exact_half_inclusion:
                self.assertTrue(torch.all(n_in_per_sample == expected_n_in))
            else:
                self.assertTrue(torch.all(n_in_per_sample != expected_n_in))

            def fake_load_model(self, path):
                path = Path(path)
                return _MockModel(path), train_indices_by_path[path]

            def fake_loss_signal(self, audit_loader, model):
                return loss_signal_by_path[model.path]

            mia_config = run_mia_module.utils.Config({
                "dataset": dataset_name,
                "data_dir": str(tmpdir_path),
                "batch_size": 2,
                "round_robin": False,
                "res_dir": str(tmpdir_path),
                "target_model_paths": [str(target_path)],
                "shadow_model_paths": [str(path) for path in shadow_paths],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5, "n_loss_samples": 1},
            })

            with (
                patch.object(run_mia_module, "load_dataset", return_value=dataset),
                patch.object(run_mia_module.utils, "get_train_indices", return_value=train_indices_by_path[target_path]),
                patch.object(attacks.BASE, "load_model", new=fake_load_model),
                patch.object(attacks.BASE, "loss_signal", new=fake_loss_signal),
            ):
                run_mia_module.run_mia(config=mia_config, device=torch.device("cpu"))

            scores_path = (
                path_utils.scores_dir(tmpdir_path, "BASE-off", target_path)
                / path_utils.scores_pickle_name(target_path, "BASE-off")
            )
            self.assertTrue(scores_path.exists())
            with open(scores_path, "rb") as file:
                scores_payload = pickle.load(file)
            self.assertEqual(len(scores_payload["scores"]), len(dataset))
            self.assertEqual(len(scores_payload["train_mask"]), len(dataset))
            self.assertEqual(scores_path.parent.name, "BASE-off-scores")
            self.assertIn("scores_attack-BASE-off_target-", scores_path.name)

            target_signal = loss_signal_by_path[target_path]
            shadow_train_sets = {
                shadow_path: set(train_indices_by_path[shadow_path].tolist())
                for shadow_path in shadow_paths
            }
            expected_scores = []
            for sample_index in range(len(dataset)):
                out_losses = [
                    loss_signal_by_path[shadow_path][sample_index]
                    for shadow_path in shadow_paths
                    if sample_index not in shadow_train_sets[shadow_path]
                ]
                ref = torch.logsumexp(-torch.stack(out_losses), dim=0) - torch.log(torch.tensor(float(len(out_losses))))
                expected_scores.append((-target_signal[sample_index] - ref).sigmoid())
            self.assertTrue(
                torch.allclose(
                    torch.tensor(scores_payload["scores"], dtype=torch.float32),
                    torch.stack(expected_scores),
                    atol=1e-6,
                )
            )

            audit_config = run_audit_module.utils.Config({
                "dataset": dataset_name,
                "data_dir": str(tmpdir_path),
                "res_dir": str(tmpdir_path),
                "audit_mode": "sample",
                "n_audit_samples": len(dataset),
                "score_paths": [str(scores_path)],
            })

            all_audit_indices = torch.arange(len(dataset), dtype=torch.long)
            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "get_audit_indices", return_value=all_audit_indices),
            ):
                run_audit_module.run_sample_audit(config=audit_config)

            metrics_path = (
                path_utils.metrics_dir(tmpdir_path, scores_path, "sample")
                / path_utils.metrics_pickle_name(scores_path, "sample")
            )
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "rb") as file:
                metrics = pickle.load(file)
            self.assertIn("AUC", metrics)
            self.assertIn("TPR@1%FPR", metrics)
            self.assertIn("TPR@0.1%FPR", metrics)
            self.assertEqual(metrics["n_audit_points"], len(dataset))

            eval_config = run_mia_module.utils.Config({
                "res_dir": str(tmpdir_path),
                "metrics_folders": [str(metrics_path.parent)],
            })
            summaries = evaluation_module.run_evaluation(config=eval_config)

            self.assertEqual(len(summaries), 1)
            roc_plot_path = tmpdir_path / f"average_roc_curves_{metrics_path.parent.stem}.png"
            self.assertTrue(roc_plot_path.exists())

            return scores_path, metrics_path

    def test_base_scores_feed_sample_audit_and_evaluation_outputs_with_complement_pairs(self):
        self._run_base_scores_feed_sample_audit_and_evaluation_outputs(
            dataset_name="cifar10",
            target_split_spec="rand-f0p5",
            shadow_split_and_indices=[
                ("rand-f0p5-s1", [0, 2, 4, 6, 8]),
                ("rand-f0p5-s1-comp", [1, 3, 5, 7, 9]),
                ("rand-f0p5-s2", [0, 1, 2, 3, 4]),
                ("rand-f0p5-s2-comp", [5, 6, 7, 8, 9]),
            ],
            expect_exact_half_inclusion=True,
        )

    def test_base_scores_feed_sample_audit_and_evaluation_outputs_for_entity_split_without_half_inclusion(self):
        scores_path, metrics_path = self._run_base_scores_feed_sample_audit_and_evaluation_outputs(
            dataset_name="CelebA2",
            target_split_spec="ent-f0p5-p0p5",
            shadow_split_and_indices=[
                ("ent-f0p5-p0p5-s1", [0, 1, 2]),
                ("ent-f0p5-p0p5-s2", [0, 3, 4]),
                ("ent-f0p5-p0p5-s3", [0, 5, 6]),
                ("ent-f0p5-p0p5-s4", [7, 8, 9]),
            ],
            expect_exact_half_inclusion=False,
        )
        self.assertIn("ent-f0p5-p0p5", str(scores_path))
        self.assertIn("ent-f0p5-p0p5", str(metrics_path))


if __name__ == "__main__":
    unittest.main()
