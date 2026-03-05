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
    def test_base_scores_feed_sample_audit_and_evaluation_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_path = tmpdir_path / "DDPM-cifar10-rand-f0p5-s0-sz32-epoch4.pth"
            shadow_paths = [
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s1-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s2-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth",
                tmpdir_path / "DDPM-cifar10-rand-f0p5-s4-sz32-epoch4.pth",
            ]
            dataset = list(range(4))

            train_indices_by_path = {
                target_path: torch.tensor([0, 2], dtype=torch.long),
                shadow_paths[0]: torch.tensor([0, 1], dtype=torch.long),
                shadow_paths[1]: torch.tensor([0, 2], dtype=torch.long),
                shadow_paths[2]: torch.tensor([1, 3], dtype=torch.long),
                shadow_paths[3]: torch.tensor([2, 3], dtype=torch.long),
            }
            loss_signal_by_path = {
                target_path: torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32),
                shadow_paths[0]: torch.tensor([0.1, 1.0, 1.1, 1.2], dtype=torch.float32),
                shadow_paths[1]: torch.tensor([1.0, 0.2, 1.1, 1.2], dtype=torch.float32),
                shadow_paths[2]: torch.tensor([1.0, 1.1, 0.3, 1.2], dtype=torch.float32),
                shadow_paths[3]: torch.tensor([1.0, 1.1, 1.2, 0.4], dtype=torch.float32),
            }
            n_in_per_sample = torch.zeros(len(dataset), dtype=torch.long)
            for path in shadow_paths:
                n_in_per_sample[train_indices_by_path[path]] += 1
            expected_n_in = len(shadow_paths) // 2
            self.assertTrue(torch.all(n_in_per_sample == expected_n_in))

            def fake_load_model(self, path):
                path = Path(path)
                return _MockModel(path), train_indices_by_path[path]

            def fake_loss_signal(self, audit_loader, model):
                return loss_signal_by_path[model.path]

            mia_config = run_mia_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": str(tmpdir_path),
                "batch_size": 2,
                "round_robin": False,
                "res_dir": str(tmpdir_path),
                "target_model_paths": [str(target_path)],
                "shadow_model_paths": [str(path) for path in shadow_paths],
                "attack": {"attack": "BASE", "offline": True, "prior": 0.5, "n_loss_samples": 1},
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

            audit_config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": str(tmpdir_path),
                "res_dir": str(tmpdir_path),
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "score_paths": [str(scores_path)],
            })

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.tensor([0, 1, 2, 3], dtype=torch.long)),
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

            eval_config = run_mia_module.utils.Config({
                "res_dir": str(tmpdir_path),
                "metrics_folders": [str(metrics_path.parent)],
            })
            summaries = evaluation_module.run_evaluation(config=eval_config)

            self.assertEqual(len(summaries), 1)
            roc_plot_path = tmpdir_path / f"average_roc_curves_{metrics_path.parent.stem}.png"
            self.assertTrue(roc_plot_path.exists())


if __name__ == "__main__":
    unittest.main()
