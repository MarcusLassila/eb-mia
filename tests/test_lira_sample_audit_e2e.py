import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mia import path_utils
from mia import run_audit as run_audit_module
from mia import run_mia as run_mia_module


class TestLiRAOfflineSampleAuditEndToEnd(unittest.TestCase):
    def test_lira_offline_sample_audit_has_no_variance_in_audit_points_over_five_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            dataset = list(range(20))
            dataset_name = "CelebA2"
            split_spec = "ent-f0p5-p0p5"
            target_paths = [
                tmpdir_path / f"DDPM-{dataset_name}-{split_spec}-s{seed}-sz32-epoch4.pth"
                for seed in range(5)
            ]
            shadow_paths = [
                tmpdir_path / f"DDPM-{dataset_name}-{split_spec}-s{seed}-sz32-epoch4.pth"
                for seed in range(10, 14)
            ]

            base_train_indices = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=torch.long)
            train_indices_by_target = {
                target_path: torch.remainder(base_train_indices + shift, len(dataset))
                for shift, target_path in enumerate(target_paths)
            }

            def fake_get_train_indices(path):
                return train_indices_by_target[Path(path)]

            def fake_lira_run_attack(self, audit_samples, target_path):
                target_seed = run_mia_module.utils.parse_properties_from_checkpoint_path(target_path)["seed"]
                offset = 0.01 * target_seed
                return torch.linspace(0.1 + offset, 0.9 + offset, steps=len(audit_samples), dtype=torch.float32)

            score_paths = []
            with (
                patch.object(run_mia_module, "load_dataset", return_value=dataset),
                patch.object(run_mia_module.utils, "get_train_indices", side_effect=fake_get_train_indices),
                patch.object(run_mia_module.attacks.LiRA, "run_attack", new=fake_lira_run_attack),
            ):
                for target_path in target_paths:
                    mia_config = run_mia_module.utils.Config({
                        "dataset": dataset_name,
                        "data_dir": str(tmpdir_path),
                        "batch_size": 4,
                        "round_robin": False,
                        "res_dir": str(tmpdir_path),
                        "target_model_paths": [str(target_path)],
                        "shadow_model_paths": [str(path) for path in shadow_paths],
                        "attack": {"name": "LiRA-off", "attack": "LiRA", "offline": True, "n_loss_samples": 1},
                    })
                    run_mia_module.run_mia(config=mia_config, device=torch.device("cpu"))
                    score_path = (
                        path_utils.scores_dir(tmpdir_path, "LiRA-off", target_path)
                        / path_utils.scores_pickle_name(target_path, "LiRA-off")
                    )
                    self.assertTrue(score_path.exists())
                    score_paths.append(score_path)

            self.assertEqual(len(score_paths), 5)

            audit_config = run_audit_module.utils.Config({
                "dataset": dataset_name,
                "data_dir": str(tmpdir_path),
                "res_dir": str(tmpdir_path),
                "audit_mode": "sample",
                "n_audit_samples": len(dataset),
                "score_paths": [str(path) for path in score_paths],
            })

            with patch.object(run_audit_module, "load_dataset", return_value=dataset):
                run_audit_module.run_sample_audit(config=audit_config)

            metrics_paths = [
                path_utils.metrics_dir(tmpdir_path, score_path, "sample")
                / path_utils.metrics_pickle_name(score_path, "sample")
                for score_path in score_paths
            ]
            self.assertEqual(len(metrics_paths), 5)
            for metrics_path in metrics_paths:
                self.assertTrue(metrics_path.exists())

            n_audit_points = []
            for metrics_path in metrics_paths:
                with open(metrics_path, "rb") as file:
                    metrics = pickle.load(file)
                n_audit_points.append(int(metrics["n_audit_points"]))

            self.assertEqual(float(np.mean(n_audit_points)), float(len(dataset)))
            self.assertEqual(float(np.std(n_audit_points)), 0.0)


if __name__ == "__main__":
    unittest.main()
