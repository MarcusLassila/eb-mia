import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, call

import torch

from mia import run_audit as run_audit_module


class DummyAttacker:

    def run_attack(self, audit_samples, target_path):
        return torch.tensor([0.1, 0.9], dtype=torch.float32)


class TestRunAuditMetrics(unittest.TestCase):
    def test_metrics_pickle_name_entity_includes_min_max(self):
        filename = run_audit_module.metrics_pickle_name(
            "/tmp/VAE-celeba-ent-f0p5-p1-s0-sz64-epoch10.pth",
            "CompositeBASE",
            "entity",
            min_samples_per_entity=1,
            max_samples_per_entity=3,
        )
        self.assertEqual(
            filename,
            "metrics_attack-CompositeBASE_target-VAE-celeba-ent-f0p5-p1-s0-sz64-epoch10_mode-entity_min-1_max-3.pkl",
        )

    def test_run_audit_infers_image_size_and_saves_audit_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth")
            shadow_path = str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s4-sz32-epoch4.pth")
            audit_config = {
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "n_audit_samples": 2,
                "audit_mode": "sample",
                "round_robin": False,
                "res_dir": tmpdir,
                "target_model_paths": [target_path],
                "shadow_model_paths": [shadow_path],
                "attack": {"attack": "BASE", "prior": 0.5, "n_loss_samples": 1},
            }
            config = run_audit_module.utils.Config(dict(audit_config))
            dataset = list(range(8))

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(run_audit_module.utils, "get_train_indices", return_value=torch.tensor([0, 1, 2, 3])),
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.tensor([0, 4])),
                patch.object(run_audit_module, "get_attacker", return_value=DummyAttacker()),
                patch.object(
                    run_audit_module.evaluation,
                    "evaluate_MIA",
                    return_value={"AUC": 0.5, "TPR@1%FPR": 0.25, "TPR@0.1%FPR": 0.1},
                ),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
                patch("builtins.print") as print_fn,
            ):
                run_audit_module.run_audit(config=config, device=torch.device("cpu"), audit_config=audit_config)

            load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
            metrics_dir = Path(tmpdir) / "BASE"
            metrics_files = sorted(metrics_dir.glob("*.pkl"))
            self.assertEqual(len(metrics_files), 1)
            self.assertEqual(
                metrics_files[0].name,
                "metrics_attack-BASE_target-DDPM-cifar10-rand-f0p5-s3-sz32-epoch4_mode-sample.pkl",
            )
            with open(metrics_files[0], "rb") as file:
                metrics = pickle.load(file)
            self.assertEqual(metrics["AUC"], 0.5)
            self.assertEqual(metrics["TPR@1%FPR"], 0.25)
            self.assertEqual(metrics["TPR@0.1%FPR"], 0.1)
            self.assertEqual(metrics["audit_config"], audit_config)
            print_fn.assert_has_calls([
                call(""),
                call("Audit summary (BASE)"),
                call(f"{'Metric':<16} {'Mean':>10}"),
                call(f"{'-' * 16} {'-' * 10}"),
                call(f"{'AUC':<16} {0.5:>10.4f}"),
                call(f"{'TPR@1%FPR':<16} {0.25:>10.4f}"),
                call(f"{'TPR@0.1%FPR':<16} {0.1:>10.4f}"),
            ])


if __name__ == "__main__":
    unittest.main()
