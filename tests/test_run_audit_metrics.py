import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import run_audit


class TestRunAuditMetrics(unittest.TestCase):
    def test_save_audit_metrics_uses_flat_benchmark_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            loss_path = Path(temporary_dir) / (
                "loss-signals-DDPM-CelebA-smpl-f0p5-s0-sz64-epoch400-n64-nl0p1.pkl"
            )
            config = run_audit.utils.Config({
                "results_root": temporary_dir,
                "results_dir_name": "sample_core_offline",
                "audit_mode": "sample",
            })
            metrics = {"AUC": 0.7}

            metrics_path = run_audit.save_audit_metrics(
                config,
                metrics,
                loss_path,
                "LiRA",
            )

            expected_dir = (Path(temporary_dir) / "sample_core_offline").resolve()
            self.assertEqual(metrics_path.parent, expected_dir)
            self.assertEqual(
                metrics_path.name,
                "metrics_attack-LiRA_target-DDPM-CelebA-smpl-f0p5-s0-sz64-epoch400.pkl",
            )
            with open(metrics_path, "rb") as file:
                saved_metrics = pickle.load(file)
            self.assertEqual(saved_metrics["target_model"], "DDPM")
            self.assertEqual(saved_metrics["target_dataset"], "CelebA")
            self.assertEqual(saved_metrics["attack"], "LiRA")

    def test_hg_lira_r_dispatch_forwards_offline_sampling_settings(self):
        captured = {}

        class FakeHGLiRAR:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        attack_config = run_audit.utils.Config({
            "attack": "HG_LiRA_r",
            "offline": True,
            "loss_transformation": "none",
            "n_gibbs_samples": 64,
            "n_gibbs_warmup": 32,
            "n_quadrature": 20,
        })
        shadow_loss_sigs = torch.ones((4, 3, 2))
        shadow_train_mask = torch.ones((4, 3), dtype=torch.bool)

        with patch.object(run_audit.attacks_sample, "HG_LiRA_r", FakeHGLiRAR):
            attacker = run_audit.get_attacker(
                attack_config,
                shadow_loss_sigs,
                shadow_train_mask,
            )

        self.assertIsInstance(attacker, FakeHGLiRAR)
        self.assertTrue(captured["offline"])
        self.assertEqual(captured["n_gibbs_samples"], 64)
        self.assertEqual(captured["n_gibbs_warmup"], 32)
        self.assertEqual(captured["n_quadrature"], 20)

    def test_hbe_entity_dispatch_forwards_model_settings(self):
        '''Verify both retained HBE attacks receive their public configuration.'''
        audit_table = {0: torch.tensor([0, 1]), 1: torch.tensor([2, 3])}
        shadow_loss_sigs = torch.ones((4, 4, 3))
        shadow_entity_mask = torch.zeros((4, 2), dtype=torch.bool)
        cases = (
            (
                "HBE_Simple",
                {"n_gibbs_samples": 64, "n_gibbs_warmup": 32},
                {"n_gibbs_samples": 64, "n_gibbs_warmup": 32},
            ),
            (
                "HBE_GlobalLatent",
                {"n_folds": 2, "quadrature_nodes": 16, "factor_steps": 20},
                {"n_folds": 2, "quadrature_nodes": 16, "factor_steps": 20},
            ),
        )
        for attack_name, settings, expected in cases:
            captured = {}

            class FakeAttack:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

            config = {"attack": attack_name, "offline": True}
            config.update(settings)
            attack_config = run_audit.utils.Config(config)
            with self.subTest(attack=attack_name):
                with patch.object(run_audit.attacks_entity, attack_name, FakeAttack):
                    attacker = run_audit.get_attacker(
                        attack_config,
                        shadow_loss_sigs,
                        audit_table=audit_table,
                        shadow_entity_mask=shadow_entity_mask,
                    )

                self.assertIsInstance(attacker, FakeAttack)
                for name, value in expected.items():
                    self.assertEqual(captured[name], value)


if __name__ == "__main__":
    unittest.main()
