import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import run_audit
from mia import result_store


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
                "attack": {"attack": "LiRA"},
            })
            metrics = {"AUC": 0.7}

            metrics_path = result_store.save_audit_metrics(
                config,
                metrics,
                loss_path,
                [],
                "LiRA",
            )

            expected_dir = (Path(temporary_dir) / "sample_core_offline").resolve()
            self.assertEqual(metrics_path.parent, expected_dir)
            self.assertRegex(
                metrics_path.name,
                r"^metrics_attack-LiRA_target-DDPM-CelebA-smpl-f0p5-s0-sz64-epoch400_sample_loss-n64-nl0p1_cfg-[0-9a-f]{16}\.pkl$",
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

    def test_result_names_separate_settings_and_store_resolved_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = run_audit.utils.Config({
                "results_root": directory, "results_dir_name": "review", "audit_mode": "sample",
                "attack": {"attack": "BASE"},
            })
            stem = "DDPM-CelebA-smpl-f0p5-s0-sz64-epoch400"
            target = Path(directory) / f"loss-signals-{stem}-n64-nl0p1.pkl"
            shadow = Path(directory) / f"loss-signals-{stem.replace('-s0-', '-s1-')}-n64-nl0p1.pkl"
            metrics = {"AUC": 0.7, "audit_indices": [0, 2]}
            first = result_store.save_audit_metrics(config, dict(metrics), target, [shadow], "BASE")
            repeated = result_store.save_audit_metrics(config, dict(metrics), target, [shadow], "BASE")
            self.assertEqual(first, repeated)
            alternatives = []
            for noise in ("0p2", "0p3"):
                changed_target = Path(str(target).replace("nl0p1", f"nl{noise}"))
                alternatives.append(result_store.save_audit_metrics(config, dict(metrics), changed_target, [shadow], "BASE"))
            config.n_loss_samples = 32
            alternatives.append(result_store.save_audit_metrics(config, dict(metrics), target, [shadow], "BASE"))
            config.seed = 9
            alternatives.append(result_store.save_audit_metrics(config, dict(metrics), target, [shadow], "BASE"))
            config.attack["apply_sigmoid"] = True
            alternatives.append(result_store.save_audit_metrics(config, dict(metrics), target, [shadow], "BASE"))
            self.assertEqual(len(set([first, *alternatives])), 6)
            with first.open("rb") as file:
                saved = pickle.load(file)
            self.assertEqual(saved["audit_indices"], [0, 2])
            self.assertEqual(saved["settings"]["target_loss_path"], str(target.resolve()))
            self.assertEqual(saved["settings"]["shadow_loss_paths"], [str(shadow.resolve())])
            self.assertEqual(saved["scenario"]["audit"]["loss_normalization"], "log_standardized")
            self.assertEqual(saved["settings"]["attack"]["loss_transformation"], "none")
            self.assertFalse(saved["settings"]["attack"]["apply_sigmoid"])
            original = first.read_bytes()
            with patch.object(result_store.pickle, "dumps", side_effect=RuntimeError("serialization failed")):
                with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                    result_store.write_artifact(first, {})
            self.assertEqual(first.read_bytes(), original)

    def test_base_defaults_preserve_rankings_and_allow_explicit_sigmoid(self):
        shadows = torch.full((2, 4, 1), 100.)
        masks = torch.zeros((2, 4), dtype=torch.bool)
        targets = torch.tensor([[60.], [65.], [70.], [75.]])
        for attack_name in ("BASE", "NormalBASE"):
            config = run_audit.utils.Config({"attack": attack_name, "offline": True})
            attacker = run_audit.get_attacker(config, shadows, masks)
            scores = attacker.run_attack(targets)
            self.assertEqual(scores.tolist(), [40., 35., 30., 25.])
            self.assertEqual(run_audit.evaluation.evaluate_MIA(scores, [1, 1, 0, 0])["AUC"], 1.)
            config.apply_sigmoid = True
            scores = run_audit.get_attacker(config, shadows, masks).run_attack(targets)
            self.assertEqual(scores.tolist(), [1.] * 4)


if __name__ == "__main__":
    unittest.main()
