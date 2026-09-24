import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mia import run_multiple_audits


class TestRunMultipleAudits(unittest.TestCase):
    def _batch_config(self):
        return {
            "results_dir_name": "sample_core_offline",
            "audit_mode": "sample",
            "round_robin": True,
            "loss_paths": ["dataset-a", "dataset-b"],
            "attacks": [
                {"name": "BASE", "attack": "BASE", "offline": True},
                {"name": "LiRA", "attack": "LiRA", "offline": False},
            ],
        }

    def test_build_audit_configs_expands_paths_and_attacks(self):
        audit_configs = run_multiple_audits.build_audit_configs(self._batch_config())

        self.assertEqual(len(audit_configs), 4)
        self.assertEqual(audit_configs[0]["target_loss_paths"], ["dataset-a"])
        self.assertEqual(audit_configs[0]["shadow_loss_paths"], ["dataset-a"])
        self.assertEqual(audit_configs[1]["attack"]["name"], "LiRA")
        self.assertEqual(audit_configs[2]["target_loss_paths"], ["dataset-b"])
        self.assertTrue(all(config["print_summary"] for config in audit_configs))

    def test_build_audit_configs_supports_entity_groups(self):
        batch_config = {
            "results_dir_name": "entity_core_offline",
            "audit_mode": "entity",
            "round_robin": True,
            "audit_groups": [
                {
                    "dataset": "CelebA",
                    "mode": "exclude_train",
                    "loss_paths": ["celeba"],
                },
                {
                    "dataset": "VGGFace2",
                    "mode": "hold_out",
                    "loss_paths": ["vggface2"],
                },
            ],
            "attacks": [{"name": "CompositeBASE", "attack": "CompositeBASE"}],
        }

        audit_configs = run_multiple_audits.build_audit_configs(batch_config)

        self.assertEqual(len(audit_configs), 2)
        self.assertEqual(audit_configs[0]["dataset"], "CelebA")
        self.assertEqual(audit_configs[0]["mode"], "exclude_train")
        self.assertEqual(audit_configs[1]["dataset"], "VGGFace2")
        self.assertEqual(audit_configs[1]["mode"], "hold_out")

    def test_retained_benchmark_configs_expand(self):
        config_dir = Path("mia/configs")
        config_paths = sorted(config_dir.glob("config_*_*_*.yaml"))

        self.assertEqual(len(list(config_dir.glob("*.yaml"))), 12)
        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                config = run_multiple_audits.load_batch_config(config_path)
                audit_configs = run_multiple_audits.build_audit_configs(config)
                self.assertTrue(audit_configs)
                self.assertIn("results_dir_name", config)
                self.assertNotIn("res_dir", config)

    def test_mismatched_configs_share_targets_and_shadows(self):
        for audit_mode in ("sample", "entity"):
            online_path = Path(f"mia/configs/config_{audit_mode}_mismatched_online.yaml")
            offline_path = Path(f"mia/configs/config_{audit_mode}_mismatched_offline.yaml")
            online_config = run_multiple_audits.load_batch_config(online_path)
            offline_config = run_multiple_audits.load_batch_config(offline_path)

            self.assertEqual(online_config["loss_path_pairs"], offline_config["loss_path_pairs"])
            online_attacks = [attack["attack"] for attack in online_config["attacks"]]
            offline_attacks = [attack["attack"] for attack in offline_config["attacks"]]
            if audit_mode == "sample":
                self.assertEqual(online_attacks, ["BASE", "LiRA", "HG_LiRA_r"])
            else:
                self.assertEqual(online_attacks, ["CompositeLiRA", "CompositeBASE"])
                self.assertEqual(
                    offline_attacks,
                    ["CompositeLiRAv2", "CompositeBASE", "HBE_Simple", "HBE_GlobalLatent"],
                )

    def test_run_audits_dispatches_isolated_processes(self):
        audit_configs = run_multiple_audits.build_audit_configs(self._batch_config())
        dispatched_configs = []

        def capture_run(command, check):
            self.assertTrue(check)
            with open(command[4], "r") as file:
                dispatched_configs.append(yaml.safe_load(file))

        with patch.object(run_multiple_audits.subprocess, "run", side_effect=capture_run):
            run_multiple_audits.run_audits(audit_configs)

        self.assertEqual(len(dispatched_configs), 4)
        self.assertEqual(dispatched_configs[0]["attack"]["name"], "BASE")


if __name__ == "__main__":
    unittest.main()
