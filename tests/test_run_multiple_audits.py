import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from mia import run_multiple_audits
from mia import result_store


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
        batch_config = self._batch_config()
        batch_config["n_loss_samples"] = 2
        audit_configs = run_multiple_audits.build_audit_configs(batch_config)

        self.assertEqual(len(audit_configs), 4)
        self.assertEqual(audit_configs[0]["target_loss_paths"], ["dataset-a"])
        self.assertEqual(audit_configs[0]["shadow_loss_paths"], ["dataset-a"])
        self.assertEqual(audit_configs[1]["attack"]["name"], "LiRA")
        self.assertEqual(audit_configs[2]["target_loss_paths"], ["dataset-b"])
        self.assertTrue(all(config["print_summary"] for config in audit_configs))
        self.assertTrue(all(config["n_loss_samples"] == 2 for config in audit_configs))

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

        self.assertEqual(len(list(config_dir.glob("*.yaml"))), 13)
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
                    ["CompositeLiRA", "CompositeBASE", "HBE_Simple", "HBE_GlobalLatent"],
                )

    def test_sample_mismatched_configs_use_mixed_reference_pools(self):
        online_path = Path("mia/configs/config_sample_mismatched_online.yaml")
        offline_path = Path("mia/configs/config_sample_mismatched_offline.yaml")
        online_config = run_multiple_audits.load_batch_config(online_path)
        offline_config = run_multiple_audits.load_batch_config(offline_path)

        self.assertEqual(online_config["loss_path_pairs"], offline_config["loss_path_pairs"])
        self.assertEqual(len(online_config["loss_path_pairs"]), 4)
        for loss_path_pair in online_config["loss_path_pairs"]:
            target_paths = loss_path_pair["target_loss_paths"]
            reference_paths = loss_path_pair["shadow_loss_paths"]
            target_dataset = "CIFAR10" if "CIFAR10" in target_paths[0] else "CelebA"

            self.assertEqual(len(target_paths), 1)
            self.assertEqual(len(reference_paths), 2)
            self.assertIn(target_paths[0], reference_paths)
            self.assertEqual(sum("/DDPM-" in path for path in reference_paths), 1)
            self.assertEqual(sum("/FM-" in path for path in reference_paths), 1)
            self.assertTrue(all(target_dataset in path for path in reference_paths))

    def test_run_audits_dispatches_isolated_processes(self):
        audit_configs = run_multiple_audits.build_audit_configs(self._batch_config())
        dispatched_configs = []

        def capture_run(command, check):
            self.assertTrue(check)
            with open(command[4], "r") as file:
                config = yaml.safe_load(file)
                dispatched_configs.append(config)
            result_path = Path(config["results_root"]) / f"metrics-{len(dispatched_configs)}.pkl"
            result_store.write_artifact(config["manifest_path"], {
                "schema_version": 2, "metrics_paths": [str(result_path)],
            })

        with tempfile.TemporaryDirectory() as directory:
            for config in audit_configs:
                config["results_root"] = directory
            with patch.object(run_multiple_audits.subprocess, "run", side_effect=capture_run):
                run_multiple_audits.run_audits(audit_configs)
            manifest_path = Path(directory) / "sample_core_offline" / "audit_manifest.json"
            self.assertEqual(len(result_store.load_manifest_paths(manifest_path)), 4)

        self.assertEqual(len(dispatched_configs), 4)
        self.assertEqual(dispatched_configs[0]["attack"]["name"], "BASE")


if __name__ == "__main__":
    unittest.main()
