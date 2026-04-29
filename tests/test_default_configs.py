from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from mia import loss_query as loss_query_module


class TestDefaultConfigs(unittest.TestCase):
    def _load(self, relative_path):
        with open(Path(relative_path), "r") as file:
            return yaml.safe_load(file)

    def test_loss_query_cli_directory_defaults(self):
        root = Path("/tmp/repo")
        with patch.object(loss_query_module.utils, "get_root", return_value=str(root)):
            args = loss_query_module.parse_args([
                "--checkpoint-paths",
                "/tmp/DDPM-CIFAR10-smpl-f0p5-s0-sz32-epoch4.pth",
                "--dataset",
                "CIFAR10",
                "--batch-size",
                "2",
                "--n-loss-samples",
                "3",
                "--noise-level",
                "0.25",
            ])

        self.assertEqual(args.data_dir, root / "datasets")
        self.assertEqual(args.res_dir, root / "mia/results")

    def test_run_audit_sample_default_configs(self):
        for relative_path in ("mia/configs/config_audit_sample.yaml",):
            config = self._load(relative_path)
            self.assertIsInstance(config, dict)
            self.assertIn("dataset", config)
            self.assertIn("data_dir", config)
            self.assertIn("res_dir", config)
            self.assertIn("audit_mode", config)
            self.assertIn("round_robin", config)
            self.assertIn("target_loss_paths", config)
            self.assertIn("shadow_loss_paths", config)
            self.assertIn("attack", config)
            self.assertIsInstance(config["target_loss_paths"], list)
            self.assertIsInstance(config["shadow_loss_paths"], list)
            self.assertIsInstance(config["attack"], dict)

    def test_run_audit_entity_default_config(self):
        config = self._load("mia/configs/config_audit_entity.yaml")
        self.assertIsInstance(config, dict)
        self.assertIn("dataset", config)
        self.assertIn("res_dir", config)
        self.assertIn("audit_mode", config)
        self.assertIn("round_robin", config)
        self.assertIn("mode", config)
        self.assertIn("min_samples_per_entity", config)
        self.assertIn("max_samples_per_entity", config)
        self.assertNotIn("n_audit_samples_per_entity", config)
        self.assertIn("target_loss_paths", config)
        self.assertIn("shadow_loss_paths", config)
        self.assertIn("attack", config)
        self.assertIsInstance(config["target_loss_paths"], list)
        self.assertIsInstance(config["shadow_loss_paths"], list)
        self.assertIsInstance(config["attack"], dict)

    def test_evaluation_default_config(self):
        config = self._load("mia/configs/config_evaluation.yaml")
        self.assertIsInstance(config, dict)
        for key in ("res_dir", "metrics_folders", "low_exponent"):
            self.assertIn(key, config)
        self.assertIsInstance(config["res_dir"], str)
        self.assertIsInstance(config["metrics_folders"], list)
        self.assertTrue(all(isinstance(folder, str) for folder in config["metrics_folders"]))
        self.assertIsInstance(config["low_exponent"], int)


if __name__ == "__main__":
    unittest.main()
