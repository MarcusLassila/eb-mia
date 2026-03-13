from pathlib import Path
import unittest

import yaml


class TestDefaultConfigs(unittest.TestCase):
    def _load(self, relative_path):
        with open(Path(relative_path), "r") as file:
            return yaml.safe_load(file)

    def test_loss_query_default_config(self):
        config = self._load("mia/configs/config_loss_query.yaml")
        self.assertIsInstance(config, dict)
        for key in (
            "dataset",
            "data_dir",
            "res_dir",
            "batch_size",
            "n_loss_samples",
            "checkpoint_paths",
            "lira_score_paths",
        ):
            self.assertIn(key, config)
        self.assertIsInstance(config["dataset"], str)
        self.assertIsInstance(config["data_dir"], str)
        self.assertIsInstance(config["res_dir"], str)
        self.assertIsInstance(config["batch_size"], int)
        self.assertIsInstance(config["n_loss_samples"], int)
        self.assertIsInstance(config["checkpoint_paths"], list)
        self.assertIsInstance(config["lira_score_paths"], list)
        self.assertTrue(all(isinstance(path, str) for path in config["checkpoint_paths"]))
        self.assertTrue(all(isinstance(path, str) for path in config["lira_score_paths"]))

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
        self.assertIn("data_dir", config)
        self.assertIn("res_dir", config)
        self.assertIn("audit_mode", config)
        self.assertIn("round_robin", config)
        self.assertIn("mode", config)
        self.assertIn("n_audit_samples_per_entity", config)
        self.assertIn("min_samples_per_entity", config)
        self.assertIn("max_samples_per_entity", config)
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
