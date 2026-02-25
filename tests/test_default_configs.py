from pathlib import Path
import unittest

import yaml


class TestDefaultConfigs(unittest.TestCase):
    def _load(self, relative_path):
        with open(Path(relative_path), "r") as file:
            return yaml.safe_load(file)

    def test_run_mia_default_config(self):
        config = self._load("mia/configs/config_mia.yaml")
        self.assertEqual(config["dataset"], "CelebA2")
        self.assertEqual(config["data_dir"], "")
        self.assertEqual(config["res_dir"], "")
        self.assertEqual(config["batch_size"], 512)
        self.assertFalse(config["round_robin"])
        self.assertEqual(config["target_model_paths"], [""])
        self.assertEqual(config["shadow_model_paths"], [""])
        self.assertEqual(config["attack"]["attack"], "BASE")
        self.assertEqual(config["attack"]["prior"], 0.5)
        self.assertEqual(config["attack"]["n_loss_samples"], 1)

    def test_run_audit_sample_default_configs(self):
        for relative_path in ("mia/configs/config_audit_sample.yaml",):
            config = self._load(relative_path)
            self.assertEqual(config["dataset"], "CelebA2")
            self.assertEqual(config["data_dir"], "")
            self.assertEqual(config["res_dir"], "")
            self.assertEqual(config["audit_mode"], "sample")
            self.assertEqual(config["n_audit_samples"], 30000)
            self.assertEqual(config["target_model_paths"], [""])
            self.assertEqual(config["attack"]["attack"], "BASE")
            self.assertEqual(config["attack"]["prior"], 0.5)
            self.assertIsNone(config["attack"]["n_loss_samples"])

    def test_run_audit_entity_default_config(self):
        config = self._load("mia/configs/config_audit_entity.yaml")
        self.assertEqual(config["dataset"], "CelebA2")
        self.assertEqual(config["data_dir"], "")
        self.assertEqual(config["res_dir"], "")
        self.assertEqual(config["audit_mode"], "entity")
        self.assertEqual(config["entity_audit_mode"], "all")
        self.assertIsNone(config["entity_audit_n_audit_samples_per_entity"])
        self.assertIsNone(config["entity_audit_min_samples_per_entity"])
        self.assertIsNone(config["entity_audit_max_samples_per_entity"])
        self.assertEqual(config["target_model_paths"], [""])
        self.assertEqual(config["attack"]["attack"], "CompositeBASE")
        self.assertEqual(config["attack"]["prior"], 0.5)


if __name__ == "__main__":
    unittest.main()
