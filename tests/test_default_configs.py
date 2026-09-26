from pathlib import Path
import unittest

import yaml


class TestDefaultConfigs(unittest.TestCase):
    def test_config_catalog_contains_only_current_audits(self):
        config_paths = sorted(Path("mia/configs").glob("*.yaml"))
        expected_names = {
            "config_entity_core_offline.yaml",
            "config_entity_core_online.yaml",
            "config_entity_lira_exploration_benchmark.yaml",
            "config_entity_lira_variance_benchmark.yaml",
            "config_entity_mismatched_offline.yaml",
            "config_entity_mismatched_online.yaml",
            "config_sample_core_offline.yaml",
            "config_sample_core_online.yaml",
            "config_sample_ddpm_cifar10_online.yaml",
            "config_sample_hg_shape_exploration_benchmark.yaml",
            "config_sample_mismatched_offline.yaml",
            "config_sample_mismatched_online.yaml",
            "config_sample_lira_v2_benchmark.yaml",
        }

        self.assertEqual({path.name for path in config_paths}, expected_names)
        for config_path in config_paths:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            self.assertIn("results_dir_name", config)
            self.assertNotIn("res_dir", config)


if __name__ == "__main__":
    unittest.main()
