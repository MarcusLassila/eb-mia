from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.stats import norm
import torch

from mia.attacks_sample import LiRA
from mia.run_audit import get_attacker
from mia.run_multiple_audits import build_audit_configs, load_batch_config


class TestLiRA(unittest.TestCase):
    '''Check the four variance choices and benchmark configuration.'''

    def setUp(self):
        '''Create reference means with distinct IN, OUT, and per-image variances.'''
        reference_means = torch.tensor([
            [1.0, 2.0], [1.2, 2.2], [0.8, 1.8],
            [2.0, 4.0], [3.0, 4.5], [4.0, 5.0],
        ], dtype=torch.float64)
        query_offsets = torch.tensor([-0.1, 0.1], dtype=torch.float64)
        self.shadow_loss_sigs = reference_means[:, :, None] + query_offsets
        self.shadow_train_mask = torch.tensor([[1, 1]] * 3 + [[0, 0]] * 3, dtype=torch.bool)
        self.target_loss_sigs = torch.tensor([
            [1.4, 1.6], [3.9, 4.1],
        ], dtype=torch.float64)

    def test_all_four_variance_estimates(self):
        '''Check global/local and shared/separate sample-mean variances.'''
        expected = {
            (True, True): (0.3325, 0.3325),
            (False, True): (np.array([0.52, 0.145]), np.array([0.52, 0.145])),
            (True, False): (0.04, 0.625),
            (False, False): (np.array([0.04, 0.04]), np.array([1.0, 0.25])),
        }
        for (use_global_var, share_variance), (variance_in, variance_out) in expected.items():
            with self.subTest(use_global_var=use_global_var, share_variance=share_variance):
                attacker = LiRA(
                    self.shadow_loss_sigs,
                    self.shadow_train_mask,
                    use_global_var=use_global_var,
                    share_variance=share_variance,
                )
                np.testing.assert_allclose(np.square(attacker.std_in), variance_in, atol=1e-12)
                np.testing.assert_allclose(np.square(attacker.std_out), variance_out, atol=1e-12)
                target_mean = -self.target_loss_sigs.mean(dim=-1).numpy()
                expected_score = norm.logpdf(target_mean, loc=attacker.mean_in, scale=attacker.std_in)
                expected_score -= norm.logpdf(target_mean, loc=attacker.mean_out, scale=attacker.std_out)
                np.testing.assert_allclose(attacker.run_attack(self.target_loss_sigs), expected_score)


    def test_scores_only_the_sample_mean(self):
        '''Check target query spread cannot change the score at fixed mean.'''
        attacker = LiRA(self.shadow_loss_sigs, self.shadow_train_mask, share_variance=False)
        dispersed_targets = self.target_loss_sigs + torch.tensor([-0.5, 0.5])
        np.testing.assert_allclose(
            attacker.run_attack(self.target_loss_sigs),
            attacker.run_attack(dispersed_targets),
        )

    def test_rejects_offline_inference(self):
        '''Check LiRA rejects the unsupported offline setting.'''
        with self.assertRaisesRegex(ValueError, "online inference only"):
            LiRA(self.shadow_loss_sigs, self.shadow_train_mask, offline=True)

    def test_audit_factory_constructs_lira(self):
        '''Check audit options reach LiRA.'''
        attack_config = SimpleNamespace(
            attack="LiRA",
            offline=False,
            loss_transformation="none",
            use_global_dispersion=False,
            share_variance=False,
        )
        attacker = get_attacker(attack_config, self.shadow_loss_sigs, self.shadow_train_mask)
        self.assertIsInstance(attacker, LiRA)
        self.assertFalse(attacker.use_global_var)
        self.assertFalse(np.array_equal(attacker.std_in, attacker.std_out))

    def test_benchmark_compares_four_attacks_across_eight_cases(self):
        '''Check all four variance settings use the same eight benchmark cases.'''
        config_path = Path("mia/configs/config_sample_lira_v2_benchmark.yaml")
        config = load_batch_config(config_path)
        audit_configs = build_audit_configs(config)
        case_names = {audit_config["case"] for audit_config in audit_configs}
        attack_options = {
            (audit_config["attack"]["use_global_dispersion"], audit_config["attack"]["share_variance"])
            for audit_config in audit_configs
        }
        self.assertEqual(len(audit_configs), 32)
        self.assertEqual(len(case_names), 8)
        self.assertEqual(attack_options, {(True, True), (True, False), (False, True), (False, False)})
        self.assertTrue(all(audit_config["loss_normalization"] == "log_standardized" for audit_config in audit_configs))


if __name__ == "__main__":
    unittest.main()
