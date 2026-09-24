from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.stats import norm
import torch

from mia.attacks_entity import CompositeLiRAv2
from mia.run_audit import get_attacker
from mia.run_multiple_audits import build_audit_configs, load_batch_config


class TestEntityLiRAVariance(unittest.TestCase):
    '''Check shared entity variance and the six-way benchmark configuration.'''

    def setUp(self):
        '''Create two entities with unequal sizes and different reference counts.'''
        self.audit_table = {0: [0], 1: [1, 2]}
        values = torch.tensor([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [4.0, 6.0, 7.0],
            [8.0, 8.0, 9.0],
            [10.0, 10.0, 11.0],
            [12.0, 12.0, 13.0],
        ], dtype=torch.float64)
        self.shadow_loss_sigs = values.unsqueeze(-1)
        self.shadow_entity_mask = torch.tensor([
            [1, 1], [1, 1], [1, 1], [0, 1], [0, 0], [0, 0],
        ], dtype=torch.bool)

    def test_shared_variance_pools_centered_residuals(self):
        '''Check global and local shared variance use the right residual degrees of freedom.'''
        sum_squares = {}
        degrees = {}
        for entity_id, indices in self.audit_table.items():
            values = -self.shadow_loss_sigs[:, indices, 0].numpy()
            memberships = self.shadow_entity_mask[:, entity_id].numpy()
            in_values = values[memberships]
            out_values = values[~memberships]
            in_residual = in_values - in_values.mean(axis=0)
            out_residual = out_values - out_values.mean(axis=0)
            sum_squares[entity_id] = np.square(in_residual).sum() + np.square(out_residual).sum()
            degrees[entity_id] = (len(in_values) + len(out_values) - 2) * len(indices)
        for use_global_dispersion in (True, False):
            with self.subTest(use_global_dispersion=use_global_dispersion):
                attacker = CompositeLiRAv2(
                    self.audit_table,
                    self.shadow_loss_sigs,
                    self.shadow_entity_mask,
                    offline=False,
                    covariance="spherical",
                    use_global_dispersion=use_global_dispersion,
                    share_variance=True,
                )
                if use_global_dispersion:
                    expected = sum(sum_squares.values()) / sum(degrees.values())
                for entity_id in self.audit_table:
                    if not use_global_dispersion:
                        expected = sum_squares[entity_id] / degrees[entity_id]
                    variance_in = attacker.var_in if use_global_dispersion else attacker.var_in[entity_id]
                    variance_out = attacker.var_out if use_global_dispersion else attacker.var_out[entity_id]
                    self.assertAlmostEqual(float(variance_in), expected)
                    self.assertAlmostEqual(float(variance_out), expected)
                scores = attacker.run_attack(self.shadow_loss_sigs[0])
                for entity_id, indices in self.audit_table.items():
                    target = -self.shadow_loss_sigs[0, indices, 0].numpy()
                    center_in = attacker.mean_in[entity_id].numpy()
                    center_out = attacker.mean_out[entity_id].numpy()
                    variance = attacker.var_in if use_global_dispersion else attacker.var_in[entity_id]
                    scale = np.sqrt(float(variance))
                    expected_score = norm.logpdf(target, loc=center_in, scale=scale).sum()
                    expected_score -= norm.logpdf(target, loc=center_out, scale=scale).sum()
                    self.assertAlmostEqual(float(scores[entity_id]), expected_score, places=5)

    def test_shared_variance_requires_online_spherical_covariance(self):
        '''Reject configurations where sharing IN and OUT variance is undefined.'''
        for offline, covariance in ((True, "spherical"), (False, "diagonal")):
            with self.subTest(offline=offline, covariance=covariance):
                with self.assertRaisesRegex(ValueError, "online spherical"):
                    CompositeLiRAv2(
                        self.audit_table,
                        self.shadow_loss_sigs,
                        self.shadow_entity_mask,
                        offline=offline,
                        covariance=covariance,
                        share_variance=True,
                    )

    def test_factory_and_benchmark_cover_six_settings(self):
        '''Check factory dispatch and eight paired matched/mixed benchmark cases.'''
        attack_config = SimpleNamespace(
            attack="CompositeLiRAv2",
            offline=False,
            covariance="spherical",
            use_global_dispersion=False,
            share_variance=True,
        )
        attacker = get_attacker(
            attack_config,
            self.shadow_loss_sigs,
            audit_table=self.audit_table,
            shadow_entity_mask=self.shadow_entity_mask,
        )
        self.assertTrue(attacker.share_variance)
        self.assertFalse(attacker.use_global_dispersion)
        config_path = Path("mia/configs/config_entity_lira_variance_benchmark.yaml")
        config = load_batch_config(config_path)
        audit_configs = build_audit_configs(config)
        cases = {audit_config["case"] for audit_config in audit_configs}
        settings = {
            (
                audit_config["attack"]["offline"],
                audit_config["attack"]["use_global_dispersion"],
                audit_config["attack"].get("share_variance", False),
            )
            for audit_config in audit_configs
        }
        self.assertEqual(len(audit_configs), 48)
        self.assertEqual(len(cases), 8)
        self.assertEqual(settings, {
            (False, True, True), (False, True, False),
            (False, False, True), (False, False, False),
            (True, True, False), (True, False, False),
        })
        self.assertTrue(all(audit_config["loss_normalization"] == "log_standardized" for audit_config in audit_configs))


if __name__ == "__main__":
    unittest.main()
