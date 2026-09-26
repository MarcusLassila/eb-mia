import unittest
from pathlib import Path

import numpy as np
import torch

from mia.attacks_sample import HG_LiRA_r, HG_LiRA_r_shared, MIA
from mia import run_audit, run_multiple_audits


class TestHGLiRARShared(unittest.TestCase):
    '''Check independent online shared-dispersion inference and config wiring.'''

    @classmethod
    def setUpClass(cls):
        '''Build balanced synthetic reference losses and one target tensor.'''
        rng = np.random.default_rng(17)
        n_models = 6
        n_points = 24
        n_queries = 5
        mask = np.zeros((n_models, n_points), dtype=bool)
        for point in range(n_points):
            mask[(np.arange(3) + point) % n_models, point] = True
        center = np.linspace(-0.8, 0.8, n_points)
        scale = np.exp(0.35 * center)
        evidence = center[None, :, None]
        evidence = evidence + rng.normal(size=(n_models, n_points))[:, :, None] * scale[None, :, None]
        evidence = evidence + rng.normal(scale=0.25, size=(n_models, n_points, n_queries))
        evidence = evidence + mask[:, :, None] * 0.7
        cls.references = torch.tensor(-evidence, dtype=torch.float64)
        cls.mask = torch.tensor(mask)
        cls.target = torch.tensor(-rng.normal(size=(n_points, n_queries)), dtype=torch.float64)

    def test_shared_model_has_finite_reproducible_online_scores(self):
        '''Check shared local scale, separate centers, and deterministic predictions.'''
        options = {"offline": False, "sig_transformation": "none",
                   "n_gibbs_samples": 4, "n_gibbs_warmup": 3, "n_quadrature": 9, "random_seed": 42}
        attacker = HG_LiRA_r_shared(self.references, self.mask, **options)
        repeated = HG_LiRA_r_shared(self.references, self.mask, **options)
        scores = attacker.run_attack(self.target)

        self.assertEqual(HG_LiRA_r_shared.__bases__, (MIA,))
        self.assertFalse(issubclass(HG_LiRA_r_shared, HG_LiRA_r))
        self.assertIsInstance(attacker.prior_in, HG_LiRA_r_shared._ClassPrior)
        self.assertIsInstance(attacker.prior_shared, HG_LiRA_r_shared._SharedPrior)
        self.assertAlmostEqual(float(attacker.prior_shared.r.mean()), 1.0)
        self.assertTrue(np.all(attacker.prior_shared.r > 0.0))
        self.assertGreater(attacker.prior_in.xi, attacker.prior_out.xi)
        self.assertEqual(scores.shape, (24,))
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(scores, repeated.run_attack(self.target))

    def test_ddpm_online_config_resolves_three_attacks_and_dispatches_hg(self):
        '''Check the saved config and the HG dispatcher on small reference data.'''
        config_path = Path("mia/configs/config_sample_ddpm_cifar10_online.yaml")
        batch_config = run_multiple_audits.load_batch_config(config_path)
        audit_configs = run_multiple_audits.build_audit_configs(batch_config)
        names = [config["attack"]["name"] for config in audit_configs]
        self.assertEqual(names, ["BASE", "LiRA", "HG_LiRA_r_shared"])
        self.assertTrue(all(config["round_robin"] for config in audit_configs))
        self.assertTrue(all(config["n_loss_samples"] == 32 for config in audit_configs))
        self.assertTrue(all(config["loss_normalization"] == "log_standardized" for config in audit_configs))
        self.assertEqual(batch_config["results_root"], "temp_results")
        self.assertEqual(batch_config["loss_paths"], ["loss-signals/DDPM-CIFAR10-sample"])

        hg_config = dict(audit_configs[-1]["attack"])
        self.assertEqual((hg_config["n_quadrature"], hg_config["n_gibbs_warmup"], hg_config["n_gibbs_samples"]), (50, 64, 128))
        hg_config.update(n_quadrature=9, n_gibbs_warmup=3, n_gibbs_samples=4)
        attacker = run_audit.get_attacker(run_audit.utils.Config(hg_config), self.references, self.mask)
        self.assertIsInstance(attacker, HG_LiRA_r_shared)
        self.assertTrue(torch.isfinite(attacker.run_attack(self.target)).all())

    def test_shared_model_requires_online_balanced_references(self):
        '''Reject unsupported scoring and membership layouts before fitting.'''
        with self.assertRaisesRegex(ValueError, "online"):
            HG_LiRA_r_shared(self.references, self.mask, offline=True)
        unbalanced_mask = self.mask.clone()
        unbalanced_mask[0, 0] = ~unbalanced_mask[0, 0]
        with self.assertRaisesRegex(ValueError, "equally many"):
            HG_LiRA_r_shared(self.references, unbalanced_mask)
