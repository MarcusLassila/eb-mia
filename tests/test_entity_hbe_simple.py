'''Validate the canonical HBE_Simple entity attack.'''
import inspect
import unittest

import numpy as np
import torch

from mia.attacks_entity import HBE_Simple


class HBESimpleTests(unittest.TestCase):
    '''Exercise the public API, O-only fitting, and entity-tail scoring.'''

    def setUp(self):
        '''Create balanced synthetic O references for two audited entities.'''
        rng = np.random.default_rng(418)
        self.audit_table = {0: np.array([0, 1]), 1: np.array([2, 3])}
        membership = np.array([
            [True, False],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
            [False, True],
        ])
        unit_entities = np.array([0, 0, 1, 1])
        center = np.array([0.1, -0.2, 0.3, -0.1])
        evidence = np.empty((len(membership), len(center), 5))
        for model_index in range(len(membership)):
            model_offset = rng.normal(0.0, 0.08, len(center))
            member_shift = 0.3 * membership[model_index, unit_entities]
            evidence[model_index] = center[:, None] + model_offset[:, None]
            evidence[model_index] += member_shift[:, None]
            evidence[model_index] += rng.normal(0.0, 0.04, (len(center), 5))
        self.shadows = torch.tensor(-evidence, dtype=torch.float64)
        self.entity_mask = torch.tensor(membership)
        target = center[:, None] + rng.normal(0.0, 0.04, (len(center), 5))
        self.target = torch.tensor(-target, dtype=torch.float64)
        self.options = {
            "loss_transformation": "none",
            "n_gibbs_samples": 12,
            "n_gibbs_warmup": 6,
            "random_seed": 7,
        }

    def test_public_api_excludes_experimental_switches(self):
        '''Expose only the fixed canonical HG-Simple formulation.'''
        parameters = inspect.signature(HBE_Simple).parameters
        self.assertNotIn("entity_effect", parameters)
        self.assertNotIn("local_dispersion", parameters)
        self.assertNotIn("predictive_mixture", parameters)

    def test_o_only_reference_invariance_and_finite_scores(self):
        '''Changing member references cannot affect the fitted null attack.'''
        first = HBE_Simple(
            self.audit_table,
            self.shadows,
            self.entity_mask,
            **self.options,
        )
        changed = self.shadows.clone()
        sample_mask = self.entity_mask[:, np.array([0, 0, 1, 1])]
        changed[sample_mask] += 100.0
        second = HBE_Simple(
            self.audit_table,
            changed,
            self.entity_mask,
            **self.options,
        )
        first_scores = torch.stack(list(first.run_attack(self.target).values()))
        second_scores = torch.stack(list(second.run_attack(self.target).values()))

        self.assertTrue(torch.isfinite(first_scores).all())
        self.assertTrue(torch.allclose(first_scores, second_scores, atol=1e-10))

    def test_entity_evidence_increase_raises_only_its_score(self):
        '''Increasing evidence for one entity raises only that upper-tail score.'''
        attacker = HBE_Simple(
            self.audit_table,
            self.shadows,
            self.entity_mask,
            **self.options,
        )
        first = attacker.run_attack(self.target)
        changed = self.target.clone()
        changed[self.audit_table[0]] -= 0.2
        second = attacker.run_attack(changed)

        self.assertGreater(float(second[0]), float(first[0]))
        self.assertAlmostEqual(float(second[1]), float(first[1]), places=12)


if __name__ == "__main__":
    unittest.main()
