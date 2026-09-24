'''Validate continuous target-wide latent inference for HBE_GlobalLatent.'''
import inspect
import unittest

import numpy as np
import torch

from mia.attacks_entity import HBE_GlobalLatent


class EntityHGGlobalLatentTests(unittest.TestCase):
    '''Exercise the information boundary, contrast invariance and cross-fitting behavior.'''

    @classmethod
    def setUpClass(cls):
        '''Create structured reference variation and a target with entity-wide membership shifts.'''
        rng = np.random.default_rng(732)
        cls.n_entities = 12
        cls.entity_size = 4
        cls.n_images = cls.n_entities * cls.entity_size
        cls.table = {
            entity: np.arange(entity * cls.entity_size, (entity + 1) * cls.entity_size)
            for entity in range(cls.n_entities)
        }
        first = np.arange(cls.n_entities) % 2 == 0
        second = np.arange(cls.n_entities) % 3 == 0
        masks = np.stack((first, ~first, second, ~second, ~first, first, ~second, second))
        cls.entity_mask = torch.tensor(masks)
        center = rng.normal(0.0, 0.25, cls.n_images)
        probe = np.tile(np.array([-1.0, -0.25, 0.35, 1.1]), cls.n_entities)
        loading = 0.18 * probe + rng.normal(0.0, 0.015, cls.n_images)
        coordinates = np.array([-1.6, -1.1, -0.6, -0.2, 0.2, 0.6, 1.1, 1.6])
        evidence = np.empty((8, cls.n_images, 8), dtype=float)
        unit_entities = np.repeat(np.arange(cls.n_entities), cls.entity_size)
        for model in range(8):
            local = rng.normal(0.0, 0.025, cls.n_images)
            member_shift = 0.35 * masks[model, unit_entities]
            evidence[model] = center[:, None] + loading[:, None] * coordinates[model]
            evidence[model] += local[:, None] + member_shift[:, None]
            evidence[model] += rng.normal(0.0, 0.04, (cls.n_images, 8))
        target_members = np.arange(cls.n_entities) % 2 == 0
        target_shift = 0.4 * target_members[unit_entities]
        target_evidence = center[:, None] + 0.75 * loading[:, None] + target_shift[:, None]
        target_evidence = target_evidence + rng.normal(0.0, 0.04, (cls.n_images, 8))
        cls.shadows = torch.tensor(-evidence, dtype=torch.float64)
        cls.target = torch.tensor(-target_evidence, dtype=torch.float64)
        cls.options = dict(n_folds=3, quadrature_nodes=12, max_hyper_units=256, factor_steps=30)
        cls.fit = HBE_GlobalLatent(
            cls.table,
            cls.shadows,
            cls.entity_mask,
            **cls.options,
        )

    def test_information_boundary_and_factor_fit(self):
        '''Accept no family metadata and recover nontrivial structured reference variation.'''
        parameters = inspect.signature(HBE_GlobalLatent).parameters
        self.assertNotIn("reference_groups", parameters)
        self.assertNotIn("target_family", parameters)
        self.assertTrue(self.fit.fit_diagnostics["reference_only_fit"])
        self.assertFalse(self.fit.fit_diagnostics["target_family_used"])
        self.assertGreater(self.fit.fit_diagnostics["factor_explained_fraction"], 0.1)
        self.assertGreater(self.fit.fit_diagnostics["loading_sd"], 0.0)

    def test_entity_offsets_do_not_change_latent_inference(self):
        '''Adding arbitrary constants within target entities leaves all contrast posteriors unchanged.'''
        first_mean, first_variance, _ = self.fit.infer_target_latent(self.target)
        changed = self.target.clone()
        offsets = np.linspace(-0.7, 0.7, self.n_entities)
        for entity, offset in enumerate(offsets):
            changed[self.table[entity]] -= offset
        second_mean, second_variance, _ = self.fit.infer_target_latent(changed)
        np.testing.assert_allclose(first_mean, second_mean, rtol=0, atol=1e-11)
        np.testing.assert_allclose(first_variance, second_variance, rtol=0, atol=1e-14)

    def test_scored_fold_cannot_change_its_latent_posterior(self):
        '''Changing one entity's contrast pattern cannot affect the posterior used for its fold.'''
        first_mean, _, _ = self.fit.infer_target_latent(self.target)
        changed = self.target.clone()
        pattern = torch.tensor([0.5, -0.4, 0.25, -0.35], dtype=changed.dtype)
        changed[self.table[0]] -= pattern[:, None]
        second_mean, _, _ = self.fit.infer_target_latent(changed)
        fold = self.fit.fold_ids[0]
        self.assertAlmostEqual(first_mean[fold], second_mean[fold], places=12)
        other_folds = np.arange(self.fit.n_folds) != fold
        self.assertGreater(np.max(np.abs(first_mean[other_folds] - second_mean[other_folds])), 1e-4)

    def test_scores_are_finite_and_entity_level_monotone(self):
        '''An entity-wide evidence increase changes only that entity and raises its upper-tail score.'''
        first = np.array([float(value) for value in self.fit.run_attack(self.target).values()])
        changed = self.target.clone()
        changed[self.table[0]] -= 0.2
        second = np.array([float(value) for value in self.fit.run_attack(changed).values()])
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(second[0], first[0])
        np.testing.assert_allclose(second[1:], first[1:], rtol=0, atol=1e-11)

    def test_o_only_reference_invariance(self):
        '''Changing reference losses for member entities leaves factor and predictions unchanged.'''
        changed = self.shadows.clone()
        sample_mask = self.entity_mask[:, np.repeat(np.arange(self.n_entities), self.entity_size)]
        changed[sample_mask] += 100.0
        replay = HBE_GlobalLatent(
            self.table,
            changed,
            self.entity_mask,
            **self.options,
        )
        first = np.array([float(value) for value in self.fit.run_attack(self.target).values()])
        second = np.array([float(value) for value in replay.run_attack(self.target).values()])
        np.testing.assert_allclose(self.fit.loading, replay.loading, rtol=0, atol=1e-10)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-9)

    def test_target_diagnostics_are_label_free_and_finite(self):
        '''Report stable cross-fitted coordinates and contrast checks without evaluation labels.'''
        self.fit.run_attack(self.target)
        diagnostics = self.fit.last_target_diagnostics
        for name in ("fold_mean_range", "mean_posterior_sd", "mean_contrast_log_density",
                     "mean_chi_square_ratio", "contrast_pit_mean"):
            self.assertTrue(np.isfinite(diagnostics[name]))
        self.assertEqual(len(diagnostics["fold_mean"]), self.fit.n_folds)
        self.assertLess(diagnostics["fold_mean_range"], 0.5)


if __name__ == "__main__":
    unittest.main()
