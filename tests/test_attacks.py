import unittest
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t

from mia import attacks_entity
from mia import attacks_sample


class TestAttacks(unittest.TestCase):
    def _hierarchical_gibbs_inputs(self):
        '''Create balanced hierarchical Gibbs test inputs and return shadows, masks, and targets.'''
        shadow_train_mask = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=bool)
        data_point_mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        membership_offset = np.where(shadow_train_mask, -0.3, 0.3)
        model_offset = np.linspace(-0.04, 0.04, 6, dtype=np.float64)
        sample_offset = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        shadow_loss_sigs = data_point_mean[None, :, None]
        shadow_loss_sigs = shadow_loss_sigs + membership_offset[:, :, None]
        shadow_loss_sigs = shadow_loss_sigs + model_offset[:, None, None]
        shadow_loss_sigs = shadow_loss_sigs + sample_offset[None, None, :]
        target_loss_sigs = data_point_mean[:, None] - 0.25
        target_loss_sigs = target_loss_sigs + sample_offset[None, :]
        shadow_tensor = torch.tensor(shadow_loss_sigs, dtype=torch.float32)
        mask_tensor = torch.tensor(shadow_train_mask, dtype=torch.bool)
        target_tensor = torch.tensor(target_loss_sigs, dtype=torch.float32)
        return shadow_tensor, mask_tensor, target_tensor

    def test_base_offline_uses_only_shadow_non_members(self):
        shadow_loss_sigs = torch.tensor([
            [1.0, 5.0],
            [9.0, 0.5],
        ], dtype=torch.float32)
        shadow_train_mask = torch.tensor([
            [1, 0],
            [0, 1],
        ], dtype=torch.bool)
        target_loss_sigs = torch.tensor([1.5, 0.75], dtype=torch.float32)

        score = attacks_sample.BASE(
            shadow_loss_sigs=shadow_loss_sigs[:, :, None],
            shadow_train_mask=shadow_train_mask,
            offline=True,
            prior=0.5,
        ).run_attack(target_loss_sigs[:, None])

        expected = torch.sigmoid(torch.tensor([7.5, 4.25], dtype=torch.float32))
        self.assertTrue(torch.allclose(score, expected, atol=1e-6))

    def test_normal_base_switches_between_global_and_per_sample_variance(self):
        shadow_loss_sigs = torch.tensor([
            [9.0, 2.0],
            [1.0, 9.0],
            [3.0, 10.0],
        ], dtype=torch.float32)
        shadow_train_mask = torch.tensor([
            [1, 0],
            [0, 1],
            [0, 0],
        ], dtype=torch.bool)

        global_attacker = attacks_sample.NormalBASE(
            shadow_loss_sigs=shadow_loss_sigs[:, :, None],
            shadow_train_mask=shadow_train_mask,
            offline=True,
            prior=0.5,
            use_global_var=True,
        )
        local_attacker = attacks_sample.NormalBASE(
            shadow_loss_sigs=shadow_loss_sigs[:, :, None],
            shadow_train_mask=shadow_train_mask,
            offline=True,
            prior=0.5,
            use_global_var=False,
        )

        mean = torch.tensor([-2.0, -6.0], dtype=torch.float32)
        expected_global_var = torch.tensor([torch.tensor([1.0, 2.0, 3.0, 10.0]).var()] * 2, dtype=torch.float32)
        expected_local_var = torch.tensor([1.0, 16.0], dtype=torch.float32)

        self.assertTrue(torch.allclose(global_attacker.ref, mean + 0.5 * expected_global_var, atol=1e-6))
        self.assertTrue(torch.allclose(local_attacker.ref, mean + 0.5 * expected_local_var, atol=1e-6))

    def test_lira_runs_directly_on_target_loss_signals(self):
        attacker = attacks_sample.LiRA(
            shadow_loss_sigs=torch.tensor([
                [[0.9], [1.1], [1.2], [1.4]],
                [[1.4], [1.5], [0.8], [0.9]],
                [[1.0], [1.2], [1.3], [1.5]],
                [[1.5], [1.6], [0.9], [1.0]],
            ], dtype=torch.float32),
            shadow_train_mask=torch.tensor([
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ], dtype=torch.bool),
            offline=False,
            loss_transformation="none",
        )

        score = attacker.run_attack(torch.tensor([[0.95], [1.05], [0.85], [0.95]], dtype=torch.float32))

        self.assertEqual(score.shape, (4,))
        self.assertTrue(torch.isfinite(score).all())

    def test_lira_defaults_to_global_variance_with_small_membership_classes(self):
        shadow_loss_sigs = torch.ones((4, 2, 1), dtype=torch.float32)
        shadow_train_mask = torch.tensor([
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ], dtype=torch.bool)

        attacker = attacks_sample.LiRA(
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_train_mask=shadow_train_mask,
            offline=False,
            loss_transformation="none",
        )

        self.assertTrue(attacker.use_global_var)






    def test_hg_lira_r_offline_score_matches_manual_cdf_mixture(self):
        '''Compare offline HG_LiRA_r with its manual predictive CDF mixture.'''
        shadow_sigs, shadow_mask, target_sigs = self._hierarchical_gibbs_inputs()
        attacker = attacks_sample.HG_LiRA_r(
            ref_sigs=shadow_sigs,
            ref_train_mask=shadow_mask,
            offline=True,
            sig_transformation="none",
            n_gibbs_samples=2,
            n_gibbs_warmup=1,
            n_quadrature=7,
        )
        target_y = attacker._sig_transformation(target_sigs.numpy())
        target_mean = target_y.mean(axis=1)
        degrees_freedom = 2.0 * attacker.eta_out.a_beta
        mean_scale = np.sqrt(
            attacker.eta_out.b_beta
            / (attacker.eta_out.a_beta * attacker.n_sigs)
        )
        nodes, weights = np.polynomial.hermite.hermgauss(
            attacker.n_quadrature
        )
        state_log_cdf = []
        for state_index in range(attacker.n_gibbs_samples):
            node_log_cdf = []
            for node, weight in zip(nodes, weights):
                predictive_mean = attacker.Ms_out[state_index]
                predictive_mean = predictive_mean + (
                    np.sqrt(2.0 * attacker.Vs_out[state_index]) * node
                )
                log_cdf = student_t.logcdf(
                    target_mean,
                    df=degrees_freedom,
                    loc=predictive_mean,
                    scale=mean_scale,
                )
                node_log_cdf.append(np.log(weight) + log_cdf)
            state_score = logsumexp(node_log_cdf, axis=0)
            state_score -= 0.5 * np.log(np.pi)
            state_log_cdf.append(state_score)
        expected = logsumexp(state_log_cdf, axis=0)
        expected -= np.log(attacker.n_gibbs_samples)

        actual = attacker.run_attack(target_sigs).numpy()

        self.assertIsNone(attacker.eta_in)
        self.assertIsNone(attacker.Ms_in)
        self.assertIsNone(attacker.Vs_in)
        self.assertTrue(np.allclose(actual, expected, atol=1e-6))


    def test_composite_base_sums_raw_sample_evidence(self):
        '''Verify CompositeBASE disables sigmoid and sums raw BASE scores.'''
        shadow_loss_sigs = torch.tensor([
            [[1.0, 1.2], [1.5, 1.7], [2.0, 2.2], [2.5, 2.7]],
            [[1.2, 1.4], [1.3, 1.5], [2.2, 2.4], [2.3, 2.5]],
            [[0.9, 1.1], [1.6, 1.8], [1.9, 2.1], [2.6, 2.8]],
            [[1.1, 1.3], [1.4, 1.6], [2.1, 2.3], [2.4, 2.6]],
        ])
        shadow_train_mask = torch.tensor([
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
        ])
        target_loss_sigs = torch.tensor([
            [0.8, 1.0],
            [1.2, 1.4],
            [1.8, 2.0],
            [2.2, 2.4],
        ])
        audit_table = {0: [0, 1], 1: [2, 3]}
        attack_config = SimpleNamespace(offline=True, prior=0.5)
        attacker = attacks_entity.CompositeBASE(
            attack_config=attack_config,
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_train_mask=shadow_train_mask,
        )

        sample_scores = attacker.base_attack.run_attack(target_loss_sigs)
        scores = attacker.run_attack(audit_table, target_loss_sigs)

        self.assertFalse(attacker.base_attack.apply_sigmoid)
        self.assertEqual(scores[0].dtype, torch.float32)
        self.assertTrue(torch.allclose(scores[0], sample_scores[:2].sum()))
        self.assertTrue(torch.allclose(scores[1], sample_scores[2:].sum()))

    def test_lira_default_mean_aggregation_matches_manual_mean(self):
        shadow_loss_sigs = torch.tensor([
            [[0.8, 1.0], [1.0, 1.2]],
            [[1.2, 1.4], [0.8, 1.0]],
            [[0.9, 1.1], [1.1, 1.3]],
            [[1.3, 1.5], [0.9, 1.1]],
        ], dtype=torch.float32)
        shadow_train_mask = torch.tensor([
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ], dtype=torch.bool)
        target_loss_sigs = torch.tensor([
            [0.9, 1.1],
            [1.0, 1.2],
        ], dtype=torch.float32)

        raw_attacker = attacks_sample.LiRA(
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_train_mask=shadow_train_mask,
            offline=False,
            loss_transformation="none",
        )
        mean_attacker = attacks_sample.LiRA(
            shadow_loss_sigs=shadow_loss_sigs.mean(dim=2, keepdim=True),
            shadow_train_mask=shadow_train_mask,
            offline=False,
            loss_transformation="none",
        )

        raw_score = raw_attacker.run_attack(target_loss_sigs)
        mean_score = mean_attacker.run_attack(target_loss_sigs.mean(dim=1, keepdim=True))

        self.assertTrue(torch.allclose(raw_score, mean_score, atol=1e-6))

    def test_composite_lira_online_uses_constructor_audit_table(self):
        attacker = attacks_entity.CompositeLiRA(
            audit_table={0: [0], 1: [2]},
            shadow_loss_sigs=torch.tensor([
                [10.0, 1.0, 5.0, 6.0],
                [2.0, 20.0, 7.0, 8.0],
                [3.0, 4.0, 30.0, 9.0],
                [4.0, 5.0, 40.0, 10.0],
            ], dtype=torch.float32),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ], dtype=torch.bool),
            offline=False,
        )

        target_loss_sigs = torch.tensor([9.0, 0.0, 31.0, 0.0], dtype=torch.float32)
        score = attacker.run_attack(target_loss_sigs)

        self.assertEqual(set(score.keys()), {0, 1})
        self.assertTrue(torch.allclose(attacker.mean_in[0], torch.tensor([-6.0], dtype=torch.float32)))
        self.assertTrue(torch.allclose(attacker.mean_out[0], torch.tensor([-3.5], dtype=torch.float32)))
        self.assertTrue(torch.isfinite(score[0]))
        self.assertTrue(torch.isfinite(score[1]))

    def test_composite_lira_online_scores_selected_indices(self):
        audit_table = {
            0: [0, 1],
            1: [2, 3],
        }
        attacker = attacks_entity.CompositeLiRA(
            audit_table=audit_table,
            shadow_loss_sigs=torch.tensor([
                [1.0, 1.2, 5.0, 5.2],
                [1.1, 1.3, 5.1, 5.3],
                [3.0, 3.2, 0.8, 1.0],
                [3.1, 3.3, 0.9, 1.1],
            ], dtype=torch.float32),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ], dtype=torch.bool),
            offline=False,
        )

        member_like_target = torch.tensor([1.05, 1.25, 5.0, 5.2], dtype=torch.float32)
        non_member_like_target = torch.tensor([3.05, 3.25, 0.85, 1.05], dtype=torch.float32)
        member_like_score = attacker.run_attack(member_like_target)
        non_member_like_score = attacker.run_attack(non_member_like_target)

        self.assertGreater(float(member_like_score[0]), float(non_member_like_score[0]))
        self.assertLess(float(member_like_score[1]), float(non_member_like_score[1]))

    def test_composite_lira_offline_uses_out_tail_score(self):
        audit_table = {
            0: [0],
            1: [2],
        }
        attacker = attacks_entity.CompositeLiRA(
            audit_table=audit_table,
            shadow_loss_sigs=torch.tensor([
                [10.0, 1.0, 5.0, 6.0],
                [2.0, 20.0, 7.0, 8.0],
                [3.0, 4.0, 30.0, 9.0],
                [4.0, 5.0, 40.0, 10.0],
            ], dtype=torch.float32),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ], dtype=torch.bool),
            offline=True,
        )

        target_loss_sigs = torch.tensor([1.0, 0.0, 10.0, 0.0], dtype=torch.float32)
        score = attacker.run_attack(target_loss_sigs)
        expected_score_entity_0 = -norm.logsf(
            np.array([-1.0], dtype=np.float32),
            loc=attacker.mean_out[0][0].cpu().numpy(),
            scale=np.sqrt(float(attacker.cov_out)),
        ).sum()

        self.assertAlmostEqual(float(score[0]), float(expected_score_entity_0), places=6)
        self.assertGreater(float(score[0]), float(score[1]))

    def test_composite_lira_uses_audit_table_for_reference_indices(self):
        audit_table = {
            0: [2, 3],
            1: [6, 7],
        }
        attacker = attacks_entity.CompositeLiRA(
            audit_table=audit_table,
            shadow_loss_sigs=torch.tensor([
                [10.0, 11.0, 1.0, 2.0, 20.0, 21.0, 3.0, 4.0],
                [12.0, 13.0, 5.0, 6.0, 22.0, 23.0, 7.0, 8.0],
            ], dtype=torch.float32),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [0, 1],
            ], dtype=torch.bool),
            offline=False,
        )

        self.assertTrue(torch.allclose(attacker.mean_in[0], torch.tensor([-1.0, -2.0], dtype=torch.float32)))
        self.assertTrue(torch.allclose(attacker.mean_out[0], torch.tensor([-5.0, -6.0], dtype=torch.float32)))

    def test_composite_lira_run_attack_returns_scores_for_audit_table(self):
        audit_table = {
            0: [2, 3],
            1: [6, 7],
        }
        attacker = attacks_entity.CompositeLiRA(
            audit_table=audit_table,
            shadow_loss_sigs=torch.tensor([
                [10.0, 11.0, 1.0, 2.0, 20.0, 21.0, 3.0, 4.0],
                [12.0, 13.0, 5.0, 6.0, 22.0, 23.0, 7.0, 8.0],
            ], dtype=torch.float32),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [0, 1],
            ], dtype=torch.bool),
            offline=False,
        )
        target_loss_sigs = torch.tensor([9.0, 9.5, 1.0, 2.0, 19.0, 19.5, 3.0, 4.0], dtype=torch.float32)
        score = attacker.run_attack(target_loss_sigs)

        self.assertEqual(set(score.keys()), {0, 1})
        self.assertTrue(torch.isfinite(score[0]))
        self.assertTrue(torch.isfinite(score[1]))


    def test_composite_lira_full_covariance_uses_out_reference_values(self):
        audit_table = {
            0: [0, 1],
            1: [2, 3],
        }
        shadow_loss_sigs = torch.tensor([
            [1.0, 1.2, 8.0, 8.3],
            [1.1, 1.4, 8.2, 8.6],
            [4.0, 4.8, 2.0, 2.1],
            [4.5, 5.7, 2.2, 2.4],
        ], dtype=torch.float64)
        shadow_entity_mask = torch.tensor([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ], dtype=torch.bool)
        attacker = attacks_entity.CompositeLiRAv2(
            audit_table=audit_table,
            shadow_loss_sigs=shadow_loss_sigs.unsqueeze(-1),
            shadow_entity_mask=shadow_entity_mask,
            offline=False,
            use_full_cov=True,
            loss_transformation="none",
            covariance_shrinkage=0.1,
        )
        phi_shadow = -shadow_loss_sigs
        out_entity_0 = phi_shadow[~shadow_entity_mask[:, 0]][:, audit_table[0]]
        out_entity_1 = phi_shadow[~shadow_entity_mask[:, 1]][:, audit_table[1]]
        residual_0 = out_entity_0 - out_entity_0.mean(dim=0, keepdim=True)
        residual_1 = out_entity_1 - out_entity_1.mean(dim=0, keepdim=True)
        expected_out = torch.cat([residual_0, residual_1], dim=0)
        degrees_of_freedom = len(out_entity_0) + len(out_entity_1) - 2
        expected_cov_out = (expected_out.T @ expected_out) / degrees_of_freedom
        diagonal_target = torch.diag(expected_cov_out.diag())
        expected_cov_out = 0.9 * expected_cov_out + 0.1 * diagonal_target
        expected_cov_out += 1e-9 * torch.eye(2, dtype=torch.float64)

        self.assertTrue(torch.allclose(attacker.cov_out, expected_cov_out, atol=1e-6))

    def test_composite_lira_v2_covariance_modes_center_within_entities(self):
        audit_table = {0: [0, 1], 1: [2, 3]}
        shadow_loss_sigs = torch.tensor([
            [1.0, 2.0, 100.0, 200.0],
            [2.0, 4.0, 100.0, 200.0],
            [3.0, 6.0, 100.0, 200.0],
            [20.0, 30.0, 10.0, 15.0],
            [22.0, 34.0, 12.0, 18.0],
            [24.0, 38.0, 14.0, 21.0],
        ], dtype=torch.float64)
        shadow_entity_mask = torch.tensor([
            [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0],
        ], dtype=torch.bool)
        global_empirical = torch.tensor([[2.5, 4.0], [4.0, 6.5]], dtype=torch.float64)
        local_empirical = {
            0: torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64),
            1: torch.tensor([[4.0, 6.0], [6.0, 9.0]], dtype=torch.float64),
        }
        for use_global_dispersion in (True, False):
            for covariance in ("spherical", "diagonal", "low_rank", "full"):
                with self.subTest(global_dispersion=use_global_dispersion, covariance=covariance):
                    attacker = attacks_entity.CompositeLiRAv2(
                        audit_table=audit_table,
                        shadow_loss_sigs=shadow_loss_sigs.unsqueeze(-1),
                        shadow_entity_mask=shadow_entity_mask,
                        offline=True,
                        covariance=covariance,
                        use_global_dispersion=use_global_dispersion,
                        covariance_rank=1,
                        covariance_shrinkage=0.1,
                        n_qmc_samples=8,
                    )
                    empirical = global_empirical if use_global_dispersion else local_empirical[0]
                    if covariance == "spherical":
                        expected = empirical.trace() / 2
                        actual = attacker.var_out if use_global_dispersion else attacker.var_out[0]
                    elif covariance == "diagonal":
                        expected = empirical.diag()
                        actual = attacker.var_out if use_global_dispersion else attacker.var_out[0]
                    else:
                        diagonal = torch.diag(empirical.diag())
                        if covariance == "full":
                            expected = 0.9 * empirical + 0.1 * diagonal
                            expected += 1e-9 * torch.eye(2, dtype=torch.float64)
                        else:
                            eigenvalues, eigenvectors = torch.linalg.eigh(empirical)
                            leading_factor = eigenvectors[:, -1] * eigenvalues[-1].sqrt()
                            factor_covariance = 0.9 * torch.outer(leading_factor, leading_factor)
                            residual_variance = empirical.diag() - factor_covariance.diag()
                            expected = factor_covariance + torch.diag(residual_variance.clamp_min(1e-9))
                        actual = attacker.cov_out if use_global_dispersion else attacker.cov_out[0]
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-8))
                    if not use_global_dispersion and covariance == "diagonal":
                        self.assertTrue(torch.allclose(attacker.var_out[1], local_empirical[1].diag()))
                    score = attacker.run_attack(shadow_loss_sigs[0])
                    self.assertTrue(torch.isfinite(score[0]))
                    self.assertTrue(torch.isfinite(score[1]))

        default_attacker = attacks_entity.CompositeLiRAv2(
            audit_table=audit_table,
            shadow_loss_sigs=shadow_loss_sigs.unsqueeze(-1),
            shadow_entity_mask=shadow_entity_mask,
            offline=True,
        )
        self.assertEqual(default_attacker.covariance, "spherical")
        self.assertAlmostEqual(float(default_attacker.var_out), 4.5)

        online_attacker = attacks_entity.CompositeLiRAv2(
            audit_table=audit_table,
            shadow_loss_sigs=shadow_loss_sigs.unsqueeze(-1),
            shadow_entity_mask=shadow_entity_mask,
            offline=False,
            covariance="full",
        )
        online_scores = online_attacker.run_attack(shadow_loss_sigs[0])
        self.assertTrue(torch.isfinite(online_scores[0]))
        self.assertTrue(torch.isfinite(online_scores[1]))

    def test_composite_lira_v2_global_spherical_allows_unequal_probe_counts(self):
        audit_table = {0: [0], 1: [1, 2]}
        shadow_loss_sigs = torch.tensor([
            [1.0, 10.0, 20.0],
            [2.0, 11.0, 21.0],
            [3.0, 12.0, 22.0],
        ], dtype=torch.float64)
        shadow_entity_mask = torch.zeros((3, 2), dtype=torch.bool)
        attacker = attacks_entity.CompositeLiRAv2(
            audit_table=audit_table,
            shadow_loss_sigs=shadow_loss_sigs.unsqueeze(-1),
            shadow_entity_mask=shadow_entity_mask,
            offline=True,
            covariance="spherical",
        )

        self.assertAlmostEqual(float(attacker.var_out), 1.0)
        self.assertEqual(len(attacker.run_attack(shadow_loss_sigs[0])), 2)
        with self.assertRaisesRegex(ValueError, "equal probe counts"):
            attacks_entity.CompositeLiRAv2(
                audit_table=audit_table,
                shadow_loss_sigs=shadow_loss_sigs.unsqueeze(-1),
                shadow_entity_mask=shadow_entity_mask,
                offline=True,
                covariance="diagonal",
            )

    def test_composite_lira_v2_log_transforms_references_and_target(self):
        audit_table = {0: [0], 1: [1]}
        shadow_loss_sigs = torch.tensor([
            [[1.0, 4.0], [2.0, 8.0]],
            [[2.0, 8.0], [1.0, 4.0]],
            [[4.0, 16.0], [2.0, 8.0]],
            [[8.0, 32.0], [4.0, 16.0]],
        ], dtype=torch.float64)
        shadow_entity_mask = torch.tensor([
            [1, 0], [1, 0], [0, 1], [0, 1],
        ], dtype=torch.bool)
        attacker = attacks_entity.CompositeLiRAv2(
            audit_table=audit_table,
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_entity_mask=shadow_entity_mask,
            offline=True,
            covariance="spherical",
            loss_transformation="log",
        )
        target_loss_sigs = torch.tensor([[2.0, 8.0], [4.0, 16.0]], dtype=torch.float64)
        transformed_target = -torch.log(target_loss_sigs).mean(dim=-1)
        score = attacker.run_attack(target_loss_sigs)
        expected_score = -norm.logsf(
            transformed_target[0].numpy(),
            loc=attacker.mean_out[0].item(),
            scale=attacker.var_out.sqrt().item(),
        )

        self.assertAlmostEqual(float(score[0]), float(expected_score), places=6)


if __name__ == "__main__":
    unittest.main()
