import unittest

import numpy as np
import torch
from scipy.stats import norm

from mia import attacks


class TestAttacks(unittest.TestCase):
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

        score = attacks.BASE(
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_train_mask=shadow_train_mask,
            offline=True,
            prior=0.5,
        ).run_attack(target_loss_sigs)

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

        global_attacker = attacks.NormalBASE(
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_train_mask=shadow_train_mask,
            offline=True,
            prior=0.5,
            use_global_var=True,
        )
        local_attacker = attacks.NormalBASE(
            shadow_loss_sigs=shadow_loss_sigs,
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
        attacker = attacks.LiRA(
            shadow_loss_sigs=torch.tensor([
                [0.9, 1.1, 1.2, 1.4],
                [1.4, 1.5, 0.8, 0.9],
                [1.0, 1.2, 1.3, 1.5],
                [1.5, 1.6, 0.9, 1.0],
            ], dtype=torch.float32),
            shadow_train_mask=torch.tensor([
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
            ], dtype=torch.bool),
            offline=False,
            loss_transformation="none",
        )

        score = attacker.run_attack(torch.tensor([0.95, 1.05, 0.85, 0.95], dtype=torch.float32))

        self.assertEqual(score.shape, (4,))
        self.assertTrue(torch.isfinite(score).all())

    def test_composite_lira_online_uses_constructor_audit_table(self):
        attacker = attacks.CompositeLiRA(
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
        attacker = attacks.CompositeLiRA(
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
        attacker = attacks.CompositeLiRA(
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
            scale=np.sqrt(float(attacker.var_out)),
        ).sum()

        self.assertAlmostEqual(float(score[0]), float(expected_score_entity_0), places=6)
        self.assertGreater(float(score[0]), float(score[1]))

    def test_composite_lira_uses_audit_table_for_reference_indices(self):
        audit_table = {
            0: [2, 3],
            1: [6, 7],
        }
        attacker = attacks.CompositeLiRA(
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
        attacker = attacks.CompositeLiRA(
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


if __name__ == "__main__":
    unittest.main()
