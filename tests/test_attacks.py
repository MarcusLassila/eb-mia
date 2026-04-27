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

    def test_composite_lira_online_uses_shadow_train_mask_for_exclude_train(self):
        attacker = attacks.CompositeLiRA(
            entity_index_table={
                0: [0, 1],
                1: [2, 3],
            },
            shadow_loss_sigs=torch.tensor([
                [10.0, 1.0, 5.0, 6.0],
                [2.0, 20.0, 7.0, 8.0],
                [3.0, 4.0, 30.0, 9.0],
            ], dtype=torch.float32),
            shadow_train_mask=torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=torch.bool),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [1, 0],
                [0, 1],
            ], dtype=torch.bool),
            mode="exclude_train",
            min_samples_per_entity=1,
            max_samples_per_entity=1,
            offline=False,
        )

        expected_mean_in_entity_0 = torch.tensor([-1.5], dtype=torch.float32)
        possible_mean_out_entity_0 = {
            -3.0,
            -4.0,
        }

        self.assertTrue(torch.allclose(attacker.mean_in[1][0], expected_mean_in_entity_0, atol=1e-6))
        self.assertIn(float(attacker.mean_out[1][0]), possible_mean_out_entity_0)

    def test_composite_lira_online_falls_back_to_global_in_mean(self):
        audit_table = {
            0: [0],
            1: [2],
            2: [4],
        }
        attacker = attacks.CompositeLiRA(
            entity_index_table={
                0: [0, 1],
                1: [2, 3],
                2: [4, 5],
            },
            shadow_loss_sigs=torch.tensor([
                [10.0, 1.0, 5.0, 6.0, 11.0, 12.0],
                [2.0, 20.0, 7.0, 8.0, 13.0, 14.0],
                [3.0, 4.0, 30.0, 9.0, 15.0, 16.0],
            ], dtype=torch.float32),
            shadow_train_mask=torch.tensor([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ], dtype=torch.bool),
            shadow_entity_mask=torch.tensor([
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ], dtype=torch.bool),
            mode="exclude_train",
            min_samples_per_entity=1,
            max_samples_per_entity=1,
            offline=False,
        )

        expected_global_in_mean = torch.tensor([-4.0], dtype=torch.float32)
        target_loss_sigs = torch.tensor([2.5, 0.0, 10.0, 0.0, 12.0, 0.0], dtype=torch.float32)
        score = attacker.run_attack(audit_table=audit_table, target_loss_sigs=target_loss_sigs)

        self.assertTrue(torch.allclose(attacker.mean_in[1][2], expected_global_in_mean, atol=1e-6))
        self.assertTrue(torch.isfinite(score[2]))

    def test_composite_lira_offline_uses_out_tail_score(self):
        audit_table = {
            0: [0],
            1: [2],
        }
        attacker = attacks.CompositeLiRA(
            entity_index_table={
                0: [0, 1],
                1: [2, 3],
            },
            shadow_loss_sigs=torch.tensor([
                [10.0, 1.0, 5.0, 6.0],
                [2.0, 20.0, 7.0, 8.0],
                [3.0, 4.0, 30.0, 9.0],
            ], dtype=torch.float32),
            shadow_train_mask=torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=torch.bool),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [1, 0],
                [0, 1],
            ], dtype=torch.bool),
            mode="exclude_train",
            min_samples_per_entity=1,
            max_samples_per_entity=1,
            offline=True,
        )

        target_loss_sigs = torch.tensor([1.0, 0.0, 10.0, 0.0], dtype=torch.float32)
        score = attacker.run_attack(audit_table=audit_table, target_loss_sigs=target_loss_sigs)
        expected_score_entity_0 = -norm.logsf(
            np.array([-1.0], dtype=np.float32),
            loc=attacker.mean_out[1][0].cpu().numpy(),
            scale=np.sqrt(float(attacker.var_out[1])),
        ).sum()

        self.assertAlmostEqual(float(score[0]), float(expected_score_entity_0), places=6)
        self.assertGreater(float(score[0]), float(score[1]))

    def test_composite_lira_hold_out_uses_unseen_tail_samples(self):
        attacker = attacks.CompositeLiRA(
            entity_index_table={
                0: [0, 1, 2, 3],
                1: [4, 5, 6, 7],
            },
            shadow_loss_sigs=torch.tensor([
                [10.0, 11.0, 1.0, 2.0, 20.0, 21.0, 3.0, 4.0],
                [12.0, 13.0, 5.0, 6.0, 22.0, 23.0, 7.0, 8.0],
            ], dtype=torch.float32),
            shadow_train_mask=torch.tensor([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
            ], dtype=torch.bool),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [0, 1],
            ], dtype=torch.bool),
            mode="hold_out",
            min_samples_per_entity=2,
            max_samples_per_entity=2,
            hold_out_frac=0.5,
            offline=False,
        )

        self.assertTrue(torch.allclose(attacker.mean_in[2][0], torch.tensor([-1.0, -2.0], dtype=torch.float32)))
        self.assertTrue(torch.allclose(attacker.mean_out[2][0], torch.tensor([-5.0, -6.0], dtype=torch.float32)))

    def test_composite_lira_run_attack_uses_audit_table(self):
        attacker = attacks.CompositeLiRA(
            entity_index_table={
                0: [0, 1, 2, 3],
                1: [4, 5, 6, 7],
            },
            shadow_loss_sigs=torch.tensor([
                [10.0, 11.0, 1.0, 2.0, 20.0, 21.0, 3.0, 4.0],
                [12.0, 13.0, 5.0, 6.0, 22.0, 23.0, 7.0, 8.0],
            ], dtype=torch.float32),
            shadow_train_mask=torch.tensor([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
            ], dtype=torch.bool),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [0, 1],
            ], dtype=torch.bool),
            mode="hold_out",
            min_samples_per_entity=2,
            max_samples_per_entity=2,
            hold_out_frac=0.5,
            offline=False,
        )
        target_loss_sigs = torch.tensor([9.0, 9.5, 1.0, 2.0, 19.0, 19.5, 3.0, 4.0], dtype=torch.float32)
        score = attacker.run_attack(
            audit_table={
                0: [2, 3],
                1: [6, 7],
            },
            target_loss_sigs=target_loss_sigs,
        )

        self.assertEqual(set(score.keys()), {0, 1})
        self.assertTrue(torch.isfinite(score[0]))
        self.assertTrue(torch.isfinite(score[1]))


if __name__ == "__main__":
    unittest.main()
