import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
