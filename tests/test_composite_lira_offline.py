import unittest

import numpy as np
import torch
from scipy.stats import norm

from mia import attacks


class TestCompositeLiRAOffline(unittest.TestCase):
    def test_offline_diagonal_uses_survival_tail_score(self):
        audit_table = {
            0: [0, 1],
            1: [2, 3],
        }
        attacker = attacks.CompositeLiRA(
            audit_table=audit_table,
            shadow_loss_sigs=torch.tensor([
                [1.0, 2.0, 5.0, 6.0],
                [1.5, 2.5, 7.0, 8.0],
                [3.0, 4.0, 5.5, 6.5],
                [3.5, 4.5, 7.5, 8.5],
            ], dtype=torch.float32),
            shadow_entity_mask=torch.tensor([
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ], dtype=torch.bool),
            offline=True,
        )
        target_loss_sigs = torch.tensor([1.25, 2.25, 5.25, 6.25], dtype=torch.float32)

        score = attacker.run_attack(target_loss_sigs)

        phi_target = -target_loss_sigs
        scale_out = np.sqrt(float(attacker.var_out))
        for entity_id, indices in audit_table.items():
            expected_log_sfs = norm.logsf(
                phi_target[indices].cpu().numpy(),
                loc=attacker.mean_out[entity_id].cpu().numpy(),
                scale=scale_out,
            )
            expected_score = -expected_log_sfs.sum()
            self.assertAlmostEqual(float(score[entity_id]), float(expected_score), places=6)


if __name__ == "__main__":
    unittest.main()
