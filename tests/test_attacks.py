import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import attacks


class TestLiRA(unittest.TestCase):
    def test_run_attack_prepends_target_loss_signal_to_shadow_loss_matrix(self):
        attacker = attacks.LiRA(
            batch_size=2,
            device=torch.device("cpu"),
            shadow_model_paths=[Path("/tmp/shadow-0.pth"), Path("/tmp/shadow-1.pth")],
            len_dataset=4,
            offline=True,
            loss_transformation="none",
        )
        audit_samples = list(range(4))
        target_loss_sig = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
        shadow_loss_sigs = torch.tensor([
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
        ], dtype=torch.float32)
        shadow_train_mask = torch.tensor([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.bool)

        with (
            patch.object(
                attacker,
                "query_shadow_models",
                return_value=(
                    None,
                    None,
                    torch.zeros(len(audit_samples), dtype=torch.float32),
                    torch.ones(len(audit_samples), dtype=torch.float32),
                    shadow_loss_sigs,
                    shadow_train_mask,
                ),
            ),
            patch.object(attacker, "load_model", return_value=(object(), torch.tensor([], dtype=torch.long))),
            patch.object(attacker, "loss_signal", return_value=target_loss_sig),
        ):
            result = attacker.run_attack(audit_samples, Path("/tmp/target.pth"))

        self.assertEqual(result["loss_sigs"].shape, (3, 4))
        self.assertTrue(torch.equal(result["loss_sigs"][0], target_loss_sig))
        self.assertTrue(torch.equal(result["loss_sigs"][1:], shadow_loss_sigs))
        self.assertTrue(torch.equal(result["shadow_train_mask"], shadow_train_mask))
        self.assertEqual(result["score"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
