import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import path_utils
from mia import run_mia as run_mia_module


class DummyAttacker:

    def __init__(self):
        self.calls = []

    def run_attack(self, audit_samples, target_path):
        self.calls.append((audit_samples, Path(target_path)))
        return torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)


class TestRunMia(unittest.TestCase):
    def test_run_mia_saves_sample_scores_for_full_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth")
            shadow_path = str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s4-sz32-epoch4.pth")
            config_dict = {
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "round_robin": False,
                "res_dir": tmpdir,
                "target_model_paths": [target_path],
                "shadow_model_paths": [shadow_path],
                "attack": {"attack": "BASE", "prior": 0.5, "n_loss_samples": 1},
            }
            config = run_mia_module.utils.Config(dict(config_dict))
            dataset = list(range(4))
            attacker = DummyAttacker()

            with (
                patch.object(run_mia_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(run_mia_module, "get_attacker", return_value=attacker) as get_attacker_fn,
                patch.object(run_mia_module.utils, "get_train_indices", return_value=torch.tensor([1, 3], dtype=torch.long)),
            ):
                run_mia_module.run_mia(config=config, device=torch.device("cpu"))

            load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
            get_attacker_fn.assert_called_once()
            self.assertEqual(len(attacker.calls), 1)
            self.assertIs(attacker.calls[0][0], dataset)
            self.assertEqual(attacker.calls[0][1], Path(target_path))

            scores_path = path_utils.scores_dir(tmpdir, "BASE", target_path) / "scores_attack-BASE_target-DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pkl"
            self.assertTrue(scores_path.exists())
            with open(scores_path, "rb") as file:
                scores = pickle.load(file)
            self.assertEqual(sorted(scores.keys()), ["scores", "train_mask"])
            self.assertEqual(len(scores["scores"]), 4)
            self.assertEqual(len(scores["train_mask"]), 4)
            for score, expected in zip(scores["scores"], [0.1, 0.2, 0.3, 0.4]):
                self.assertAlmostEqual(score, expected, places=6)
            self.assertEqual(scores["train_mask"], [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
