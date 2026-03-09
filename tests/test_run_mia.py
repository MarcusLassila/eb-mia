import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import path_utils
from mia import run_audit as run_audit_module
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
                "attack": {"name": "base-prior-0p5", "attack": "BASE", "offline": True, "prior": 0.5, "n_loss_samples": 1},
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
            get_attacker_kwargs = get_attacker_fn.call_args.kwargs
            self.assertEqual(get_attacker_kwargs["len_dataset"], len(dataset))
            self.assertEqual(len(attacker.calls), 1)
            self.assertIs(attacker.calls[0][0], dataset)
            self.assertEqual(attacker.calls[0][1], Path(target_path))

            scores_path = path_utils.scores_dir(tmpdir, "base-prior-0p5", target_path) / "scores_attack-base-prior-0p5_target-DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pkl"
            self.assertTrue(scores_path.exists())
            with open(scores_path, "rb") as file:
                scores = pickle.load(file)
            self.assertEqual(sorted(scores.keys()), ["scores", "train_mask"])
            self.assertEqual(len(scores["scores"]), 4)
            self.assertEqual(len(scores["train_mask"]), 4)
            for score, expected in zip(scores["scores"], [0.1, 0.2, 0.3, 0.4]):
                self.assertAlmostEqual(score, expected, places=6)
            self.assertEqual(scores["train_mask"], [0, 1, 0, 1])

    def test_lira_run_mia_scores_support_sample_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth")
            shadow_path = str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s4-sz32-epoch4.pth")
            mia_config = run_mia_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "round_robin": False,
                "res_dir": tmpdir,
                "target_model_paths": [target_path],
                "shadow_model_paths": [shadow_path],
                "attack": {
                    "name": "LiRA-nll-global",
                    "attack": "LiRA",
                    "offline": True,
                    "n_loss_samples": 1,
                    "use_global_var": True,
                    "loss_transformation": "nll",
                },
            })
            dataset = list(range(4))
            lira_scores = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)

            with (
                patch.object(run_mia_module, "load_dataset", return_value=dataset) as mia_load_dataset_fn,
                patch.object(run_mia_module.utils, "get_train_indices", return_value=torch.tensor([0, 2], dtype=torch.long)),
                patch.object(run_mia_module.attacks.LiRA, "run_attack", return_value=lira_scores) as run_attack_fn,
            ):
                run_mia_module.run_mia(config=mia_config, device=torch.device("cpu"))

            mia_load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
            self.assertEqual(run_attack_fn.call_count, 1)
            run_attack_args = run_attack_fn.call_args.args
            self.assertIs(run_attack_args[0], dataset)
            self.assertEqual(run_attack_args[1], Path(target_path))

            scores_path = path_utils.scores_dir(tmpdir, "LiRA-nll-global", target_path) / path_utils.scores_pickle_name(target_path, "LiRA-nll-global")
            self.assertTrue(scores_path.exists())

            audit_config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "res_dir": tmpdir,
                "score_paths": [str(scores_path)],
            })

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset) as audit_load_dataset_fn,
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.tensor([0, 1, 2, 3], dtype=torch.long)),
                patch.object(
                    run_audit_module.evaluation,
                    "evaluate_MIA",
                    return_value={"AUC": 0.5, "TPR@1%FPR": 0.25, "TPR@0.1%FPR": 0.1, "n_audit_points": 4},
                ) as eval_fn,
            ):
                run_audit_module.run_sample_audit(config=audit_config)

            audit_load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
            eval_kwargs = eval_fn.call_args.kwargs
            self.assertTrue(torch.equal(eval_kwargs["score"], lira_scores))
            self.assertTrue(torch.equal(eval_kwargs["ground_truth"], torch.tensor([1, 0, 1, 0], dtype=torch.long)))

            metrics_path = (
                path_utils.metrics_dir(tmpdir, scores_path, "sample")
                / path_utils.metrics_pickle_name(scores_path, "sample")
            )
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "rb") as file:
                metrics = pickle.load(file)
            self.assertEqual(metrics["n_audit_points"], 4)
            self.assertIn("audit_config", metrics)
            self.assertIsInstance(metrics["audit_config"], dict)

    def test_get_attacker_passes_lira_config_parameters(self):
        sentinel = object()
        with patch.object(run_mia_module.attacks, "LiRA", return_value=sentinel) as lira_cls:
            attack_config = run_mia_module.utils.Config({
                "name": "LiRA-custom",
                "attack": "LiRA",
                "offline": False,
                "n_loss_samples": 3,
                "use_global_var": False,
                "loss_transformation": "none",
            })
            attacker = run_mia_module.get_attacker(
                attack_config=attack_config,
                batch_size=8,
                device=torch.device("cpu"),
                shadow_model_paths=[Path("/tmp/shadow.pth")],
                len_dataset=17,
            )

        self.assertIs(attacker, sentinel)
        lira_cls.assert_called_once_with(
            batch_size=8,
            device=torch.device("cpu"),
            shadow_model_paths=[Path("/tmp/shadow.pth")],
            len_dataset=17,
            offline=False,
            n_loss_samples=3,
            use_global_var=False,
            loss_transformation="none",
        )

    def test_run_mia_requires_attack_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = run_mia_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "round_robin": False,
                "res_dir": tmpdir,
                "target_model_paths": [str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth")],
                "shadow_model_paths": [str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s4-sz32-epoch4.pth")],
                "attack": {"attack": "BASE", "offline": True, "prior": 0.5, "n_loss_samples": 1},
            })
            with self.assertRaisesRegex(ValueError, "define 'name'"):
                run_mia_module.run_mia(config=config, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
