import pickle
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch, call

import torch

from mia import path_utils
from mia import run_audit as run_audit_module


class DummyEntityDataset:

    def __init__(self, entity_ids):
        self.entity_ids = torch.tensor(entity_ids, dtype=torch.long)

    def get_entity_index_table(self):
        table = defaultdict(list)
        for idx, entity_id in enumerate(self.entity_ids.tolist()):
            table[entity_id].append(idx)
        return table

    def __getitem__(self, index):
        return int(index)

    def __len__(self):
        return len(self.entity_ids)


class TestRunAuditMetrics(unittest.TestCase):
    def test_metrics_pickle_name_entity_includes_exact_min_max(self):
        filename = path_utils.metrics_pickle_name(
            "/tmp/scores_attack-BASE_target-VAE-celeba-ent-f0p5-p1-s0-sz64-epoch10.pkl",
            "entity",
            n_audit_samples_per_entity=2,
            min_samples_per_entity=1,
            max_samples_per_entity=3,
        )
        self.assertEqual(
            filename,
            "metrics_attack-BASE_target-VAE-celeba-ent-f0p5-p1-s0-sz64-epoch10_mode-entity_n-2_min-1_max-3.pkl",
        )

    def test_run_sample_audit_infers_image_size_and_saves_audit_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth")
            attack = "TESTATTACK"
            scores_path = path_utils.scores_dir(tmpdir, attack, target_path) / path_utils.scores_pickle_name(target_path, attack)
            audit_config = {
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "n_audit_samples": 2,
                "audit_mode": "sample",
                "round_robin": False,
                "res_dir": tmpdir,
                "score_paths": [str(scores_path)],
            }
            config = run_audit_module.utils.Config(dict(audit_config))
            dataset = list(range(8))
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scores_path, "wb") as file:
                pickle.dump(
                    {"scores": [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6], "train_mask": [1, 1, 1, 1, 0, 0, 0, 0]},
                    file,
                )

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.tensor([0, 4])),
                patch.object(
                    run_audit_module.evaluation,
                    "evaluate_MIA",
                    return_value={"AUC": 0.5, "TPR@1%FPR": 0.25, "TPR@0.1%FPR": 0.1},
                ) as eval_fn,
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
                patch("builtins.print") as print_fn,
            ):
                run_audit_module.run_sample_audit(config=config)

            load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
            eval_kwargs = eval_fn.call_args.kwargs
            self.assertTrue(torch.equal(eval_kwargs["score"], torch.tensor([0.1, 0.9], dtype=torch.float32)))
            self.assertTrue(torch.equal(eval_kwargs["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            metrics_dir = path_utils.metrics_dir(tmpdir, scores_path, "sample")
            metrics_files = sorted(metrics_dir.glob("*.pkl"))
            metrics_files = [path for path in metrics_files if path.name.startswith("metrics_")]
            self.assertEqual(len(metrics_files), 1)
            self.assertEqual(
                metrics_files[0].name,
                "metrics_attack-TESTATTACK_target-DDPM-cifar10-rand-f0p5-s3-sz32-epoch4_mode-sample.pkl",
            )
            with open(metrics_files[0], "rb") as file:
                metrics = pickle.load(file)
            self.assertEqual(metrics["AUC"], 0.5)
            self.assertEqual(metrics["TPR@1%FPR"], 0.25)
            self.assertEqual(metrics["TPR@0.1%FPR"], 0.1)
            self.assertIn("audit_config", metrics)
            self.assertIsInstance(metrics["audit_config"], dict)
            print_fn.assert_has_calls([
                call(""),
                call("Audit summary (TESTATTACK)"),
                call(f"{'Metric':<16} {'Mean':>10}"),
                call(f"{'-' * 16} {'-' * 10}"),
                call(f"{'AUC':<16} {0.5:>10.4f}"),
                call(f"{'TPR@1%FPR':<16} {0.25:>10.4f}"),
                call(f"{'TPR@0.1%FPR':<16} {0.1:>10.4f}"),
            ])

    def test_run_entity_audit_composes_base_scores_and_saves_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1-s0-sz64-epoch10.pth")
            scores_path = path_utils.scores_dir(tmpdir, "BASE", target_path) / path_utils.scores_pickle_name(target_path, "BASE")
            audit_config = {
                "dataset": "celeba",
                "data_dir": tmpdir,
                "batch_size": 2,
                "audit_mode": "entity",
                "mode": "max_one_train_sample",
                "n_audit_samples_per_entity": 2,
                "min_samples_per_entity": 1,
                "max_samples_per_entity": 2,
                "round_robin": False,
                "res_dir": tmpdir,
                "score_paths": [str(scores_path)],
            }
            config = run_audit_module.utils.Config(dict(audit_config))
            dataset = DummyEntityDataset([0, 0, 1, 1, 2, 2, 3, 3])
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scores_path, "wb") as file:
                pickle.dump(
                    {"scores": [0.2, 0.5, 0.1, 0.1, 0.1, 0.1, 0.3, 0.4], "train_mask": [1, 0, 1, 0, 1, 0, 0, 0]},
                    file,
                )

            captured = {}

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 0.6, "TPR@1%FPR": 0.3, "TPR@0.1%FPR": 0.2}

            with (
                patch.object(run_audit_module, "EntityDataset", DummyEntityDataset),
                patch.object(run_audit_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch("builtins.print") as print_fn,
            ):
                run_audit_module.run_entity_audit(config=config)

            load_dataset_fn.assert_called_once_with("celeba", data_dir=tmpdir, size=64)
            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            expected_scores = torch.tensor([0.6, 0.58], dtype=torch.float32)
            self.assertTrue(torch.allclose(captured["score"], expected_scores, atol=1e-6))
            print_fn.assert_any_call("Audit summary (CompositeBASE)")

            metrics_path = (
                path_utils.metrics_dir(tmpdir, scores_path, "entity", entity_audit_mode="max_one_train_sample")
                / "metrics_attack-BASE_target-DDPM-celeba-ent-f0p5-p1-s0-sz64-epoch10_mode-entity_n-2_min-1_max-2.pkl"
            )
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "rb") as file:
                metrics = pickle.load(file)
            self.assertEqual(metrics["AUC"], 0.6)
            self.assertIn("audit_config", metrics)
            self.assertIsInstance(metrics["audit_config"], dict)

    def test_run_entity_audit_rejects_all_mode_with_n_audit_samples_per_entity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1-s0-sz64-epoch10.pth")
            scores_path = path_utils.scores_dir(tmpdir, "BASE", target_path) / path_utils.scores_pickle_name(target_path, "BASE")
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scores_path, "wb") as file:
                pickle.dump(
                    {"scores": [0.2, 0.5, 0.1, 0.1], "train_mask": [1, 0, 1, 0]},
                    file,
                )
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "all",
                "n_audit_samples_per_entity": 2,
                "res_dir": tmpdir,
                "score_paths": [str(scores_path)],
            })
            with self.assertRaisesRegex(ValueError, "mode='all'.*n_audit_samples_per_entity"):
                run_audit_module.run_entity_audit(config=config)


if __name__ == "__main__":
    unittest.main()
