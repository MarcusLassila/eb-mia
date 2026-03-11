import pickle
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

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


class TestCompositeLiRAEndToEnd(unittest.TestCase):
    def test_run_entity_audit_composes_lira_scores_and_saves_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1-s0-sz64-epoch10.pth")
            attack = "LiRA-online-none"
            scores_path = path_utils.scores_dir(tmpdir, attack, target_path) / path_utils.scores_pickle_name(target_path, attack)
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scores_path, "wb") as file:
                pickle.dump(
                    {
                        "scores": [0.0, 0.0, 0.0, 0.0],
                        "train_mask": [1, 0, 0, 0],
                        "loss_sigs": [
                            [0.90, 0.95, 1.40, 1.45],
                            [1.00, 1.10, 1.20, 1.30],
                            [1.40, 1.50, 0.80, 0.90],
                            [1.05, 1.15, 1.25, 1.35],
                            [1.45, 1.55, 0.85, 0.95],
                        ],
                        "shadow_train_mask": [
                            [1, 1, 0, 0],
                            [0, 0, 1, 1],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0],
                        ],
                    },
                    file,
                )
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "all",
                "res_dir": tmpdir,
                "score_paths": [str(scores_path)],
            })
            dataset = DummyEntityDataset([0, 0, 1, 1])
            captured = {}

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 1.0, "TPR@1%FPR": 1.0, "TPR@0.1%FPR": 1.0, "n_audit_points": 2}

            with (
                patch.object(run_audit_module, "EntityDataset", DummyEntityDataset),
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            self.assertEqual(captured["score"].shape, (2,))
            self.assertGreater(float(captured["score"][0]), float(captured["score"][1]))

            metrics_path = (
                path_utils.metrics_dir(tmpdir, scores_path, "entity", entity_audit_mode="all")
                / path_utils.metrics_pickle_name(scores_path, "entity")
            )
            self.assertTrue(metrics_path.exists())


if __name__ == "__main__":
    unittest.main()
