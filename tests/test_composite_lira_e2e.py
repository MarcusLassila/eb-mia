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

    @property
    def max_entity_id(self):
        return int(self.entity_ids.max().item())

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
    def _write_loss_file(self, res_dir, target_path, loss_sigs, train_mask):
        loss_path = path_utils.loss_signals_dir(res_dir, target_path) / path_utils.loss_signals_pickle_name(target_path, 1)
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_path, "wb") as file:
            pickle.dump({"loss_sigs": loss_sigs, "train_mask": train_mask}, file)
        return loss_path

    def test_run_entity_audit_composes_lira_scores_and_saves_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s0-sz64-epoch10.pth"
            shadow_path_a = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s1-sz64-epoch10.pth"
            shadow_path_b = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s2-sz64-epoch10.pth"
            shadow_path_c = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s3-sz64-epoch10.pth"
            shadow_path_d = Path(tmpdir) / "DDPM-celeba-ent-f0p5-p1p0-s4-sz64-epoch10.pth"
            target_loss_path = self._write_loss_file(tmpdir, target_path, [0.90, 0.95, 1.40, 1.45], [1, 0, 0, 0])
            shadow_loss_paths = [
                self._write_loss_file(tmpdir, shadow_path_a, [1.00, 1.10, 1.20, 1.30], [1, 1, 0, 0]),
                self._write_loss_file(tmpdir, shadow_path_b, [1.40, 1.50, 0.80, 0.90], [0, 0, 1, 1]),
                self._write_loss_file(tmpdir, shadow_path_c, [1.05, 1.15, 1.25, 1.35], [1, 0, 0, 0]),
                self._write_loss_file(tmpdir, shadow_path_d, [1.45, 1.55, 0.85, 0.95], [0, 0, 1, 0]),
            ]
            config = run_audit_module.utils.Config({
                "dataset": "celeba",
                "data_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "all",
                "res_dir": tmpdir,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(path) for path in shadow_loss_paths],
                "attack": {
                    "name": "LiRA-online-none",
                    "attack": "LiRA",
                    "offline": False,
                    "use_global_var": True,
                    "loss_transformation": "none",
                },
            })
            dataset = DummyEntityDataset([10, 10, 20, 20])
            captured = {}

            def fake_evaluate(score, ground_truth):
                captured["score"] = score.clone()
                captured["ground_truth"] = ground_truth.clone()
                return {"AUC": 1.0, "TPR@1%FPR": 1.0, "TPR@0.1%FPR": 1.0, "n_audit_points": 2}

            with (
                patch.object(run_audit_module, "EntityDataset", DummyEntityDataset),
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module.evaluation, "evaluate_MIA", side_effect=fake_evaluate),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=config)

            self.assertTrue(torch.equal(captured["ground_truth"], torch.tensor([1, 0], dtype=torch.long)))
            self.assertEqual(captured["score"].shape, (2,))
            self.assertGreater(float(captured["score"][0]), float(captured["score"][1]))
            metrics_path = (
                path_utils.metrics_dir_from_target(tmpdir, target_path, "LiRA-online-none", "entity", entity_audit_mode="all")
                / path_utils.metrics_pickle_name_from_target(target_path, "LiRA-online-none", "entity")
            )
            self.assertTrue(metrics_path.exists())


if __name__ == "__main__":
    unittest.main()
