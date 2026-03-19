import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from data.datasets import EntityDataset
from mia import loss_query as loss_query_module
from mia import path_utils
from mia import run_audit as run_audit_module


class _TinyEntityDataset(EntityDataset):
    def __init__(self):
        self._entity_ids = torch.tensor([10, 10, 20, 20], dtype=torch.long)

    @property
    def entity_ids(self):
        return self._entity_ids

    @property
    def unique_entity_ids(self):
        return torch.unique(self._entity_ids, sorted=True)

    @property
    def max_entity_id(self):
        return int(self._entity_ids.max().item())

    @property
    def n_entities(self):
        return int(torch.unique(self._entity_ids).numel())

    def get_entity_index_table(self):
        return {10: [0, 1], 20: [2, 3]}

    def __getitem__(self, index):
        return torch.tensor([float(index)], dtype=torch.float32)

    def __len__(self):
        return len(self._entity_ids)


class TestAuditEndToEnd(unittest.TestCase):
    def test_loss_query_and_entity_audit_run_end_to_end_for_base(self):
        dataset = _TinyEntityDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_path = tmpdir_path / "DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch10.pth"
            shadow_path_a = tmpdir_path / "DDPM-CelebA2-ent-f0p5-p0p5-s1-sz64-epoch10.pth"
            shadow_path_b = tmpdir_path / "DDPM-CelebA2-ent-f0p5-p0p5-s2-sz64-epoch10.pth"
            loss_sig_by_path = {
                target_path: (torch.tensor([0.2, 0.5, 0.1, 0.1], dtype=torch.float32), torch.tensor([1, 0, 0, 0], dtype=torch.bool)),
                shadow_path_a: (torch.tensor([0.9, 0.2, 1.1, 1.2], dtype=torch.float32), torch.tensor([1, 1, 0, 0], dtype=torch.bool)),
                shadow_path_b: (torch.tensor([1.0, 0.3, 0.8, 0.9], dtype=torch.float32), torch.tensor([0, 0, 1, 1], dtype=torch.bool)),
            }

            def fake_query_loss(self, loaded_dataset, model_path):
                assert loaded_dataset is dataset
                return loss_sig_by_path[Path(model_path)]

            query_config = loss_query_module.utils.Config({
                "dataset": "CelebA2",
                "data_dir": tmpdir,
                "batch_size": 2,
                "res_dir": tmpdir,
                "n_loss_samples": 3,
                "checkpoint_paths": [str(target_path), str(shadow_path_a), str(shadow_path_b)],
            })

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset),
                patch.object(loss_query_module.LossQuery, "query_loss", new=fake_query_loss),
            ):
                loss_query_module.run_loss_query(query_config, device=torch.device("cpu"))

            target_loss_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 3)
            shadow_loss_path_a = path_utils.loss_signals_dir(tmpdir, shadow_path_a) / path_utils.loss_signals_pickle_name(shadow_path_a, 3)
            shadow_loss_path_b = path_utils.loss_signals_dir(tmpdir, shadow_path_b) / path_utils.loss_signals_pickle_name(shadow_path_b, 3)
            self.assertTrue(target_loss_path.exists())

            audit_config = run_audit_module.utils.Config({
                "dataset": "CelebA2",
                "data_dir": tmpdir,
                "res_dir": tmpdir,
                "audit_mode": "entity",
                "mode": "all",
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5},
            })

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_entity_audit(config=audit_config)

            metrics_path = (
                path_utils.metrics_dir_from_target(tmpdir, target_path, "BASE-off", "entity", entity_audit_mode="all")
                / path_utils.metrics_pickle_name_from_target(target_path, "BASE-off", "entity")
            )
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "rb") as file:
                metrics = pickle.load(file)
            self.assertIn("AUC", metrics)
            self.assertIn("audit_config", metrics)


if __name__ == "__main__":
    unittest.main()
