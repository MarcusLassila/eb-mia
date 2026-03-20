import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from mia import evaluation as evaluation_module
from mia import loss_query as loss_query_module
from mia import path_utils
from mia import run_audit as run_audit_module


class TestLiRASampleAuditEndToEnd(unittest.TestCase):
    def test_lira_sample_audit_runs_from_loss_query_to_evaluation(self):
        dataset = list(range(4))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_path = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth"
            shadow_path_a = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s1-sz32-epoch4.pth"
            shadow_path_b = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s2-sz32-epoch4.pth"
            loss_sig_by_path = {
                target_path: (torch.tensor([0.90, 0.95, 1.40, 1.45], dtype=torch.float32), torch.tensor([1, 0, 1, 0], dtype=torch.bool)),
                shadow_path_a: (torch.tensor([1.00, 1.10, 1.20, 1.30], dtype=torch.float32), torch.tensor([1, 1, 0, 0], dtype=torch.bool)),
                shadow_path_b: (torch.tensor([1.40, 1.50, 0.80, 0.90], dtype=torch.float32), torch.tensor([0, 0, 1, 1], dtype=torch.bool)),
            }

            def fake_query_loss(self, loaded_dataset, model_path):
                assert loaded_dataset is dataset
                return loss_sig_by_path[Path(model_path)]

            query_config = loss_query_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "res_dir": tmpdir,
                "n_loss_samples": 5,
                "checkpoint_paths": [str(target_path), str(shadow_path_a), str(shadow_path_b)],
            })

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset),
                patch.object(loss_query_module.LossQuery, "query_loss", new=fake_query_loss),
            ):
                loss_query_module.run_loss_query(query_config, device=torch.device("cpu"))

            target_loss_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 5)
            shadow_loss_path_a = path_utils.loss_signals_dir(tmpdir, shadow_path_a) / path_utils.loss_signals_pickle_name(shadow_path_a, 5)
            shadow_loss_path_b = path_utils.loss_signals_dir(tmpdir, shadow_path_b) / path_utils.loss_signals_pickle_name(shadow_path_b, 5)

            audit_config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "res_dir": tmpdir,
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b)],
                "attack": {
                    "name": "LiRA-off-none",
                    "attack": "LiRA",
                    "offline": True,
                    "use_global_var": True,
                    "loss_transformation": "none",
                },
            })

            with (
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "get_audit_indices", return_value=torch.arange(4, dtype=torch.long)),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_sample_audit(config=audit_config)

            metrics_dir = path_utils.metrics_dir_from_target(tmpdir, target_path, "LiRA-off-none", "sample")
            metrics_path = metrics_dir / path_utils.metrics_pickle_name_from_target(target_path, "LiRA-off-none", "sample")
            self.assertTrue(metrics_path.exists())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                summaries = evaluation_module.run_evaluation(
                    config=run_audit_module.utils.Config({"res_dir": tmpdir, "metrics_folders": [str(metrics_dir)]}),
                )
            self.assertEqual(len(summaries), 1)
            self.assertIn("Evaluation summary", stdout.getvalue())
            self.assertTrue((tmpdir_path / f"average_roc_curves_{metrics_dir.stem}.png").exists())


if __name__ == "__main__":
    unittest.main()
