import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

torchdiffeq_stub = types.ModuleType("torchdiffeq")

def odeint_stub(func, y0, t, *args, **kwargs):
    return torch.stack([y0 for _ in range(t.shape[0])], dim=0)

torchdiffeq_stub.odeint = odeint_stub
sys.modules.setdefault("torchdiffeq", torchdiffeq_stub)

from mia import evaluation as evaluation_module
from mia import loss_query as loss_query_module
from mia import path_utils
from mia import result_store
from mia import run_audit as run_audit_module


class TestLiRASampleAuditEndToEnd(unittest.TestCase):
    def test_lira_sample_audit_runs_from_loss_query_to_evaluation(self):
        dataset = list(range(4))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            target_path = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s0-sz32-epoch4.pth"
            shadow_path_a = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s1-sz32-epoch4.pth"
            shadow_path_b = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s2-sz32-epoch4.pth"
            shadow_path_c = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth"
            shadow_path_d = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s4-sz32-epoch4.pth"
            shadow_path_e = tmpdir_path / "DDPM-cifar10-smpl-f0p5-s5-sz32-epoch4.pth"
            loss_sig_by_path = {
                target_path: (torch.tensor([0.90, 0.95, 1.40, 1.45], dtype=torch.float32), torch.tensor([1, 0, 1, 0], dtype=torch.bool)),
                shadow_path_a: (torch.tensor([1.00, 1.10, 1.20, 1.30], dtype=torch.float32), torch.tensor([1, 1, 0, 0], dtype=torch.bool)),
                shadow_path_b: (torch.tensor([1.40, 1.50, 0.80, 0.90], dtype=torch.float32), torch.tensor([0, 0, 1, 1], dtype=torch.bool)),
                shadow_path_c: (torch.tensor([1.10, 1.20, 1.10, 1.20], dtype=torch.float32), torch.tensor([0, 0, 0, 0], dtype=torch.bool)),
                shadow_path_d: (torch.tensor([1.15, 1.25, 1.35, 1.45], dtype=torch.float32), torch.tensor([1, 0, 0, 1], dtype=torch.bool)),
                shadow_path_e: (torch.tensor([1.25, 1.35, 1.45, 1.55], dtype=torch.float32), torch.tensor([0, 1, 1, 0], dtype=torch.bool)),
            }

            def fake_query(self, loaded_dataset, model_path, n_data_points=None):
                assert loaded_dataset is dataset
                assert n_data_points is None
                return loss_sig_by_path[Path(model_path)]

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset),
                patch.object(loss_query_module.LossQuery, "query", new=fake_query),
            ):
                loss_query_module.run_loss_query(
                    checkpoint_paths=[str(target_path), str(shadow_path_a), str(shadow_path_b), str(shadow_path_c), str(shadow_path_d), str(shadow_path_e)],
                    checkpoint_properties=loss_query_module.utils.parse_properties_from_checkpoint_path(target_path),
                    dataset="cifar10",
                    data_dir=tmpdir,
                    batch_size=2,
                    res_dir=tmpdir,
                    n_samples=5,
                    device=torch.device("cpu"),
                    noise_level=0.1,
                )

            target_loss_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 5)
            shadow_loss_path_a = path_utils.loss_signals_dir(tmpdir, shadow_path_a) / path_utils.loss_signals_pickle_name(shadow_path_a, 5)
            shadow_loss_path_b = path_utils.loss_signals_dir(tmpdir, shadow_path_b) / path_utils.loss_signals_pickle_name(shadow_path_b, 5)
            shadow_loss_path_c = path_utils.loss_signals_dir(tmpdir, shadow_path_c) / path_utils.loss_signals_pickle_name(shadow_path_c, 5)
            shadow_loss_path_d = path_utils.loss_signals_dir(tmpdir, shadow_path_d) / path_utils.loss_signals_pickle_name(shadow_path_d, 5)
            shadow_loss_path_e = path_utils.loss_signals_dir(tmpdir, shadow_path_e) / path_utils.loss_signals_pickle_name(shadow_path_e, 5)

            audit_config = run_audit_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "results_root": tmpdir,
                "results_dir_name": "test_lira_sample",
                "audit_mode": "sample",
                "n_audit_samples": 4,
                "target_loss_paths": [str(target_loss_path)],
                "shadow_loss_paths": [str(shadow_loss_path_a), str(shadow_loss_path_b), str(shadow_loss_path_c), str(shadow_loss_path_d), str(shadow_loss_path_e)],
                "attack": {
                    "name": "LiRA-on-none",
                    "attack": "LiRA",
                    "offline": False,
                    "use_global_dispersion": True,
                    "share_variance": True,
                    "loss_transformation": "none",
                },
            })

            with (
                patch.object(run_audit_module, "select_sample_audit_indices", return_value=torch.arange(4, dtype=torch.long)),
                patch.object(run_audit_module, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                run_audit_module.run_sample_audit(config=audit_config)

            manifest_path = Path(tmpdir) / "test_lira_sample" / "audit_manifest.json"
            metrics_path, = result_store.load_manifest_paths(manifest_path)
            self.assertTrue(metrics_path.exists())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                evaluation_module.print_grouped_metrics(metrics_path.parent)
            self.assertIn("Benchmark summary", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
