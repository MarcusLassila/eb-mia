import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from mia import evaluation


def _metrics_dict(offset=0.0):
    fpr = np.array([0.0, 0.001, 0.01, 1.0], dtype=float)
    tpr = np.array([0.0, 0.2 + offset, 0.5 + offset, 1.0], dtype=float)
    return {
        "AUC": 0.7 + offset,
        "TPR@1%FPR": float(np.interp(1e-2, fpr, tpr)),
        "TPR@0.1%FPR": float(np.interp(1e-3, fpr, tpr)),
        "FPR": fpr,
        "TPR": tpr,
        "thresholds": np.array([np.inf, 0.3, 0.2, 0.1], dtype=float),
    }


class TestEvaluationCli(unittest.TestCase):
    def test_main_uses_cli_metrics_folders_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as file:
                yaml.safe_dump(
                    {
                        "res_dir": tmpdir,
                        "attack": "BASE",
                        "target_model_path": str(Path(tmpdir) / "DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000.pth"),
                        "metrics_folders": ["sample"],
                        "low_exponent": -4,
                    },
                    file,
                )
            with patch.object(evaluation, "run_evaluation") as run_eval_fn:
                evaluation.main([
                    "--config",
                    str(config_path),
                    "--metrics-folders",
                    "sample",
                    "entity-all",
                    "--low-exponent",
                    "-3",
                ])
            run_eval_fn.assert_called_once()
            kwargs = run_eval_fn.call_args.kwargs
            self.assertEqual(kwargs["metrics_folders_override"], ["sample", "entity-all"])
            self.assertEqual(kwargs["low_exponent_override"], -3)

    def test_collect_metrics_folder_summaries_requires_same_target_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            folder_a = tmpdir / "A"
            folder_b = tmpdir / "B"
            folder_a.mkdir()
            folder_b.mkdir()
            file_a = folder_a / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-sample.pkl"
            file_b = folder_b / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s1-sz64-epoch1000_mode-sample.pkl"
            with open(file_a, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            with open(file_b, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            with self.assertRaisesRegex(ValueError, "same target models"):
                evaluation.collect_metrics_folder_summaries([folder_a, folder_b])

    def test_collect_metrics_folder_summaries_uses_short_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parent = tmpdir / "BASE-DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder = parent / "sample"
            folder.mkdir(parents=True)
            metrics_path = folder / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-sample.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder])
            label = summaries[0]["label"]
            self.assertIn("BASE-ent-f0p5-p0p5-epoch1000/sample", label)
            self.assertNotIn("DDPM", label)
            self.assertNotIn("CelebA2", label)
            self.assertNotIn("sz64", label)

    def test_collect_metrics_folder_summaries_asserts_fixed_fpr_metrics_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "sample"
            folder.mkdir()
            bad_metrics = _metrics_dict()
            bad_metrics["TPR@1%FPR"] = bad_metrics["TPR@1%FPR"] + 0.1
            metrics_path = folder / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-sample.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(bad_metrics, f)
            with self.assertRaises(AssertionError):
                evaluation.collect_metrics_folder_summaries([folder])

    def test_print_metrics_table_formats_percentages(self):
        summaries = [{
            "label": "BASE-ent-f0p5-p0p5-epoch1000/sample",
            "AUC": {"mean": 0.75, "std": 0.05},
            "TPR@1%FPR": {"mean": 0.12, "std": 0.01},
            "TPR@0.1%FPR": {"mean": 0.03, "std": 0.005},
        }]
        with patch("builtins.print") as print_fn:
            evaluation.print_metrics_table(summaries)
        printed = "\n".join(" ".join(map(str, call.args)) for call in print_fn.call_args_list)
        self.assertIn("75.00% ± 5.00%", printed)
        self.assertIn("12.00% ± 1.00%", printed)
        self.assertIn("3.00% ± 0.50%", printed)


if __name__ == "__main__":
    unittest.main()
