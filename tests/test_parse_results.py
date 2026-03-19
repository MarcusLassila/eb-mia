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
            parent = tmpdir / "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder = parent / "BASE-sample"
            folder.mkdir(parents=True)
            metrics_path = folder / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-sample.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder])
            label = summaries[0]["label"]
            self.assertEqual(label, "BASE-f0p5-p0p5-e1000-sample")
            self.assertNotIn("/", label)
            self.assertNotIn("DDPM", label)
            self.assertNotIn("CelebA2", label)
            self.assertNotIn("sz64", label)

    def test_collect_metrics_folder_summaries_preserves_offline_attack_suffix_in_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parent = tmpdir / "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder = parent / "BASE-off-sample"
            folder.mkdir(parents=True)
            metrics_path = folder / "metrics_attack-BASE-off_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-sample.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder])
            self.assertEqual(summaries[0]["label"], "BASE-off-f0p5-p0p5-e1000-sample")

    def test_collect_metrics_folder_summaries_label_ignores_metrics_subfolder_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parent = tmpdir / "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder = parent / "BASE-entity-all"
            folder.mkdir(parents=True)
            metrics_path = folder / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-entity-all.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder])
            self.assertEqual(summaries[0]["label"], "BASE-f0p5-p0p5-e1000-ent-all")

    def test_collect_metrics_folder_summaries_compacts_rand_and_max_one_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parent = tmpdir / "DDPM-cifar10-rand-f0p5-sz32-epoch4"
            folder = parent / "BASE-entity-max_one_train"
            folder.mkdir(parents=True)
            metrics_path = folder / "metrics_attack-BASE_target-DDPM-cifar10-rand-f0p5-s0-sz32-epoch4_mode-entity.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder])
            self.assertEqual(summaries[0]["label"], "BASE-f0p5-e4-ent-max_one")

    def test_collect_metrics_folder_summaries_compacts_exclude_train_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parent = tmpdir / "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder = parent / "BASE-entity-exclude_train"
            folder.mkdir(parents=True)
            metrics_path = folder / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-entity.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder])
            self.assertEqual(summaries[0]["label"], "BASE-f0p5-p0p5-e1000-ent-excl_train")

    def test_collect_metrics_folder_summaries_adds_n_suffix_from_metrics_dir_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parent = tmpdir / "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder = parent / "BASE-entity-max_one_train-n10"
            folder.mkdir(parents=True)
            metrics_path = folder / "metrics_attack-BASE_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-entity_n-10_min-none_max-none.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder])
            self.assertEqual(summaries[0]["label"], "BASE-f0p5-p0p5-e1000-ent-max_one-n10")

    def test_collect_metrics_folder_summaries_allows_different_entity_sampling_multiplicities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parent = tmpdir / "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder_a = parent / "BASE-entity-max_one_train"
            folder_b = parent / "BASE-entity-exclude_train"
            folder_a.mkdir(parents=True)
            folder_b.mkdir(parents=True)
            targets = [
                "DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000",
                "DDPM-CelebA2-ent-f0p5-p0p5-s1-sz64-epoch1000",
            ]
            for target in targets:
                metrics_a_1 = folder_a / f"metrics_attack-BASE_target-{target}_mode-entity_n-2_min-none_max-none.pkl"
                metrics_a_2 = folder_a / f"metrics_attack-BASE_target-{target}_mode-entity_n-10_min-none_max-none.pkl"
                metrics_b = folder_b / f"metrics_attack-BASE_target-{target}_mode-entity_min-none_max-none.pkl"
                with open(metrics_a_1, "wb") as f:
                    pickle.dump(_metrics_dict(), f)
                with open(metrics_a_2, "wb") as f:
                    pickle.dump(_metrics_dict(), f)
                with open(metrics_b, "wb") as f:
                    pickle.dump(_metrics_dict(), f)
            _, summaries = evaluation.collect_metrics_folder_summaries([folder_a, folder_b])
            expected_targets = tuple(sorted(targets))
            self.assertEqual(summaries[0]["target_stems"], expected_targets)
            self.assertEqual(summaries[1]["target_stems"], expected_targets)

    def test_plot_average_roc_curves_uses_compact_title(self):
        fpr_space = np.array([1e-4, 1e-2, 1.0], dtype=float)
        summaries = [{
            "target_stems": ("DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000",),
            "mean_tpr": np.array([1e-4, 0.5, 1.0], dtype=float),
            "AUC": {"mean": 0.75, "std": 0.05},
            "label": "BASE-f0p5-p0p5-e1000-sample",
            "path": Path("sample"),
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mia.evaluation.plt.title") as title_fn:
                evaluation.plot_average_roc_curves(tmpdir, fpr_space, summaries)
        title = title_fn.call_args.args[0]
        self.assertEqual(title, "DDPM-CelebA2-sz64 | 1 target models")
        self.assertNotIn("Average ROC", title)

    def test_plot_average_roc_curves_adds_mode_stem_to_label(self):
        fpr_space = np.array([1e-4, 1e-2, 1.0], dtype=float)
        summaries = [{
            "target_stems": ("DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000",),
            "mean_tpr": np.array([1e-4, 0.5, 1.0], dtype=float),
            "AUC": {"mean": 0.75, "std": 0.05},
            "label": "BASE-f0p5-p0p5-e1000-ent-all",
            "path": Path("entity-all"),
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mia.evaluation.plt.loglog") as loglog_fn:
                evaluation.plot_average_roc_curves(tmpdir, fpr_space, summaries)
        first_label = loglog_fn.call_args_list[0].kwargs["label"]
        self.assertIn("BASE-f0p5-p0p5-e1000-ent-all | AUC:", first_label)

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

    def test_collect_metrics_folder_summaries_preserves_low_endpoint_interpolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "DDPM-CelebA2-ent-f0p5-p0p5-sz64-epoch1000"
            folder = parent / "BASE-off-entity-exclude_train-n10"
            folder.mkdir(parents=True)
            fpr = np.array([0.0, 0.0, 0.001, 1.0], dtype=float)
            tpr = np.array([0.0, 0.1, 0.2, 1.0], dtype=float)
            metrics = {
                "AUC": 0.7,
                "TPR@1%FPR": float(np.interp(1e-2, fpr, tpr)),
                "TPR@0.1%FPR": float(np.interp(1e-3, fpr, tpr)),
                "FPR": fpr,
                "TPR": tpr,
                "thresholds": np.array([np.inf, 0.3, 0.2, 0.1], dtype=float),
            }
            metrics_path = folder / "metrics_attack-BASE-off_target-DDPM-CelebA2-ent-f0p5-p0p5-s0-sz64-epoch1000_mode-entity_n-10_min-none_max-none.pkl"
            with open(metrics_path, "wb") as f:
                pickle.dump(metrics, f)
            fpr_space, summaries = evaluation.collect_metrics_folder_summaries([folder], low_exponent=-4)
            self.assertEqual(float(fpr_space[0]), 1e-4)
            self.assertGreater(float(summaries[0]["mean_tpr"][0]), 0.0)
            self.assertAlmostEqual(float(summaries[0]["mean_tpr"][0]), float(np.interp(1e-4, fpr, tpr)))

    def test_print_metrics_table_formats_percentages(self):
        summaries = [{
            "label": "BASE-f0p5-p0p5-e1000-sample",
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
