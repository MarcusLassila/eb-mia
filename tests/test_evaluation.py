import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from mia import evaluation
from mia import result_store


class TestEvaluation(unittest.TestCase):
    def _write_metrics(self, directory, target, model, dataset, attack, auc):
        metrics = {
            "attack": attack,
            "target_stem": target,
            "target_model": model,
            "target_dataset": dataset,
            "scenario_id": "setting-a",
            "settings": {"attack": {"attack": attack}},
            "AUC": auc,
            "pAUC@1%FPR": auc,
            "TPR@1%FPR": auc,
            "TPR@0.1%FPR": auc,
            "TPR@0.01%FPR": auc,
            "FPR": np.asarray([0.0, 0.01, 1.0]),
            "TPR": np.asarray([0.0, auc, 1.0]),
        }
        path = Path(directory) / f"metrics_attack-{attack}_target-{target}.pkl"
        with open(path, "wb") as file:
            pickle.dump(metrics, file)
        manifest_path = Path(directory) / "audit_manifest.json"
        paths = result_store.load_manifest_paths(manifest_path) if manifest_path.exists() else []
        paths.append(path)
        result_store.write_artifact(manifest_path, {"schema_version": 2, "metrics_paths": [str(item) for item in paths]})

    def test_collect_and_print_metrics_groups_targets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            self._write_metrics(temporary_dir, "DDPM-a", "DDPM", "CelebA", "LiRA", 0.6)
            self._write_metrics(temporary_dir, "DDPM-b", "DDPM", "CelebA", "LiRA", 0.8)
            long_attack = "HG-Aggregate-Pooled-Gaussian-64"
            self._write_metrics(temporary_dir, "FM-a", "FlowMatching", "VGGFace2", long_attack, 0.7)

            grouped = evaluation.collect_grouped_metrics(temporary_dir)
            ddpm_summary = evaluation.summarize_metrics(grouped[("DDPM", "CelebA", "setting-a", "LiRA")])
            output = io.StringIO()
            with redirect_stdout(output):
                evaluation.print_grouped_metrics(temporary_dir)

            default_output = output.getvalue()
            self.assertNotIn(" AUC ", default_output)
            output = io.StringIO()
            with redirect_stdout(output):
                evaluation.print_grouped_metrics(temporary_dir, show_auc=True)

            self.assertEqual(ddpm_summary["n_targets"], 2)
            self.assertAlmostEqual(ddpm_summary["AUC"], 0.7)
            self.assertIn("FlowMatching", output.getvalue())
            self.assertIn("DDPM", output.getvalue())
            self.assertIn("AUC", output.getvalue())
            for printed in (default_output, output.getvalue()):
                lines = printed.strip().splitlines()[1:]
                self.assertEqual(len({len(line) for line in lines}), 1)
                self.assertIn(long_attack, printed)
                header = lines[0]
                for label, value in [("pAUC@1%", "0.7000"), ("TPR@1%", "0.7000"), ("TPR@0.1%", "0.7000")]:
                    end = header.index(label) + len(label)
                    for row in lines[2:]:
                        self.assertEqual(row[end - len(value):end], value)

    def test_plot_average_roc_curves_writes_png_and_tikz_per_model_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            self._write_metrics(
                temporary_dir,
                "DDPM-a",
                "DDPM",
                "CelebA",
                "LiRA",
                0.7,
            )
            self._write_metrics(
                temporary_dir,
                "DDPM-a",
                "DDPM",
                "CelebA",
                "BASE",
                0.6,
            )

            output_paths = evaluation.plot_average_roc_curves(temporary_dir)

            self.assertEqual(len(output_paths), 2)
            self.assertTrue(all(path.exists() for path in output_paths))
            png_path = Path(temporary_dir) / "average_roc_curves_DDPM-CelebA_setting-a.png"
            self.assertTrue(png_path.exists())

    def test_manifest_ignores_stale_files_and_rejects_missing_or_duplicate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_metrics(directory, "a", "DDPM", "CelebA", "BASE", 0.7)
            stale = Path(directory) / "metrics_attack-stale_target-old.pkl"
            stale.write_bytes(b"not a valid pickle")
            self.assertEqual(len(evaluation.collect_grouped_metrics(directory)), 1)
            manifest_path = Path(directory) / "audit_manifest.json"
            paths = result_store.load_manifest_paths(manifest_path)
            result_store.write_artifact(manifest_path, {
                "schema_version": 2, "metrics_paths": [str(paths[0]), str(paths[0])],
            })
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluation.collect_grouped_metrics(directory)
            result_store.write_artifact(manifest_path, {
                "schema_version": 2, "metrics_paths": [str(Path(directory) / "missing.pkl")],
            })
            with self.assertRaises(FileNotFoundError):
                evaluation.collect_grouped_metrics(directory)

    def test_compared_attacks_require_identical_target_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_metrics(directory, "a", "DDPM", "CelebA", "BASE", 0.7)
            self._write_metrics(directory, "b", "DDPM", "CelebA", "LiRA", 0.8)
            with self.assertRaisesRegex(ValueError, "different target sets"):
                evaluation.collect_grouped_metrics(directory)

    def test_scenarios_are_separate_and_duplicate_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_metrics(directory, "a", "DDPM", "CelebA", "BASE", 0.7)
            manifest_path = Path(directory) / "audit_manifest.json"
            first_path, = result_store.load_manifest_paths(manifest_path)
            with first_path.open("rb") as file:
                metrics = pickle.load(file)
            other_path = Path(directory) / "other.pkl"
            metrics["scenario_id"] = "setting-b"
            result_store.write_artifact(other_path, metrics)
            result_store.write_artifact(manifest_path, {
                "schema_version": 2, "metrics_paths": [str(first_path), str(other_path)],
            })
            self.assertEqual(len(evaluation.collect_grouped_metrics(directory)), 2)
            metrics["scenario_id"] = "setting-a"
            result_store.write_artifact(other_path, metrics)
            with self.assertRaisesRegex(ValueError, "Duplicate target"):
                evaluation.collect_grouped_metrics(directory)


if __name__ == "__main__":
    unittest.main()
