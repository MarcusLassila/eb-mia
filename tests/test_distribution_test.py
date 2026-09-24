import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mia import distribution_test

class TestDistributionTest(unittest.TestCase):
    def write_signals(self, folder, model_index, train_mask, scale=1.0):
        '''Write one checkpoint fixture with positive query samples; return its path.'''
        query_offsets = np.arange(5, dtype=float) * 0.1
        point_bases = np.array([1.0, 2.0]) + model_index * scale
        signals = point_bases[:, None] + query_offsets
        path = folder / f"loss_signals-model{model_index}.pkl"
        with open(path, "wb") as file:
            pickle.dump({"loss_sigs": signals.tolist(), "train_mask": train_mask}, file)
        return path

    def test_load_selected_signals_matches_log_standardized(self):
        '''Match audit log standardization within each model; preserve memberships.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.write_signals(Path(tmpdir), 2, [True, False])
            selected, masks, indices = distribution_test.load_selected_signals(
                [path], max_points=2, seed=0, normalization="log_standardized"
            )
            raw = np.array([[3.0, 3.1, 3.2, 3.3, 3.4], [4.0, 4.1, 4.2, 4.3, 4.4]], dtype=np.float32)
            logged = np.log(raw)
            expected = (logged - logged.mean()) / logged.std(ddof=0)
            self.assertTrue(np.allclose(selected[0], expected, atol=1e-6))
            self.assertEqual(indices.tolist(), [0, 1])
            self.assertEqual(masks.tolist(), [[True, False]])

    def test_run_distribution_analysis_writes_grouped_statistics_and_plots(self):
        '''Write per-model query tests and member-separated model-mean tests.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "signals"
            input_dir.mkdir()
            for model_index in range(6):
                mask = [model_index < 3, model_index >= 3]
                self.write_signals(input_dir, model_index, mask)
            output_dir = Path(tmpdir) / "results"
            result = distribution_test.run_distribution_analysis(input_dir, output_dir, max_points=2)
            with open(result["csv_path"], newline="") as file:
                records = list(csv.DictReader(file))
            with open(result["summary_path"]) as file:
                summary = json.load(file)
            query_records = [row for row in records if row["scope"] == "query_outputs"]
            mean_records = [row for row in records if row["scope"] == "model_means"]
            self.assertEqual(len(query_records), 12)
            self.assertEqual(len(mean_records), 4)
            self.assertTrue(all(row["shapiro_p"] for row in records))
            self.assertTrue(all(row["anderson_5pct_critical"] for row in records))
            self.assertEqual(summary["n_models"], 6)
            self.assertEqual(summary["data_indices"], [0, 1])
            self.assertEqual(len(result["plot_paths"]), 32)
            self.assertTrue(all(path.is_file() for path in result["plot_paths"]))
            tikz_paths = [path for path in result["plot_paths"] if path.suffix == ".tex"]
            self.assertEqual(len(tikz_paths), 16)
            self.assertTrue(all(r"\begin{axis}[" in path.read_text() for path in tikz_paths))
            histogram_tikz = next(path for path in tikz_paths if "_hist" in path.name)
            self.assertIn("ybar interval", histogram_tikz.read_text())
            self.assertIn("Gaussian (mu=", histogram_tikz.read_text())
            self.assertEqual(result["csv_path"].parent, output_dir / "distributional_analysis")
            member_mean = next(row for row in mean_records if row["data_index"] == "0" and row["group"] == "member")
            self.assertAlmostEqual(float(member_mean["mean"]), 2.2, places=5)

    def test_rejects_incompatible_model_shapes_and_nonpositive_log_input(self):
        '''Reject files that cannot share a query shape or be log transformed.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            first = self.write_signals(folder, 0, [True, False])
            second = folder / "loss_signals-short.pkl"
            with open(second, "wb") as file:
                pickle.dump({"loss_sigs": [[1.0, 2.0, 3.0]], "train_mask": [False]}, file)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                distribution_test.load_selected_signals([first, second], 2, 0, "none")
            with open(first, "wb") as file:
                pickle.dump({"loss_sigs": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                             "train_mask": [True, False]}, file)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                distribution_test.load_selected_signals([first], 2, 0, "log_standardized")

    def test_cli_defaults(self):
        '''Use the requested output root and default raw-signal mode.'''
        args = distribution_test.parse_args(["--input-dir", "/tmp/signals"])
        self.assertEqual(args.output_dir, "temp_results")
        self.assertEqual(args.normalization, "none")


if __name__ == "__main__":
    unittest.main()
