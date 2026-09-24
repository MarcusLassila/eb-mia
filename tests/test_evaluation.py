import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from mia import evaluation


class TestEvaluation(unittest.TestCase):
    def _write_metrics(self, directory, target, model, dataset, attack, auc):
        metrics = {
            "attack": attack,
            "target_stem": target,
            "target_model": model,
            "target_dataset": dataset,
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

    def test_collect_and_print_metrics_groups_targets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            self._write_metrics(temporary_dir, "DDPM-a", "DDPM", "CelebA", "LiRA", 0.6)
            self._write_metrics(temporary_dir, "DDPM-b", "DDPM", "CelebA", "LiRA", 0.8)
            long_attack = "HG-Aggregate-Pooled-Gaussian-64"
            self._write_metrics(temporary_dir, "FM-a", "FlowMatching", "VGGFace2", long_attack, 0.7)

            grouped = evaluation.collect_grouped_metrics(temporary_dir)
            ddpm_summary = evaluation.summarize_metrics(grouped[("DDPM", "CelebA", "LiRA")])
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
                "DDPM-b",
                "DDPM",
                "CelebA",
                "BASE",
                0.6,
            )

            output_paths = evaluation.plot_average_roc_curves(temporary_dir)

            self.assertEqual(len(output_paths), 2)
            self.assertTrue(all(path.exists() for path in output_paths))
            png_path = Path(temporary_dir) / "average_roc_curves_DDPM-CelebA.png"
            self.assertTrue(png_path.exists())


if __name__ == "__main__":
    unittest.main()
