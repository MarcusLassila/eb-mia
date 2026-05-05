import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mia import loss_distribution


class TestLossDistribution(unittest.TestCase):
    def _write_loss_file(self, directory, name, loss_sigs, train_mask):
        '''
        Write a loss-signal pickle fixture.
        Args:
            directory (str | Path): Directory for the fixture.
            name (str): Fixture filename.
            loss_sigs (list): Loss-signal payload.
            train_mask (list): Membership mask payload.
        Returns:
            Path: Fixture path.
        '''
        loss_path = Path(directory) / name
        with open(loss_path, "wb") as file:
            pickle.dump(
                {
                    "loss_sigs": loss_sigs,
                    "train_mask": train_mask,
                },
                file,
            )
        return loss_path

    def test_load_loss_signal_samples_requires_2d_loss_sigs(self):
        '''
        Reject older averaged 1D loss-signal payloads.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            loss_path = self._write_loss_file(
                tmpdir,
                "loss_signals-target-ls3-nl0p1.pkl",
                [1.0, 2.0, 3.0],
                [True, False, True],
            )

            with self.assertRaisesRegex(ValueError, "2D loss_sigs"):
                loss_distribution.load_loss_signal_samples(loss_path)

    def test_load_loss_signal_samples_loads_raw_2d_values(self):
        '''
        Load raw 2D loss samples without averaging.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            loss_path = self._write_loss_file(
                tmpdir,
                "loss_signals-target-ls2-nl0p1.pkl",
                [[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]],
                [True, False, True],
            )

            loss_sigs, train_mask = loss_distribution.load_loss_signal_samples(loss_path)

            self.assertEqual(loss_sigs.shape, (3, 2))
            self.assertTrue(np.allclose(loss_sigs[0], np.asarray([1.0, 3.0])))
            self.assertTrue(np.array_equal(train_mask, np.asarray([True, False, True])))

    def test_collect_point_loss_values_groups_by_membership(self):
        '''
        Collect one point's loss samples into member and nonmember groups.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            member_path = self._write_loss_file(
                tmpdir,
                "loss_signals-member-ls2-nl0p1.pkl",
                [[0.0, 0.0], [1.0, 2.0]],
                [False, True],
            )
            nonmember_path = self._write_loss_file(
                tmpdir,
                "loss_signals-nonmember-ls2-nl0p1.pkl",
                [[0.0, 0.0], [10.0, 20.0]],
                [True, False],
            )

            grouped_values, sources = loss_distribution.collect_point_loss_values(
                [member_path, nonmember_path],
                data_index=1,
            )
            mean_values, _ = loss_distribution.collect_point_loss_values(
                [member_path, nonmember_path],
                data_index=1,
                sample_mode="mean",
            )

            self.assertTrue(np.allclose(grouped_values["member"], np.asarray([1.0, 2.0])))
            self.assertTrue(np.allclose(grouped_values["nonmember"], np.asarray([10.0, 20.0])))
            self.assertTrue(np.allclose(mean_values["member"], np.asarray([1.5])))
            self.assertTrue(np.allclose(mean_values["nonmember"], np.asarray([15.0])))
            self.assertEqual([source["group"] for source in sources], ["member", "nonmember"])

    def test_fit_parametric_distributions_returns_gaussian_and_student_t(self):
        '''
        Fit Gaussian and Student-t parameters for variable finite samples.
        Returns:
            None
        '''
        values = np.asarray([0.8, 1.0, 1.2, 1.7, 2.1, 2.5], dtype=float)

        fits = loss_distribution.fit_parametric_distributions(values, "test")

        self.assertIn("gaussian", fits)
        self.assertIn("student_t", fits)
        self.assertGreater(fits["gaussian"]["scale"], 0.0)
        self.assertGreater(fits["student_t"]["scale"], 0.0)

    def test_uniform_transform_summary_reports_ks_diagnostics(self):
        '''
        Summarize fitted-CDF transforms with uniformity diagnostics.
        Returns:
            None
        '''
        values = np.asarray([0.8, 1.0, 1.2, 1.7, 2.1, 2.5], dtype=float)
        fits = loss_distribution.fit_parametric_distributions(values, "test")

        summary = loss_distribution.uniform_transform_summary(values, fits)
        bins = loss_distribution.uniform_histogram_bins(n_values=256, max_bins=40)

        self.assertEqual(bins, 16)
        self.assertIn("gaussian", summary)
        self.assertIn("student_t", summary)
        self.assertGreaterEqual(summary["gaussian"]["min"], 0.0)
        self.assertLessEqual(summary["gaussian"]["max"], 1.0)
        self.assertGreaterEqual(summary["gaussian"]["ks_pvalue"], 0.0)
        self.assertLessEqual(summary["gaussian"]["ks_pvalue"], 1.0)

    def test_run_loss_distribution_analysis_writes_summary(self):
        '''
        Run the analysis pipeline and write a JSON summary.
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            loss_dir = Path(tmpdir) / "losses"
            output_dir = Path(tmpdir) / "analysis"
            loss_dir.mkdir()
            for index in range(4):
                offset = float(index)
                train_mask = [index < 2]
                self._write_loss_file(
                    loss_dir,
                    f"loss_signals-target-{index}-ls4-nl0p1.pkl",
                    [[1.0 + offset, 1.5 + offset, 2.0 + offset, 2.5 + offset]],
                    train_mask,
                )

            with (
                patch.object(loss_distribution, "plot_histogram_kde", return_value=output_dir / "hist.png"),
                patch.object(loss_distribution, "plot_cdf", return_value=output_dir / "cdf.png"),
                patch.object(loss_distribution, "plot_log_density", return_value=output_dir / "log_density.png"),
                patch.object(loss_distribution, "plot_qq", return_value=output_dir / "qq.png"),
                patch.object(loss_distribution, "plot_survival", return_value=output_dir / "survival.png"),
                patch.object(loss_distribution, "plot_uniform_transform", return_value=output_dir / "uniform.png"),
            ):
                result = loss_distribution.run_loss_distribution_analysis(
                    loss_paths=[loss_dir],
                    data_index=0,
                    output_dir=output_dir,
                    bins=5,
                )

            with open(result["summary_path"], "r") as file:
                summary = json.load(file)
            self.assertEqual(summary["data_index"], 0)
            self.assertEqual(summary["sample_mode"], "flatten")
            self.assertEqual(summary["n_loss_files"], 4)
            self.assertEqual(summary["groups"]["member"]["n_values"], 8)
            self.assertEqual(summary["groups"]["nonmember"]["n_values"], 8)
            self.assertIn("uniform_transforms", summary["groups"]["member"])
            self.assertEqual(len(result["plot_paths"]), 12)


if __name__ == "__main__":
    unittest.main()
