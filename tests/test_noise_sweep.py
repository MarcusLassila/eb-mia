import csv
import pickle
import tempfile
import unittest
from pathlib import Path

from mia.noise_sweep import main


class TestNoiseSweep(unittest.TestCase):
    '''Check matched round-robin attacks and publication plot outputs.'''

    def test_two_noise_levels_write_offline_attack_curves_and_tikz(self):
        '''Run a small complete split sweep and inspect saved metrics and plots.'''
        masks = (
            (True, True, False, False),
            (False, False, True, True),
            (True, False, True, False),
            (False, True, False, True),
            (True, False, False, True),
            (False, True, True, False),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            loss_dir = Path(temporary_dir) / "losses"
            output_dir = Path(temporary_dir) / "plots"
            loss_dir.mkdir()
            for noise_level, noise_token in ((0.1, "0p1"), (0.9, "0p9")):
                for model_index, mask in enumerate(masks):
                    split_index, complement_index = divmod(model_index, 2)
                    complement_token = "-comp" if complement_index else ""
                    target_stem = f"FlowMatching-CelebA-smpl-f0p5-s{split_index}{complement_token}-sz64-epoch700"
                    filename = f"loss_signals-{target_stem}-ls1-nl{noise_token}.pkl"
                    losses = [
                        [0.5 + 0.04 * model_index + 0.03 * point_index + 0.01 * ((model_index + point_index) % 3) + 0.02 * noise_level * point_index]
                        for point_index in range(4)
                    ]
                    with (loss_dir / filename).open("wb") as file:
                        pickle.dump({"loss_sigs": losses, "train_mask": mask}, file)
            result = main(["--loss-dir", str(loss_dir), "--output-dir", str(output_dir), "--normalization", "none"])
            with result["summary_csv"].open(newline="") as file:
                summaries = list(csv.DictReader(file))
            with result["target_csv"].open(newline="") as file:
                targets = list(csv.DictReader(file))
            self.assertEqual(len(summaries), 4)
            self.assertEqual(len(targets), 24)
            self.assertEqual({row["attack"] for row in summaries}, {"BASE", "LiRA"})
            self.assertEqual({row["noise_level"] for row in summaries}, {"0.1", "0.9"})
            self.assertTrue(all(row["noise_level"] == row["file_level"] for row in summaries))
            self.assertTrue(all(row["n_targets"] == "6" for row in summaries))
            self.assertTrue(all(0.0 <= float(row["pAUC@1%FPR_mean"]) <= 1.0 for row in summaries))
            self.assertEqual(len(result["plots"]), 4)
            self.assertTrue(all(path.exists() for path in result["plots"]))
            tikz_paths = [path for path in result["plots"] if path.suffix == ".tex"]
            self.assertEqual(len(tikz_paths), 2)
            self.assertTrue(all("\\addlegendentry{LiRA}" in path.read_text() for path in tikz_paths))
            self.assertTrue(all("GlobalThreshold" not in path.read_text() for path in tikz_paths))
            expected_ticks = ",".join(f"{step / 10:.2f}" for step in range(11))
            self.assertTrue(all(f"xtick={{{expected_ticks}}},ytick={{{expected_ticks}}}" in path.read_text() for path in tikz_paths))

            inverted_output = Path(temporary_dir) / "inverted"
            inverted = main([
                "--loss-dir", str(loss_dir), "--output-dir", str(inverted_output),
                "--normalization", "none", "--invert-noise-levels",
            ])
            with inverted["summary_csv"].open(newline="") as file:
                inverted_summaries = list(csv.DictReader(file))
            self.assertEqual(inverted_summaries[0]["noise_level"], "0.1")
            self.assertEqual(inverted_summaries[0]["file_level"], "0.9")
            self.assertEqual(inverted_summaries[-1]["noise_level"], "0.9")
            self.assertEqual(inverted_summaries[-1]["file_level"], "0.1")
