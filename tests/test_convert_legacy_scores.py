import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import convert_legacy_scores as convert_module

class TestConvertLegacyScores(unittest.TestCase):
    def test_convert_single_score_file_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            score_path = tmpdir / "scores_attack-BASE_target-DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pkl"
            checkpoint_path = tmpdir / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth"
            with open(score_path, "wb") as file:
                pickle.dump([0.1, 0.2, 0.3, 0.4], file)
            checkpoint_path.touch()
            with (
                patch.object(convert_module.utils, "get_train_indices", return_value=torch.tensor([1, 3], dtype=torch.long)),
                patch.object(convert_module, "load_dataset", return_value=list(range(4))),
            ):
                output_paths = convert_module.convert_legacy_scores(
                    scores_path=score_path,
                    indices_source_path=checkpoint_path,
                    data_dir=tmpdir,
                )
            self.assertEqual(output_paths, [score_path.resolve()])
            with open(score_path, "rb") as file:
                converted = pickle.load(file)
            self.assertEqual(converted["scores"], [0.1, 0.2, 0.3, 0.4])
            self.assertEqual(converted["train_mask"], [0, 1, 0, 1])

    def test_convert_score_directory_from_split_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            scores_dir = tmpdir / "scores"
            splits_dir = tmpdir / "splits"
            output_dir = tmpdir / "converted"
            scores_dir.mkdir()
            splits_dir.mkdir()

            score_0 = scores_dir / "scores_attack-BASE_target-DDPM-cifar10-rand-f0p5-s0-sz32-epoch4.pkl"
            score_1 = scores_dir / "scores_attack-BASE_target-DDPM-cifar10-rand-f0p5-s1-sz32-epoch4.pkl"
            with open(score_0, "wb") as file:
                pickle.dump([0.1, 0.2, 0.3, 0.4], file)
            with open(score_1, "wb") as file:
                pickle.dump([0.5, 0.6, 0.7, 0.8], file)

            split_0 = splits_dir / "cifar10-rand-f0p5-s0.pkl"
            split_1 = splits_dir / "cifar10-rand-f0p5-s1.pkl"
            with open(split_0, "wb") as file:
                pickle.dump([0, 2], file)
            with open(split_1, "wb") as file:
                pickle.dump([1, 3], file)

            with patch.object(convert_module, "load_dataset", return_value=list(range(4))):
                output_paths = convert_module.convert_legacy_scores(
                    scores_path=scores_dir,
                    indices_source_path=splits_dir,
                    data_dir=tmpdir,
                    output_dir=output_dir,
                )
            self.assertEqual(len(output_paths), 2)
            with open(output_dir / score_0.name, "rb") as file:
                converted_0 = pickle.load(file)
            with open(output_dir / score_1.name, "rb") as file:
                converted_1 = pickle.load(file)
            self.assertEqual(converted_0["train_mask"], [1, 0, 1, 0])
            self.assertEqual(converted_1["train_mask"], [0, 1, 0, 1])

    def test_convert_asserts_dataset_length_matches_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            score_path = tmpdir / "scores_attack-BASE_target-DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pkl"
            checkpoint_path = tmpdir / "DDPM-cifar10-rand-f0p5-s3-sz32-epoch4.pth"
            with open(score_path, "wb") as file:
                pickle.dump([0.1, 0.2, 0.3, 0.4], file)
            checkpoint_path.touch()
            with (
                patch.object(convert_module.utils, "get_train_indices", return_value=torch.tensor([0, 1], dtype=torch.long)),
                patch.object(convert_module, "load_dataset", return_value=list(range(3))),
            ):
                with self.assertRaises(AssertionError):
                    convert_module.convert_legacy_scores(
                        scores_path=score_path,
                        indices_source_path=checkpoint_path,
                        data_dir=tmpdir,
                    )

if __name__ == "__main__":
    unittest.main()
