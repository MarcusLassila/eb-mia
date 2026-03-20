import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import loss_query as loss_query_module
from mia import path_utils


class TestLossQuery(unittest.TestCase):
    def test_run_loss_query_saves_loss_signals_for_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = str(Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth")
            config = loss_query_module.utils.Config({
                "dataset": "cifar10",
                "data_dir": tmpdir,
                "batch_size": 2,
                "res_dir": tmpdir,
                "n_loss_samples": 10,
                "checkpoint_paths": [target_path],
            })
            dataset = list(range(4))

            with (
                patch.object(loss_query_module, "load_dataset", return_value=dataset) as load_dataset_fn,
                patch.object(
                    loss_query_module.LossQuery,
                    "query_loss",
                    return_value=(
                        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
                        torch.tensor([0, 1, 0, 1], dtype=torch.bool),
                    ),
                ) as query_loss_fn,
            ):
                saved_paths = loss_query_module.run_loss_query(config=config, device=torch.device("cpu"))

            load_dataset_fn.assert_called_once_with("cifar10", data_dir=tmpdir, size=32)
            query_loss_fn.assert_called_once()
            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 10)
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            self.assertTrue(torch.allclose(torch.tensor(payload["loss_sigs"]), torch.tensor([0.1, 0.2, 0.3, 0.4])))
            self.assertEqual(payload["train_mask"], [False, True, False, True])

    def test_run_loss_query_migrates_lira_scores_to_loss_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "DDPM-cifar10-smpl-f0p5-s3-sz32-epoch4.pth"
            score_path = path_utils.scores_dir(tmpdir, "LiRA-off", target_path) / path_utils.scores_pickle_name(target_path, "LiRA-off")
            score_path.parent.mkdir(parents=True, exist_ok=True)
            with open(score_path, "wb") as file:
                pickle.dump(
                    {
                        "scores": [0.0, 0.0, 0.0, 0.0],
                        "train_mask": [1, 0, 1, 0],
                        "loss_sigs": [
                            [0.1, 0.2, 0.3, 0.4],
                            [1.1, 1.2, 1.3, 1.4],
                        ],
                    },
                    file,
                )
            config = loss_query_module.utils.Config({
                "res_dir": tmpdir,
                "n_loss_samples": 7,
                "lira_score_paths": [str(score_path)],
            })

            saved_paths = loss_query_module.run_loss_query(config=config, device=torch.device("cpu"))

            saved_path = path_utils.loss_signals_dir(tmpdir, target_path) / path_utils.loss_signals_pickle_name(target_path, 7)
            self.assertEqual(saved_paths, [saved_path])
            with open(saved_path, "rb") as file:
                payload = pickle.load(file)
            self.assertTrue(torch.allclose(torch.tensor(payload["loss_sigs"]), torch.tensor([0.1, 0.2, 0.3, 0.4])))
            self.assertEqual(payload["train_mask"], [True, False, True, False])

    def test_run_loss_query_requires_n_loss_samples(self):
        config = loss_query_module.utils.Config({"checkpoint_paths": ["/tmp/model.pth"]})
        with self.assertRaisesRegex(ValueError, "n_loss_samples"):
            loss_query_module.run_loss_query(config=config, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
