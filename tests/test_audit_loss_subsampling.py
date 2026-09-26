import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mia import path_utils
from mia import run_audit
from mia.utils import subsample_loss_signals


class TestAuditLossSubsampling(unittest.TestCase):
    '''Check random loss-query subsets and audit config wiring.'''

    def test_sorted_rows_are_sampled_independently_and_reproducibly(self):
        '''Select distinct unordered subsets from sorted model and point rows.'''
        signals = torch.arange(10, dtype=torch.float32).repeat(2, 4, 1)
        generator = torch.Generator().manual_seed(7)
        selected = subsample_loss_signals(signals, 3, generator)
        repeated = subsample_loss_signals(signals, 3, torch.Generator().manual_seed(7))

        self.assertEqual(selected.shape, (2, 4, 3))
        self.assertTrue(torch.equal(selected, repeated))
        self.assertTrue(all(len(set(row.tolist())) == 3 for row in selected.reshape(-1, 3)))
        self.assertGreater(len({tuple(row.tolist()) for row in selected.reshape(-1, 3)}), 1)
        self.assertTrue(any(row.tolist() != [0, 1, 2] for row in selected.reshape(-1, 3)))
        self.assertEqual(subsample_loss_signals(signals, 3).shape, selected.shape)
        self.assertTrue(torch.equal(subsample_loss_signals(signals, 10), signals))

        for invalid_count in (0, -1, 11):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaisesRegex(ValueError, "n_loss_samples"):
                    subsample_loss_signals(signals, invalid_count)

    def test_sample_and_entity_audits_use_configured_count_before_normalization(self):
        '''Check both audit modes select two queries and default to the full tensor.'''
        original_normalize = run_audit.normalize_loss_signals
        masks = ([1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0])
        for mode in ("sample", "entity"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary_dir:
                directory = Path(temporary_dir)
                loss_paths = []
                mode_token = "smpl-f0p5" if mode == "sample" else "ent-f0p5-p0p5"
                for model_index, mask in enumerate(masks):
                    checkpoint = directory / f"DDPM-CelebA-{mode_token}-s{model_index}-sz64-epoch10.pth"
                    loss_path = directory / path_utils.loss_signals_pickle_name(checkpoint, 5)
                    losses = [[1.0 + model_index + point_index + sample_index / 10
                               for sample_index in range(5)] for point_index in range(4)]
                    with loss_path.open("wb") as file:
                        pickle.dump({"loss_sigs": losses, "train_mask": mask}, file)
                    loss_paths.append(loss_path)

                config_values = {
                    "audit_mode": mode,
                    "dataset": "CelebA",
                    "mode": "all",
                    "seed": 7,
                    "audit_random_seed": 7,
                    "n_loss_samples": 2,
                    "loss_normalization": "none",
                    "round_robin": False,
                    "target_loss_paths": [str(loss_paths[0])],
                    "shadow_loss_paths": [str(path) for path in loss_paths[1:]],
                    "results_root": temporary_dir,
                    "results_dir_name": "loss_subsampling",
                    "print_summary": False,
                    "attack": {"attack": "BASE" if mode == "sample" else "CompositeBASE",
                               "offline": True, "prior": 0.5},
                }
                config = run_audit.utils.Config(config_values)
                captured = []

                def capture_normalization(target, shadows, normalization):
                    '''Record tensors at the normalization boundary and return normalized tensors.'''
                    captured.append((target.clone(), shadows.clone()))
                    return original_normalize(target, shadows, normalization)

                audit = run_audit.run_sample_audit if mode == "sample" else run_audit.run_entity_audit
                metadata = {"n_samples": 4, "n_entities": 2, "entity_ids": [0, 0, 1, 1]}
                with (
                    patch.object(run_audit, "normalize_loss_signals", side_effect=capture_normalization),
                    patch.object(run_audit, "load_dataset_metadata", return_value=metadata),
                    patch.object(run_audit, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
                ):
                    audit(config)
                    audit(config)
                    del config.n_loss_samples
                    audit(config)

                self.assertEqual(len(captured), 3)
                self.assertEqual(captured[0][0].shape, (4, 2))
                self.assertEqual(captured[0][1].shape, (2, 4, 2))
                self.assertTrue(torch.equal(captured[0][0], captured[1][0]))
                self.assertTrue(torch.equal(captured[0][1], captured[1][1]))
                self.assertEqual(captured[2][0].shape, (4, 5))
                self.assertEqual(captured[2][1].shape, (2, 4, 5))
