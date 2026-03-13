import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Subset

from accelerate.accelerate import AcceleratorLite
from data.data import EntityDataset
from generative_models.ddpm import DDPM
from mia import evaluation as evaluation_module
from mia import path_utils
from mia import run_audit as run_audit_module
from mia import run_mia as run_mia_module
from training import train_split
from training.train_loop import TrainConfig, TrainLoop


class _TinyCelebA2(EntityDataset):
    def __init__(self):
        images = []
        entity_ids = []
        for entity_id in range(8):
            raw_entity_id = 10 * entity_id + 7
            for sample_id in range(2):
                image = torch.zeros(1, 4, 4, dtype=torch.float32)
                image[:, :, entity_id % 4] = -0.5 + 0.15 * entity_id
                image[:, sample_id::2, :] += 0.1
                image[:, :, (entity_id + sample_id) % 4] += 0.05
                images.append(image.clamp(-1.0, 1.0))
                entity_ids.append(raw_entity_id)
        self.images = torch.stack(images)
        self._entity_ids = torch.tensor(entity_ids, dtype=torch.long)

    @property
    def entity_ids(self):
        return self._entity_ids

    @property
    def n_entities(self):
        return int(torch.unique(self._entity_ids).numel())

    @property
    def max_entity_id(self):
        return int(self._entity_ids.max().item())

    def __getitem__(self, index):
        return self.images[int(index)]

    def __len__(self):
        return self.images.shape[0]


class TestLiRAOfflineEntityAuditEndToEnd(unittest.TestCase):
    def _train_ddpm_checkpoint(self, dataset, train_indices, savepath, seed):
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        train_mask[torch.tensor(train_indices, dtype=torch.long)] = True
        nontrain_indices = torch.nonzero(~train_mask, as_tuple=True)[0]
        val_indices = nontrain_indices[:2]

        train_dataset = Subset(dataset, torch.tensor(train_indices, dtype=torch.long))
        val_dataset = Subset(dataset, val_indices)

        ddpm_config = {
            "image_dim": (1, 4, 4),
            "time_steps": 10,
            "beta_schedule": "linear",
            "base_channels": 32,
            "channel_mult": (1,),
            "n_attention_heads": 1,
            "attention_resolutions": (4,),
            "dropout": 0.0,
            "resample_with_conv": True,
            "use_sdpa": True,
        }
        model = DDPM(**ddpm_config)
        train_config = TrainConfig(
            batch_size=2,
            simul_batch_size=2,
            epochs=2,
            epochs_per_checkpoint=2,
            lr=1e-3,
            weight_decay=0.0,
            ema_decay=0.0,
            grad_clip=0.0,
            autocast_dtype="float16",
            lr_scheduler="none",
        )
        accelerator = AcceleratorLite(torch_compile=False, base_seed=seed)
        TrainLoop(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_config=train_config,
            model_config=ddpm_config,
            accelerator=accelerator,
            savepath=savepath,
        ).train()
        return savepath.with_name(f"{savepath.stem}-epoch2{savepath.suffix}")

    def test_lira_offline_entity_audit_runs_from_trained_shadow_pairs_to_evaluation(self):
        dataset = _TinyCelebA2()
        entity_fraction = 0.5
        per_entity_fraction = 0.5
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoints_dir = tmpdir_path / "checkpoints"
            results_dir = tmpdir_path / "mia_results"
            splits_dir = tmpdir_path / "splits"
            checkpoints_dir.mkdir()
            results_dir.mkdir()
            splits_dir.mkdir()

            target_split_path = train_split.create_entity_subset(
                dataset_name="CelebA2",
                entity_ids=dataset.entity_ids,
                entity_fraction=entity_fraction,
                per_entity_fraction=per_entity_fraction,
                seed=0,
                output_dir=splits_dir,
            )
            target_path = self._train_ddpm_checkpoint(
                dataset=dataset,
                train_indices=train_split.load_indices(target_split_path),
                savepath=checkpoints_dir / f"DDPM-{target_split_path.stem}-sz4.pth",
                seed=0,
            )

            shadow_split_paths = []
            for seed in (1, 2):
                split_path = train_split.create_entity_subset(
                    dataset_name="CelebA2",
                    entity_ids=dataset.entity_ids,
                    entity_fraction=entity_fraction,
                    per_entity_fraction=per_entity_fraction,
                    seed=seed,
                    output_dir=splits_dir,
                )
                shadow_split_paths.append(split_path)
                shadow_split_paths.append(
                    train_split.create_entity_complement_subset(
                        subset_path=split_path,
                        entity_ids=dataset.entity_ids,
                        output_dir=splits_dir,
                    )
                )

            shadow_paths = []
            for seed, split_path in enumerate(shadow_split_paths, start=1):
                shadow_paths.append(
                    self._train_ddpm_checkpoint(
                        dataset=dataset,
                        train_indices=train_split.load_indices(split_path),
                        savepath=checkpoints_dir / f"DDPM-{split_path.stem}-sz4.pth",
                        seed=seed,
                    )
                )

            self.assertEqual(len(shadow_paths), 4)
            shadow_entity_mask = torch.zeros((len(shadow_paths), dataset.max_entity_id + 1), dtype=torch.bool)
            for shadow_idx, split_path in enumerate(shadow_split_paths):
                shadow_indices = train_split.load_indices(split_path)
                shadow_entity_ids = torch.unique(dataset.entity_ids[torch.tensor(shadow_indices, dtype=torch.long)])
                shadow_entity_mask[shadow_idx, shadow_entity_ids] = True
            unique_entity_ids = torch.unique(dataset.entity_ids)
            self.assertTrue(torch.equal(shadow_entity_mask[:, unique_entity_ids].sum(dim=0), torch.full((dataset.n_entities,), 2, dtype=torch.int64)))

            mia_config = run_mia_module.utils.Config({
                "dataset": "CelebA2",
                "data_dir": str(tmpdir_path),
                "batch_size": 4,
                "round_robin": False,
                "res_dir": str(results_dir),
                "target_model_paths": [str(target_path)],
                "shadow_model_paths": [str(path) for path in shadow_paths],
                "attack": {
                    "name": "LiRA-off",
                    "attack": "LiRA",
                    "offline": True,
                    "n_loss_samples": 1,
                    "loss_transformation": "none",
                },
            })
            audit_config = run_audit_module.utils.Config({
                "dataset": "CelebA2",
                "data_dir": str(tmpdir_path),
                "res_dir": str(results_dir),
                "score_paths": [str(path_utils.scores_dir(results_dir, "LiRA-off", target_path))],
                "audit_mode": "entity",
                "mode": "all",
            })

            with (
                patch.object(run_mia_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
            ):
                run_mia_module.run_mia(config=mia_config, device=torch.device("cpu"))
                run_audit_module.run_entity_audit(config=audit_config)

            scores_path = (
                path_utils.scores_dir(results_dir, "LiRA-off", target_path)
                / path_utils.scores_pickle_name(target_path, "LiRA-off")
            )
            self.assertTrue(scores_path.exists())
            with open(scores_path, "rb") as file:
                scores_payload = pickle.load(file)
            self.assertEqual(len(scores_payload["scores"]), len(dataset))
            self.assertEqual(len(scores_payload["train_mask"]), len(dataset))
            self.assertEqual(torch.tensor(scores_payload["loss_sigs"]).shape, (5, len(dataset)))
            self.assertEqual(torch.tensor(scores_payload["shadow_train_mask"]).shape, (4, len(dataset)))

            saved_shadow_train_mask = torch.tensor(scores_payload["shadow_train_mask"], dtype=torch.bool)
            saved_shadow_entity_mask = torch.zeros((saved_shadow_train_mask.shape[0], dataset.max_entity_id + 1), dtype=torch.bool)
            entity_index_table = dataset.get_entity_index_table()
            for shadow_idx, sample_mask in enumerate(saved_shadow_train_mask):
                for entity_id, indices in entity_index_table.items():
                    saved_shadow_entity_mask[shadow_idx, entity_id] = torch.any(sample_mask[indices])
            unique_entity_ids = torch.unique(dataset.entity_ids)
            self.assertTrue(torch.equal(saved_shadow_entity_mask[:, unique_entity_ids].sum(dim=0), torch.full((dataset.n_entities,), 2, dtype=torch.int64)))

            metrics_path = (
                path_utils.metrics_dir(results_dir, scores_path, "entity", entity_audit_mode="all")
                / path_utils.metrics_pickle_name(scores_path, "entity")
            )
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "rb") as file:
                metrics = pickle.load(file)
            self.assertIn("AUC", metrics)
            self.assertIn("TPR@1%FPR", metrics)
            self.assertIn("TPR@0.1%FPR", metrics)
            self.assertEqual(metrics["n_audit_points"], dataset.n_entities)

            eval_config = run_mia_module.utils.Config({
                "res_dir": str(results_dir),
                "metrics_folders": [str(metrics_path.parent)],
            })
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                summaries = evaluation_module.run_evaluation(config=eval_config)

            self.assertEqual(len(summaries), 1)
            self.assertIn("Evaluation summary", stdout.getvalue())
            roc_plot_path = results_dir / f"average_roc_curves_{metrics_path.parent.stem}.png"
            self.assertTrue(roc_plot_path.exists())


if __name__ == "__main__":
    unittest.main()
