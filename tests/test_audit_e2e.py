import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Subset

from accelerate.accelerate import AcceleratorLite
from data.data import EntityDataset
from generative_models.ddpm import DDPM
from mia import path_utils
from mia import run_audit as run_audit_module
from mia import run_mia as run_mia_module
from training.train_loop import TrainConfig, TrainLoop
from training import train_split


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


class TestAuditEndToEnd(unittest.TestCase):
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
            epochs=1,
            epochs_per_checkpoint=1,
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
        return savepath.with_name(f"{savepath.stem}-epoch1{savepath.suffix}")

    def test_entity_audit_runs_end_to_end_with_base_scores(self):
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

            checkpoint_paths = []
            for seed in range(5):
                split_path = train_split.create_entity_subset(
                    dataset_name="CelebA2",
                    entity_ids=dataset.entity_ids,
                    entity_fraction=entity_fraction,
                    per_entity_fraction=per_entity_fraction,
                    seed=seed,
                    output_dir=splits_dir,
                )
                savepath = checkpoints_dir / f"DDPM-{split_path.stem}-sz4.pth"
                checkpoint_paths.append(
                    self._train_ddpm_checkpoint(
                        dataset=dataset,
                        train_indices=train_split.load_indices(split_path),
                        savepath=savepath,
                        seed=seed,
                    )
                )

            target_path = checkpoint_paths[0]
            shadow_paths = checkpoint_paths[1:]

            mia_config = run_mia_module.utils.Config({
                "dataset": "CelebA2",
                "data_dir": str(tmpdir_path),
                "batch_size": 4,
                "round_robin": False,
                "res_dir": str(results_dir),
                "target_model_paths": [str(target_path)],
                "shadow_model_paths": [str(path) for path in shadow_paths],
                "attack": {"name": "BASE-off", "attack": "BASE", "offline": True, "prior": 0.5, "n_loss_samples": 1},
            })
            audit_config = run_audit_module.utils.Config({
                "dataset": "CelebA2",
                "data_dir": str(tmpdir_path),
                "res_dir": str(results_dir),
                "score_paths": [str(path_utils.scores_dir(results_dir, "BASE-off", target_path))],
                "attack": {"attack": "CompositeBASE", "prior": 0.5},
                "audit_mode": "entity",
                "mode": "all",
            })
            base_scores = torch.linspace(0.1, 0.9, steps=len(dataset), dtype=torch.float32)

            with (
                patch.object(run_mia_module, "load_dataset", return_value=dataset),
                patch.object(run_audit_module, "load_dataset", return_value=dataset),
                patch.object(
                    run_mia_module.attacks.BASE,
                    "run_attack",
                    return_value={
                        "score": base_scores,
                        "loss_sigs": None,
                        "shadow_train_mask": None,
                    },
                ),
            ):
                run_mia_module.run_mia(config=mia_config, device=torch.device("cpu"))
                run_audit_module.run_entity_audit(config=audit_config)

            scores_path = (
                path_utils.scores_dir(results_dir, "BASE-off", target_path)
                / path_utils.scores_pickle_name(target_path, "BASE-off")
            )
            self.assertTrue(scores_path.exists())
            with open(scores_path, "rb") as file:
                sample_scores = pickle.load(file)
            self.assertEqual(len(sample_scores["scores"]), len(dataset))
            self.assertEqual(len(sample_scores["train_mask"]), len(dataset))

            metrics_path = (
                path_utils.metrics_dir(results_dir, scores_path, "entity", entity_audit_mode="all")
                / path_utils.metrics_pickle_name(scores_path, "entity")
            )
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "rb") as file:
                metrics = pickle.load(file)
            self.assertIn("AUC", metrics)
            self.assertIn("TPR@1%FPR", metrics)
            self.assertIn("audit_config", metrics)
            self.assertIsInstance(metrics["audit_config"], dict)


if __name__ == "__main__":
    unittest.main()
