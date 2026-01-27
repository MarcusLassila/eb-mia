from accelerate.accelerate import AcceleratorLite
from data import data
from generative_models.ddpm import DDPM
from training.train_loop import TrainConfig, TrainLoop
from generative_models.vae import VAE
from . import train_split
import utils

import torch
from torch.utils.data import Subset

from pathlib import Path
import yaml

def get_train_config(config):
    return TrainConfig(
        batch_size=config.batch_size,
        simul_batch_size=config.simul_batch_size,
        epochs=config.epochs,
        epochs_per_checkpoint=config.epochs_per_checkpoint,
        lr=config.lr,
        weight_decay=config.weight_decay,
        ema_decay=config.ema_decay,
        grad_clip=config.grad_clip,
        autocast_dtype=config.autocast_dtype,
        lr_scheduler=config.lr_scheduler,
    )

def train_model(accelerator, config, savedir, dataset, train_mask, id_):
    '''Train a single model on a masked dataset split. Args: accelerator, config, savedir (Path), dataset, train_mask (torch.BoolTensor), id_ (int). Returns: None.'''
    channels, height, width = dataset[0].shape
    assert height == width
    train_indices = utils.mask_to_index(train_mask)
    nontrain_indices = utils.mask_to_index(~train_mask)
    val_size = int(config.val_frac * len(dataset))
    val_indices = nontrain_indices[:val_size]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_config = get_train_config(config)
    savepath = savedir / Path(f"{config.dataset}-{config.model}-{id_}.pth")
    match config.model:
        case "DDPM":
            image_dim = dataset[0].shape
            model_config = {
                "image_dim": image_dim,
                "time_steps": 1000,
                "beta_schedule": "linear",
                "base_channels": config.base_channels,
                "channel_mult": config.channel_mult,
                "n_attention_heads": config.n_attention_heads,
                "attention_resolutions": config.attention_resolutions,
                "dropout": config.dropout,
                "resample_with_conv": True,
                "use_sdpa": True,
            }
            model = DDPM(**model_config)
        case "VAE":
            model_config = {
                "in_ch": channels,
                "in_dim": height,
                "latent_dim": config.latent_dim,
                "n_rsamples": 1,
            }
            model = VAE(**model_config)
        case _:
            raise ValueError(f"No generative model named {config.model}.")

    TrainLoop(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_config=train_config,
        model_config=model_config,
        accelerator=accelerator,
        savepath=savepath,
    ).train()

def train_model_from_indices_file(
        accelerator,
        config,
        savedir,
        train_indices_path,
        id_,
    ):
    '''Train a model from explicit training indices. Args: accelerator, config, savedir (Path), train_indices_path (Path), id_ (int). Returns: None.'''
    dataset = getattr(data, config.dataset)(data_dir=config.data_dir, transform=None)  # Use default transforms
    train_indices = torch.tensor(train_split.load_indices(train_indices_path), dtype=torch.long)
    train_mask = utils.index_to_mask(train_indices, len(dataset))
    train_model(accelerator, config, savedir, dataset, train_mask, id_)


def main(config_file, id_):
    '''Load config and train a model from explicit indices. Args: config_file (str), id_ (int). Returns: None.'''
    root = utils.get_root()
    with open(f"{root}/training/configs/{config_file}.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    config.id = id_
    dataloader_config = getattr(config, "dataloader_config", None)
    accelerator = AcceleratorLite(torch_compile=config.torch_compile, base_seed=42*id_, dataloader_config=dataloader_config)
    savedir = Path(config.save_dir)
    savedir.mkdir(parents=True, exist_ok=True)
    train_indices_path = getattr(config, "train_indices_path", None)
    if train_indices_path is None:
        raise ValueError("must provide a path to the train indices.")
    train_indices_path = Path(train_indices_path)
    if not train_indices_path.is_absolute() and root is not None:
        train_indices_path = Path(root) / train_indices_path
    train_model_from_indices_file(
        accelerator=accelerator,
        config=config,
        savedir=savedir,
        train_indices_path=train_indices_path,
        id_=id_,
    )

if __name__ == "__main__":
    import argparse
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_train_celeba",
    )
    args = parser.parse_args()
    main(args.config, args.id)
