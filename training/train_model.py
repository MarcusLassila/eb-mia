from accelerate.accelerate import AcceleratorLite
from data.utils import load_dataset
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
        use_ema=bool(config.ema_decay),
        ema_decay=config.ema_decay,
        grad_clip=config.grad_clip,
        autocast_dtype=config.autocast_dtype,
        lr_scheduler=config.lr_scheduler,
    )

def build_checkpoint_savepath(config, savedir, split_stem, is_gray):
    '''Build checkpoint save path. Args: config, savedir (Path), split_stem (str), is_gray (bool). Returns: Path.'''
    gray_token = "-gray" if is_gray else ""
    suffix = f"-{config.suffix}" if config.suffix else ""
    filename = f"{config.model}-{split_stem}-sz{config.image_resolution}{gray_token}{suffix}.pth"
    return Path(savedir) / filename

def train_model(accelerator, config, savedir, dataset, train_mask, split_stem):
    '''Train a single model on a masked dataset split. Args: accelerator, config, savedir (Path), dataset, train_mask (torch.BoolTensor), split_stem (str). Returns: None.'''
    image_dim = dataset[0].shape
    channels, height, width = image_dim
    assert height == width
    assert height == config.image_resolution
    train_indices = utils.mask_to_index(train_mask)
    nontrain_indices = utils.mask_to_index(~train_mask)
    val_size = int(config.val_frac * len(dataset))
    val_indices = nontrain_indices[:val_size]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_config = get_train_config(config)
    savepath = build_checkpoint_savepath(config, savedir, split_stem, channels == 1)
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
                "channels_per_head": config.channels_per_head,
                "attention_resolutions": config.attention_resolutions,
                "dropout": config.dropout,
                "resample_with_conv": True,
                "use_sdpa": config.use_sdpa,
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
    ):
    '''Train a model from explicit training indices. Args: accelerator, config, savedir (Path), train_indices_path (Path). Returns: None.'''
    dataset = load_dataset(
        config.dataset,
        data_dir=config.data_dir,
        transform=None,
        size=config.image_resolution,
        grayscale=getattr(config, "grayscale", False),
    )  # Use default transforms
    split_stem = train_indices_path.stem
    train_indices = torch.tensor(train_split.load_indices(train_indices_path), dtype=torch.long)
    train_mask = utils.index_to_mask(train_indices, len(dataset))
    train_model(accelerator, config, savedir, dataset, train_mask, split_stem)


def main(config_file, suffix="", train_indices_path=None):
    '''Load config and train a model from explicit indices. Args: config_file (str), suffix (str). Returns: None.'''
    root = utils.get_root()
    with open(f"{root}/training/configs/{config_file}.yaml", "r") as file:
        config = yaml.safe_load(file)
    if train_indices_path is not None:
        config["train_indices_path"] = train_indices_path
    print(yaml.dump(config, sort_keys=False))
    config = utils.Config(config)
    config.suffix = suffix
    dataloader_config = getattr(config, "dataloader_config", None)
    seed = getattr(config, "seed", 0)
    accelerator = AcceleratorLite(torch_compile=config.torch_compile, base_seed=seed, dataloader_config=dataloader_config)
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
    )

if __name__ == "__main__":
    import argparse
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train-indices-path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args.config, args.suffix, args.train_indices_path)
