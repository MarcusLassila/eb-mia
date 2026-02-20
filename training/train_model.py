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

def resolve_path(path_str, root):
    '''Resolve a path string to absolute path using repository root for relative inputs. Args: path_str (str|Path), root (str|Path|None). Returns: Path.'''
    path = Path(path_str)
    if not path.is_absolute() and root is not None:
        return Path(root) / path
    return path

def train_model_from_scratch(
        accelerator,
        config,
        data_dir,
        savedir,
        train_indices_path,
    ):
    '''Train a model from config and explicit train indices. Args: accelerator, config, data_dir (Path), savedir (Path), train_indices_path (Path). Returns: None.'''
    dataset = load_dataset(
        config.dataset,
        data_dir=str(data_dir),
        transform=None,
        size=config.image_resolution,
        grayscale=getattr(config, "grayscale", False),
    )  # Use default transforms
    image_dim = dataset[0].shape
    channels, height, width = image_dim
    assert height == width
    assert height == config.image_resolution
    train_indices = train_split.load_indices(train_indices_path)
    train_mask = utils.index_to_mask(torch.tensor(train_indices, dtype=torch.long), len(dataset))
    train_indices = utils.mask_to_index(train_mask)
    nontrain_indices = utils.mask_to_index(~train_mask)
    val_size = int(config.val_frac * len(dataset))
    val_indices = nontrain_indices[:val_size]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_config = get_train_config(config)
    savepath = build_checkpoint_savepath(config, savedir, train_indices_path.stem, channels == 1)
    match config.model:
        case "DDPM":
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

    param_counts = utils.count_params(model.network)
    accelerator.print(f"Total params:     {param_counts['n_params']:_}")
    accelerator.print(f"Trainable params: {param_counts['n_trainable_params']:_}")

    TrainLoop(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_config=train_config,
        model_config=model_config,
        accelerator=accelerator,
        savepath=savepath,
    ).train()

def train_model_from_checkpoint(
        accelerator,
        config,
        savedir,
        data_dir,
        checkpoint,
        checkpoint_path,
        train_indices_path=None,
    ):
    '''Resume model training from checkpoint state. Args: accelerator, config (Config|None), savedir (Path), data_dir (Path), checkpoint (dict), checkpoint_path (Path), train_indices_path (Path|None). Returns: None.'''
    properties = utils.parse_properties_from_checkpoint_path(checkpoint_path)
    dataset = load_dataset(
        properties["dataset"],
        data_dir=str(data_dir),
        transform=None,
        size=properties["size"],
        grayscale=properties["gray"],
    )  # Use default transforms

    if train_indices_path is None:
        train_indices = checkpoint["train_indices"]
        val_indices = checkpoint["val_indices"]
    else:
        train_indices = train_split.load_indices(train_indices_path)
        train_mask = utils.index_to_mask(torch.tensor(train_indices, dtype=torch.long), len(dataset))
        nontrain_indices = utils.mask_to_index(~train_mask)
        val_size = len(checkpoint["val_indices"])
        if config is not None:
            val_size = int(config.val_frac * len(dataset))
        val_indices = nontrain_indices[:val_size]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    if config is None:
        train_config = TrainConfig(**checkpoint["train_config"])
    else:
        train_config = get_train_config(config)
    model_config = checkpoint["model_config"]
    match properties["model"]:
        case "DDPM":
            model = DDPM(**model_config)
        case "VAE":
            model = VAE(**model_config)
        case _:
            raise ValueError(f"No generative model named {properties['model']}.")

    param_counts = utils.count_params(model.network)
    accelerator.print(f"Total params:     {param_counts['n_params']:_}")
    accelerator.print(f"Trainable params: {param_counts['n_trainable_params']:_}")
    checkpoint_epoch = properties["epoch"]
    save_stem = checkpoint_path.stem
    if checkpoint_epoch is not None:
        save_stem = save_stem.removesuffix(f"-epoch{checkpoint_epoch}")
    savepath = Path(savedir) / f"{save_stem}{checkpoint_path.suffix}"

    TrainLoop(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_config=train_config,
        model_config=model_config,
        accelerator=accelerator,
        savepath=savepath,
        resume_checkpoint=checkpoint,
    ).train()

def main(config_file=None, suffix="", train_indices_path=None, checkpoint_path=None, data_dir=None, save_dir=None):
    '''Load config and train from indices or resume from checkpoint. Args: config_file (str|None), suffix (str), train_indices_path (str|None), checkpoint_path (str|None), data_dir (str|Path), save_dir (str|Path). Returns: None.'''
    root = utils.get_root()
    if data_dir is None or save_dir is None:
        raise ValueError("must provide data_dir and save_dir.")

    data_dir = resolve_path(data_dir, root)
    save_dir = resolve_path(save_dir, root)
    save_dir.mkdir(parents=True, exist_ok=True)
    if train_indices_path is not None:
        train_indices_path = resolve_path(train_indices_path, root)
    if checkpoint_path is not None:
        checkpoint_path = resolve_path(checkpoint_path, root)

    if config_file is None:
        config = None
    else:
        config_path = resolve_path(Path("training") / "configs" / f"{config_file}.yaml", root)
        with open(config_path, "r") as file:
            config = utils.Config(yaml.safe_load(file))
        config.suffix = suffix

    if checkpoint_path is None:
        if config is None:
            raise ValueError("must provide a train config unless resuming from a checkpoint.")
        if train_indices_path is None:
            raise ValueError("must provide a path to the train indices unless resuming from a checkpoint.")
        accelerator = AcceleratorLite(
            torch_compile=config.torch_compile,
            base_seed=getattr(config, "seed", 0),
            dataloader_config=getattr(config, "dataloader_config", None),
        )
        accelerator.print(config)
        train_model_from_scratch(
            accelerator=accelerator,
            config=config,
            data_dir=data_dir,
            savedir=save_dir,
            train_indices_path=train_indices_path,
        )
    else:
        checkpoint = utils.load_checkpoint(str(checkpoint_path), torch.device("cpu"))
        if config is None:
            accelerator = AcceleratorLite(torch_compile=False, base_seed=0, dataloader_config=None)
        else:
            accelerator = AcceleratorLite(
                torch_compile=config.torch_compile,
                base_seed=getattr(config, "seed", 0),
                dataloader_config=getattr(config, "dataloader_config", None),
            )
            accelerator.print(config)
        train_model_from_checkpoint(
            accelerator=accelerator,
            config=config,
            savedir=save_dir,
            data_dir=data_dir,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
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
        default=None,
    )
    parser.add_argument(
        "--train-indices-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    if args.checkpoint_path is None and (args.config is None or args.train_indices_path is None):
        parser.error("--config and --train-indices-path are required unless --checkpoint-path is provided.")
    main(args.config, args.suffix, args.train_indices_path, args.checkpoint_path, args.data_dir, args.save_dir)
