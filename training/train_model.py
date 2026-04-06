from accelerate.accelerate import AcceleratorLite
from data.utils import load_dataset
from generative_models.ddpm import DDPM
from generative_models.flow_matching import FlowMatching
from training.train_loop import TrainConfig, TrainLoop
from generative_models.vae import VAE
from . import train_split
import utils

import torch
from torch.utils.data import Subset

from pathlib import Path
import yaml

def get_train_config(config):
    '''
    Build the training-loop configuration from a parsed model config.
    Args:
        config (Config): Parsed training configuration.
    Returns:
        TrainConfig: Training-loop configuration.
    '''
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
        lr_scheduler_params=getattr(config, "lr_scheduler_params", {}),
    )

def get_train_and_val_datasets(dataset_name, image_resolution, data_dir, val_frac=None, grayscale=False, train_indices_path=None, checkpoint=None):
    '''
    Build train and validation subsets with deterministic validation transforms.
    Args:
        dataset_name (str): Dataset name passed to the dataset loader.
        image_resolution (int): Target image resolution.
        data_dir (Path | str): Dataset directory.
        val_frac (float | None): Validation fraction for fresh training runs.
        grayscale (bool): Whether to load grayscale images.
        train_indices_path (Path | None): Optional path to explicit train indices.
        checkpoint (dict | None): Optional checkpoint payload used for resume.
    Returns:
        tuple[Subset, Subset]: Training and validation dataset subsets.
    '''
    assert (train_indices_path is not None) or (checkpoint is not None)
    if checkpoint is None and val_frac is None:
        raise ValueError("Must specify validation fraction unless resuming training from a checkpoint")
    dataset = load_dataset(
        dataset_name=dataset_name,
        data_dir=str(data_dir),
        transform=None,
        size=image_resolution,
        grayscale=grayscale,
    ) # Use default transforms
    deterministic_dataset = load_dataset(
        dataset_name=dataset_name,
        data_dir=str(data_dir),
        transform=None,
        size=image_resolution,
        grayscale=grayscale,
        random_horizontal_flip=False, # Disable random augmentation for validation
    )
    if checkpoint is not None:
        if train_indices_path is None:
            train_indices = checkpoint["train_indices"]
            val_indices = checkpoint["val_indices"]
        else:
            train_indices = train_split.load_indices(train_indices_path, len_dataset=len(dataset))
            train_mask = utils.index_to_mask(torch.tensor(train_indices, dtype=torch.long), len(dataset))
            nontrain_indices = utils.mask_to_index(~train_mask)
            val_size = len(checkpoint["val_indices"])
            val_indices = nontrain_indices[:val_size]
    else:
        train_indices = train_split.load_indices(train_indices_path, len_dataset=len(dataset))
        train_mask = utils.index_to_mask(torch.tensor(train_indices, dtype=torch.long), len(dataset))
        train_indices = utils.mask_to_index(train_mask)
        nontrain_indices = utils.mask_to_index(~train_mask)
        val_size = int(val_frac * len(dataset))
        val_indices = nontrain_indices[:val_size]
    if not torch.is_tensor(train_indices):
        train_indices = torch.tensor(train_indices, dtype=torch.long)
    if not torch.is_tensor(val_indices):
        val_indices = torch.tensor(val_indices, dtype=torch.long)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(deterministic_dataset, val_indices)
    return train_dataset, val_dataset

def build_checkpoint_savepath(config, savedir, split_stem, is_gray):
    '''
    Build the checkpoint save path.
    Args:
        config (Config): Training configuration.
        savedir (Path): Directory used for checkpoint output.
        split_stem (str): Train-split stem used in the filename.
        is_gray (bool): Whether the checkpoint stores grayscale images.
    Returns:
        Path: Checkpoint save path.
    '''
    gray_token = "-gray" if is_gray else ""
    suffix = f"-{config.suffix}" if config.suffix else ""
    filename = f"{config.model}-{split_stem}-sz{config.image_resolution}{gray_token}{suffix}.pth"
    return Path(savedir) / filename

def train_model_from_scratch(
        accelerator,
        config,
        data_dir,
        savedir,
        train_indices_path,
    ):
    '''
    Train a model from config and explicit train indices.
    Args:
        accelerator (AcceleratorLite): Training accelerator.
        config (Config): Training configuration.
        data_dir (Path): Dataset directory.
        savedir (Path): Checkpoint output directory.
        train_indices_path (Path): Path to the training indices file.
    Returns:
        None
    '''
    train_dataset, val_dataset = get_train_and_val_datasets(
        dataset_name=config.dataset,
        image_resolution=config.image_resolution,
        data_dir=data_dir,
        val_frac=config.val_frac,
        grayscale=getattr(config, "grayscale", False),
        train_indices_path=train_indices_path,
    )
    image_dim = train_dataset[0].shape
    channels, height, width = image_dim
    assert height == width
    assert height == config.image_resolution
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
                "n_res_blocks_per_level": config.n_res_blocks_per_level,
                "n_attention_heads": config.n_attention_heads,
                "channels_per_head": config.channels_per_head,
                "attention_resolutions": config.attention_resolutions,
                "dropout": config.dropout,
                "resample_with_conv": True,
                "use_sdpa": config.use_sdpa,
            }
            model = DDPM(**model_config)
        case "FlowMatching":
            model_config = {
                "image_dim": image_dim,
                "std_min": config.std_min,
                "base_channels": config.base_channels,
                "channel_mult": config.channel_mult,
                "n_res_blocks_per_level": config.n_res_blocks_per_level,
                "n_attention_heads": config.n_attention_heads,
                "channels_per_head": config.channels_per_head,
                "attention_resolutions": config.attention_resolutions,
                "dropout": config.dropout,
                "resample_with_conv": True,
                "use_sdpa": config.use_sdpa,
            }
            model = FlowMatching(**model_config)
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
    '''
    Resume model training from checkpoint state.
    Args:
        accelerator (AcceleratorLite): Training accelerator.
        config (Config | None): Optional training configuration override.
        savedir (Path): Checkpoint output directory.
        data_dir (Path): Dataset directory.
        checkpoint (dict): Loaded checkpoint payload.
        checkpoint_path (Path): Source checkpoint path.
        train_indices_path (Path | None): Optional replacement training indices path.
    Returns:
        None
    '''
    properties = utils.parse_properties_from_checkpoint_path(checkpoint_path)
    train_dataset, val_dataset = get_train_and_val_datasets(
        dataset_name=properties["dataset"],
        image_resolution=properties["size"],
        data_dir=data_dir,
        val_frac=None,
        grayscale=properties["gray"],
        train_indices_path=train_indices_path,
        checkpoint=checkpoint,
    )
    if config is None:
        train_config = TrainConfig(**checkpoint["train_config"])
    else:
        train_config = get_train_config(config)
    model_config = checkpoint["model_config"]
    match properties["model"]:
        case "DDPM":
            model = DDPM(**model_config)
        case "FlowMatching":
            model = FlowMatching(**model_config)
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

def main(config_file=None, suffix="", train_indices_path=None, checkpoint_path=None, data_dir=None, save_dir=None, torch_compile=False):
    '''
    Load config and train from indices or resume from a checkpoint.
    Args:
        config_file (str | None): Training config name without the `.yaml` suffix.
        suffix (str): Optional suffix appended to saved checkpoint names.
        train_indices_path (str | None): Optional training indices path.
        checkpoint_path (str | None): Optional checkpoint path to resume from.
        data_dir (str | Path): Dataset directory.
        save_dir (str | Path): Checkpoint output directory.
        torch_compile (bool): Whether to enable `torch.compile`.
    Returns:
        None
    '''
    root = utils.get_root()
    if data_dir is None or save_dir is None:
        raise ValueError("must provide data_dir and save_dir.")

    data_dir = utils.resolve_path(data_dir, root)
    save_dir = utils.resolve_path(save_dir, root)
    save_dir.mkdir(parents=True, exist_ok=True)
    if train_indices_path is not None:
        train_indices_path = utils.resolve_path(train_indices_path, root)
    if checkpoint_path is not None:
        checkpoint_path = utils.resolve_path(checkpoint_path, root)

    if config_file is None:
        config = None
    else:
        config_path = utils.resolve_path(Path("training") / "configs" / f"{config_file}.yaml", root)
        with open(config_path, "r") as file:
            config = utils.Config(yaml.safe_load(file))
        config.suffix = suffix

    if checkpoint_path is None:
        if config is None:
            raise ValueError("must provide a train config unless resuming from a checkpoint.")
        if train_indices_path is None:
            raise ValueError("must provide a path to the train indices unless resuming from a checkpoint.")
        accelerator = AcceleratorLite(
            torch_compile=bool(config.torch_compile or torch_compile),
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
            accelerator = AcceleratorLite(torch_compile=torch_compile, base_seed=0, dataloader_config=None)
        else:
            accelerator = AcceleratorLite(
                torch_compile=bool(config.torch_compile or torch_compile),
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
    parser.add_argument(
        "--torch-compile",
        action="store_true",
    )
    args = parser.parse_args()
    if args.checkpoint_path is None and (args.config is None or args.train_indices_path is None):
        parser.error("--config and --train-indices-path are required unless --checkpoint-path is provided.")
    main(args.config, args.suffix, args.train_indices_path, args.checkpoint_path, args.data_dir, args.save_dir, args.torch_compile)
