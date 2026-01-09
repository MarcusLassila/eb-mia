from accelerate.accelerate import AcceleratorLite
from data import data
from ddpm.ddpm import DDPM
from vae.train import train_vae
from vae.vae import VAE
import utils

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from pathlib import Path
import yaml

def train_model(accelerator, config, savedir, dataset, train_mask, id_):
    channels, height, width = dataset[0].shape
    assert height == width
    train_indices = utils.mask_to_index(train_mask)
    nontrain_indices = utils.mask_to_index(~train_mask)
    val_size = int(config.val_frac * len(dataset))
    val_indices = nontrain_indices[:val_size]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    match config.model:
        case "DDPM":
            beta = torch.linspace(start=1e-4, end=0.02, steps=1000)
            model = DDPM(
                beta=beta,
                channel_mult=config.channel_mult,
                image_dim=dataset[0].shape,
                base_channels=config.base_channels,
                dropout=config.dropout,
                resample_with_conv=True,
                accelerator=accelerator,
            )
            model.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=config.batch_size,
                lr=config.lr,
                n_epochs=config.epochs,
                savepath=savedir/Path(f"{config.dataset}-{config.model}-{id_}.pth"),
                simul_batch_size=config.simul_batch_size,
                grad_clip=1.0,
                epochs_per_checkpoint=config.epochs_per_checkpoint,
            )
        case "VAE":
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            model = VAE(in_ch=channels, in_dim=height, latent_dim=config.latent_dim)
            train_vae(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=config.epochs,
                device=accelerator.device,
                lr=config.lr,
                savepath=savedir/Path(f"{config.dataset}-{config.model}-{id_}.pth"),
                weight_decay=config.weight_decay,
            )

def train_model_pair(
        accelerator,
        config,
        savedir,
    ):
    dataset = getattr(data, config.dataset)(data_dir=config.data_dir, transform=None)  # Use default transforms
    n_indices = len(dataset)

    if accelerator.is_master_process:
        train_mask = torch.rand(n_indices, device=accelerator.device) > 0.5
    else:
        train_mask = torch.empty(n_indices, dtype=torch.bool, device=accelerator.device)
    accelerator.broadcast(train_mask)
    for id_, mask in (config.id, train_mask), (config.id + 1, ~train_mask):
        train_model(accelerator, config, savedir, dataset, mask, id_)

def main(config_file, id_):
    root = utils.get_root()
    with open(f"{root}/mia/configs/{config_file}.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    config.id = id_
    accelerator = AcceleratorLite(torch_compile=config.torch_compile, base_seed=42*id_)
    savedir = Path(f"./trained_models")
    savedir.mkdir(parents=True, exist_ok=True)
    train_model_pair(
        accelerator=accelerator,
        config=config,
        savedir=savedir,
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
