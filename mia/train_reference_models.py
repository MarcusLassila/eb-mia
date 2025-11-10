from data import data
from vae.train import train_vae
from vae.vae import VAE
import utils

import torch
from torch.utils.data import DataLoader, Subset

from pathlib import Path
import yaml

def train_ref_model_pair(
        config,
        device,
        savedir,
    ):
    dataset = getattr(data, config.dataset)()
    channels, height, width = dataset[0].shape
    assert height == width
    n_indices = len(dataset)
    train_mask = torch.rand(n_indices) > 0.5
    for i, mask in (config.id, train_mask), (config.id + 1, ~train_mask):
        train_indices = utils.mask_to_index(mask)
        nontrain_indices = utils.mask_to_index(~mask)
        val_size = int(config.val_frac * len(dataset))
        val_indices = nontrain_indices[torch.randperm(n=nontrain_indices.shape[0])[:val_size]]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        model = globals()[config.model](in_ch=channels, in_dim=height, latent_dim=config.latent_dim)
        train_vae(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=config.epochs,
            device=device,
            lr=config.lr,
            savepath=savedir/Path(f"{config.dataset}-{config.model}-{i}.pth"),
            weight_decay=config.weight_decay,
        )

def main(id_):
    root = utils.get_root()
    with open(f"{root}/mia/config_train.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    config.id = id_
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = Path(f"{root}/trained_models/ref_models")
    savedir.mkdir(parents=True, exist_ok=True)
    train_ref_model_pair(
        config=config,
        device=device,
        savedir=savedir,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    main(args.id)
