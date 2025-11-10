from data import data
from vae.train import train_vae
import utils

import torch
from torch.utils.data import DataLoader, Subset

from pathlib import Path
import yaml

def train_target_model(
        config,
        device,
        savedir,
    ):
    dataset = getattr(data, config.dataset)()
    channels, height, width = dataset[0].shape
    assert height == width
    n_indices = len(dataset)
    mask = torch.rand(n_indices) > 0.5
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
        savepath=savedir/Path(f"{config.dataset}-{config.model}-{config.id}.pth"),
        weight_decay=config.weight_decay,
    )

def main(model_id):
    root = utils.get_root()
    with open(f"{root}/mia/config_train.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    config.id = model_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = Path(f"{root}/trained_models/target_models")
    savedir.mkdir(parents=True, exist_ok=True)
    train_target_model(
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
