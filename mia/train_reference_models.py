from data import data
from tractable_ebm import bmm
from vae import vae, train as train_vae
import utils

import torch
from torch.utils.data import DataLoader, Subset

from pathlib import Path
import yaml

def partitioned_train_masks(n_models, n_indices):
    '''
    Partition the number of data point indices among a number of models such that:
    1. Every point is assigned to exactly half of the models.
    2. Model 2k+1 is assigned the complement of indices assigned to model 2k.
    '''
    assert n_models % 2 == 0
    index_masks = torch.zeros(size=(n_models, n_indices), dtype=torch.bool)
    for i in range(0, n_models, 2):
        rand_mask = torch.rand(n_indices) > 0.5
        index_masks[i, rand_mask] = True
        index_masks[i + 1, ~rand_mask] = True
    return index_masks

def train_ref_models(
        config,
        device,
        savedir,
    ):
    dataset = getattr(data, config.dataset)()
    channels, height, width = dataset[0].shape
    assert height == width
    Path(savedir).mkdir(parents=True, exist_ok=True)
    train_masks = partitioned_train_masks(n_models=config.n_ref_models, n_indices=len(dataset))
    for i_model in range(config.n_ref_models):
        mask = train_masks[i_model]

        train_indices = utils.mask_to_index(mask)
        nontrain_indices = utils.mask_to_index(~mask)
        val_size = int(0.05 * len(dataset))
        val_indices = nontrain_indices[torch.randperm(n=nontrain_indices.shape[0])[:val_size]]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        if config.model == "VAE":
            model = vae.VAE(in_ch=channels, in_dim=height, latent_dim=config.latent_dim)
            assert model.__class__.__name__ == "VAE"
            train_vae.train_vae(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=config.epochs,
                device=device,
                lr=config.lr,
                weight_decay=config.weight_decay,
                savepath=savedir+f"/{config.dataset}_model_{i_model}.pth"
            )
        elif config.model == "BMM":
            model = bmm.BernoulliMixtureModel(in_dim=(channels, height, width), n_mixtures=config.n_mixtures)
            bmm.train_bmm(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=config.epochs,
                device=device,
                savepath=savedir+f"/{config.dataset}_model_{i_model}.pth",
                lr=config.lr,
            )
        else:
            raise ValueError(f"Unavailable model: {config.model}")

def main():
    root = utils.get_root()
    with open(f"{root}/mia/config_train.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = f"{root}/trained_models/{config.model}"
    train_ref_models(
        config=config,
        device=device,
        savedir=savedir,
    )

if __name__ == "__main__":
    main()
