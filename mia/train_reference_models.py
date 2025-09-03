from data import data
from vae import train, vae
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
        dataset_name,
        model_name,
        batch_size,
        device,
        epochs,
        latent_dim,
        n_ref_models,
        lr,
        weight_decay,
        savedir,
    ):
    dataset = getattr(data, dataset_name)()
    channels, height, width = dataset[0].shape
    assert height == width
    Path(savedir).mkdir(parents=True, exist_ok=True)
    train_masks = partitioned_train_masks(n_models=n_ref_models, n_indices=len(dataset))
    for i_model in range(n_ref_models):
        mask = train_masks[i_model]

        train_indices = utils.mask_to_index(mask)
        nontrain_indices = utils.mask_to_index(~mask)
        val_size = int(0.05 * len(dataset))
        val_indices = nontrain_indices[torch.randperm(n=nontrain_indices.shape[0])[:val_size]]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = vae.VAE(in_ch=channels, in_dim=height, latent_dim=latent_dim) # hardcoded for now
        assert model.__class__.__name__ == model_name
        train.train_vae(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            savepath=savedir+f"{dataset_name}_model_{i_model}.pth"
        )

def main():
    root = utils.get_root()
    with open(f"{root}/mia/config_train.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = f"{root}/trained_models/{config.model}"
    train_ref_models(
        dataset_name=config.dataset,
        model_name=config.model,
        batch_size=config.batch_size,
        device=device,
        epochs=config.epochs,
        latent_dim=config.latent_dim,
        n_ref_models=config.n_ref_models,
        lr=config.lr,
        weight_decay=config.weight_decay,
        savedir=savedir,
    )

if __name__ == "__main__":
    main()
