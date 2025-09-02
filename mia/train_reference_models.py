from data import data
from vae import train, vae
import utils

import torch
from torch.utils.data import DataLoader, Subset

from pathlib import Path

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500
LATENT_DIM = 128
LR = 1e-3
NUM_REF_MODELS = 10
SAVEDIR = "./trained_models/VAE/"

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

def main():
    dataset = data.CIFAR10()
    channels, height, width = dataset[0].shape
    assert height == width
    Path(SAVEDIR).mkdir(parents=True, exist_ok=True)
    train_masks = partitioned_train_masks(n_models=NUM_REF_MODELS, n_indices=len(dataset))
    for i_model in range(NUM_REF_MODELS):
        mask = train_masks[i_model]

        train_indices = utils.mask_to_index(mask)
        nontrain_indices = utils.mask_to_index(~mask)
        val_size = int(0.05 * len(dataset))
        val_indices = nontrain_indices[torch.randperm(n=nontrain_indices.shape[0])[:val_size]]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = vae.VAE(in_ch=channels, in_dim=height, latent_dim=LATENT_DIM)
        train.train_vae(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=EPOCHS,
            device=DEVICE,
            lr=LR,
            savepath=SAVEDIR+f"{dataset.__class__.__name__}_model_{i_model}.pth"
        )

if __name__ == "__main__":
    main()
