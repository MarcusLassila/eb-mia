from data import data
from vae import train, vae

import torch
from torch.utils.data import DataLoader, random_split

from pathlib import Path

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500
LATENT_DIM = 128
LR = 1e-3
NUM_REF_MODELS = 1
SAVEDIR = "./trained_models/VAE/"

def main():
    dataset = data.CIFAR10()
    channels, height, width = dataset[0].shape
    assert height == width
    Path(SAVEDIR).mkdir(parents=True, exist_ok=True)
    for i_model in range(NUM_REF_MODELS):
        train_dataset, test_dataset = random_split(
            dataset,
            lengths=[len(dataset) // 2, len(dataset) // 2],
        )
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = vae.VAE(in_channels=channels, in_dim=height, latent_dim=LATENT_DIM)
        train.train_vae(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader,
            epochs=EPOCHS,
            device=DEVICE,
            lr=LR,
            savepath=SAVEDIR+f"{dataset.__class__.__name__}_model_{i_model}.pth"
        )

if __name__ == "__main__":
    main()
