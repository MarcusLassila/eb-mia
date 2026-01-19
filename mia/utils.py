from ddpm.ddpm import DDPM
from vae.vae import VAE

import numpy as np
import torch

from pathlib import Path
import subprocess
import yaml

class Config:

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def __str__(self):
        return yaml.dump(self.__dict__)

def get_root():
    ''' Return path to the root of the repository. '''
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
        return root.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return None

def get_dataset_and_model_from_path(path):
    dataset, model, *_ = Path(path).stem.split("-") # Assumes model checkpoint is saved as dataset-model-otherstuff.pth
    return dataset, model

def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)

def load_model(path: str, device: torch.device, n_loss_samples: int = 20):
    checkpoint = load_checkpoint(path, device)
    _, model_type = get_dataset_and_model_from_path(path)
    match model_type:
        case "DDPM":
            model = DDPM(**checkpoint["model_config"])
            model.to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            train_indices = checkpoint["train_indices"]
        case "VAE":
            model = VAE(**checkpoint["model_config"])
            model.n_rsamples = n_loss_samples
            model.to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            train_indices = checkpoint["train_indices"]
        case _:
            raise ValueError(f"Unsupported model: {model_type}")
    model.eval()
    return model, train_indices

def get_train_indices(path: str):
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    return checkpoint["train_indices"]

def mask_to_index(mask: torch.Tensor):
    return mask.nonzero(as_tuple=True)[0]

def index_to_mask(index: torch.Tensor, n_indices):
    mask = torch.zeros(n_indices, dtype=torch.bool)
    mask[index] = True
    return mask

def count_params(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'n_params': n_params, 'n_trainable_params': n_trainable}
