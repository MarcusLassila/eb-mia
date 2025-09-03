from vae import vae
import utils

import torch

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

def load_checkpoint(dataset: str, model_type: str, index_model: int, device: torch.device):
    root = utils.get_root()
    path = f"{root}/trained_models/{model_type}/{dataset}_model_{index_model}.pth"
    return torch.load(path, map_location=device)

def load_model(dataset: str, model_type: str, index_model: int, device: torch.device):
    checkpoint = load_checkpoint(dataset, model_type, index_model, device)
    match model_type:
        case "VAE":
            model = vae.VAE(
                in_ch=checkpoint["in_channels"],
                in_dim=checkpoint["in_dim"],
                latent_dim=checkpoint["latent_dim"],
            )
            model.to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            train_indices = checkpoint["train_indices"]
        case _:
            raise ValueError(f"Unsupported model: {model_type}")
    return model, train_indices

def get_train_indices(dataset: str, model_type: str, index_model: int):
    checkpoint = load_checkpoint(dataset, model_type, index_model, torch.device("cpu"))
    return checkpoint["train_indices"]

def mask_to_index(mask: torch.Tensor):
    return mask.nonzero(as_tuple=True)[0]

def index_to_mask(index: torch.Tensor, n_indices):
    mask = torch.zeros(n_indices, dtype=torch.bool)
    mask[index] = True
    return mask
