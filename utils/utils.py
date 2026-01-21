from pathlib import Path
import subprocess
import yaml

import torch

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
    ''' Extract dataset and model names from a checkpoint path.

    Args:
        path: Checkpoint path (str or Path).

    Returns:
        Tuple[str, str]: Dataset name and model name.
    '''
    dataset, model, *_ = Path(path).stem.split("-")
    return dataset, model

def load_checkpoint(path: str, device: torch.device):
    ''' Load a model checkpoint to a target device.

    Args:
        path: Checkpoint file path.
        device: Target torch device.

    Returns:
        dict: Loaded checkpoint.
    '''
    return torch.load(path, map_location=device)

def get_train_indices(path: str):
    ''' Load training indices from a checkpoint.

    Args:
        path: Checkpoint file path.

    Returns:
        torch.Tensor: Training indices.
    '''
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
