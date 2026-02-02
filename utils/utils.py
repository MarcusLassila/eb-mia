from pathlib import Path
import re
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

def parse_properties_from_checkpoint_path(path):
    '''Parse properties from a checkpoint path. Args: path (str|Path). Returns: dict.'''
    stem = Path(path).stem
    epoch_match = re.search(r"-epoch(?P<epoch>\d+)$", stem)
    epoch = None
    if epoch_match is not None:
        epoch = int(epoch_match.group("epoch"))
        stem = stem[:epoch_match.start()]
    match = re.match(
        r"^(?P<model>[^-]+)-(?P<split>.+)-sz(?P<size>\d+)(?P<gray>-gray)?(?P<suffix>(?:-.+)?)$",
        stem,
    )
    assert match is not None
    model_name = match.group("model")
    split_stem = match.group("split")
    size = int(match.group("size"))
    gray = match.group("gray") is not None
    suffix = match.group("suffix")
    suffix = suffix[1:] if suffix else ""

    properties = {
        "model": model_name,
        "dataset": None,
        "size": size,
        "gray": gray,
        "split": split_stem,
        "split_mode": None,
        "seed": None,
        "complement": None,
        "suffix": suffix,
        "epoch": epoch,
    }

    rand_match = re.match(
        r"^(?P<dataset>.+)-rand-f(?P<fraction>[^-]+)-s(?P<seed>\d+)(?P<comp>-comp)?$",
        split_stem,
    )
    ent_match = re.match(
        r"^(?P<dataset>.+)-ent-f(?P<entity_fraction>[^-]+)-p(?P<per_entity_fraction>[^-]+)-s(?P<seed>\d+)(?P<comp>-comp)?$",
        split_stem,
    )
    if rand_match is not None:
        properties.update({
            "dataset": rand_match.group("dataset"),
            "split_mode": "random",
            "fraction": float(rand_match.group("fraction").replace("p", ".")),
            "seed": int(rand_match.group("seed")),
            "complement": rand_match.group("comp") is not None,
        })
    else:
        assert ent_match is not None
        properties.update({
            "dataset": ent_match.group("dataset"),
            "split_mode": "entity",
            "entity_fraction": float(ent_match.group("entity_fraction").replace("p", ".")),
            "per_entity_fraction": float(ent_match.group("per_entity_fraction").replace("p", ".")),
            "seed": int(ent_match.group("seed")),
            "complement": ent_match.group("comp") is not None,
        })
    return properties

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
