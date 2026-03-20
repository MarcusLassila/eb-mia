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

def resolve_path(path_str, root):
    '''Resolve a path string to absolute path using repository root for relative inputs. Args: path_str (str|Path), root (str|Path|None). Returns: Path.'''
    path = Path(path_str)
    if not path.is_absolute() and root is not None:
        return Path(root) / path
    return path

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

    smpl_match = re.match(
        r"^(?P<dataset>.+)-smpl-f(?P<fraction>\d+p\d+)-s(?P<seed>\d+)(?P<comp>-comp)?$",
        split_stem,
    )
    ent_match = re.match(
        r"^(?P<dataset>.+)-ent-f(?P<entity_fraction>\d+p\d+)-p(?P<per_entity_fraction>\d+p\d+)-s(?P<seed>\d+)(?P<comp>-comp)?$",
        split_stem,
    )
    if smpl_match is not None:
        properties.update({
            "dataset": smpl_match.group("dataset"),
            "split_mode": "sample",
            "fraction": float(smpl_match.group("fraction").replace("p", ".")),
            "seed": int(smpl_match.group("seed")),
            "complement": smpl_match.group("comp") is not None,
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


def has_torch_compile_wrapped_state_dict(state_dict):
    '''Return True if state_dict contains torch.compile wrapped keys. Args: state_dict (Mapping). Returns: bool.'''
    return any(key.startswith("_orig_mod.") for key in state_dict)


def unwrap_torch_compile_state_dict(state_dict):
    '''Return state_dict with torch.compile wrapper prefix removed from keys. Args: state_dict (Mapping). Returns: Mapping.'''
    unwrapped_state_dict = state_dict.__class__()
    for key, value in state_dict.items():
        unwrapped_key = key.removeprefix("_orig_mod.")
        if unwrapped_key in unwrapped_state_dict:
            raise ValueError(f"state_dict key collision after unwrapping: {unwrapped_key}")
        unwrapped_state_dict[unwrapped_key] = value
    return unwrapped_state_dict


def unwrap_checkpoint_state_dicts(checkpoint):
    '''Return checkpoint with known model state dicts unwrapped from torch.compile prefix. Args: checkpoint (dict). Returns: dict.'''
    unwrapped_checkpoint = dict(checkpoint)
    state_dict_keys = ("network_state_dict", "raw_network_state_dict", "ema_network_state_dict")
    for state_dict_key in state_dict_keys:
        if state_dict_key in unwrapped_checkpoint:
            unwrapped_checkpoint[state_dict_key] = unwrap_torch_compile_state_dict(unwrapped_checkpoint[state_dict_key])
    return unwrapped_checkpoint


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
