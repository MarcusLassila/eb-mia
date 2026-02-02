import torch

from utils import load_checkpoint, parse_properties_from_checkpoint_path

from .ddpm import DDPM
from .vae import VAE

def load_model(path: str, device: torch.device):
    ''' Load a generative model and its training indices from a checkpoint.

    Args:
        path: Checkpoint file path.
        device: Target torch device.

    Returns:
        Tuple[AbstractGenerativeModel, torch.Tensor]: Loaded model and train indices.
    '''
    checkpoint = load_checkpoint(path, device)
    properties = parse_properties_from_checkpoint_path(path)
    model_type = properties["model"]
    match model_type:
        case "DDPM":
            model = DDPM(**checkpoint["model_config"])
        case "VAE":
            model = VAE(**checkpoint["model_config"])
        case _:
            raise ValueError(f"Unsupported model: {model_type}")
    model.move_to(device)
    model.network.load_state_dict(checkpoint["network_state_dict"])
    model.network.eval()
    train_indices = checkpoint["train_indices"]
    assert model.image_size == properties["size"]
    return model, train_indices
