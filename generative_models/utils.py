import torch

from utils import get_dataset_and_model_from_path, load_checkpoint

from .ddpm import create_ddpm_noise_model, DDPM
from .vae import VAE, VAE_Network

def load_model(path: str, device: torch.device):
    ''' Load a model (network), the generative class (e.g. DDPM or VAE), and its training indices from a checkpoint.

    Args:
        path: Checkpoint file path.
        device: Target torch device.

    Returns:
        Tuple[torch.nn.Module, AbstractGenerativeModel, torch.Tensor]: Loaded model and train indices.
    '''
    checkpoint = load_checkpoint(path, device)
    _, model_type = get_dataset_and_model_from_path(path)
    match model_type:
        case "DDPM":
            model = create_ddpm_noise_model(**checkpoint["configs"]["model_config"])
            generative_class = DDPM(**checkpoint["configs"]["generative_class_config"])
        case "VAE":
            model = VAE_Network(**checkpoint["configs"]["model_config"])
            generative_class = VAE(**checkpoint["configs"]["generative_class_config"])
            
        case _:
            raise ValueError(f"Unsupported model: {model_type}")
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    generative_class.move_to(device)
    train_indices = checkpoint["train_indices"]
    return model, generative_class, train_indices
