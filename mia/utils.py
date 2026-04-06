import torch
import pickle

def indices_of_shadow_models(index_target, n_models):
    '''
    Return shadow indices excluding the target and its complement.
    Args:
        index_target (int): Index of the target model.
        n_models (int): Total number of models.
    Returns:
        list[int]: Selected shadow model indices.
    '''
    assert 0 <= index_target < n_models
    if index_target % 2 == 0:
        excluded_indices = {index_target, index_target + 1}
    else:
        excluded_indices = {index_target - 1, index_target}
    return sorted(set(range(n_models)) - excluded_indices)

def load_loss_signals(loss_path):
    '''
    Load one target checkpoint's loss signals and train mask.
    Args:
        loss_path (str | Path): Loss-signal pickle path.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Loss signals and membership mask.
    '''
    with open(loss_path, "rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Loss-signal pickle must contain keys 'loss_sigs' and 'train_mask'.")
    if "loss_sigs" not in payload or "train_mask" not in payload:
        raise ValueError("Loss-signal pickle must contain keys 'loss_sigs' and 'train_mask'.")
    loss_sigs = torch.tensor(payload["loss_sigs"], dtype=torch.float32)
    train_mask = torch.tensor(payload["train_mask"], dtype=torch.bool)
    if loss_sigs.ndim != 1 or train_mask.ndim != 1:
        raise ValueError(f"Loss-signal pickle must store 1D tensors: {loss_path}")
    if len(loss_sigs) != len(train_mask):
        raise ValueError(f"Loss-signal length and train mask length mismatch in {loss_path}.")
    return loss_sigs, train_mask
