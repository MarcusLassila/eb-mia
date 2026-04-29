from utils import mask_to_index

import torch
import pickle

def indices_of_shadow_models(index_target, train_mask):
    '''
    Return shadow indices excluding the target and its complement.
    Args:
        index_target (int): Index of the target model.
        train_mask (torch.Tensor): Training mask over model pool.
    Returns:
        list[int]: Selected shadow model indices.
    '''
    target_train_mask = train_mask[index_target]
    index_complement = None
    indices = []
    for i, mask in enumerate(train_mask):
        if torch.all(mask & target_train_mask == 0):
            assert index_complement is None, "Should be only one complement model"
            index_complement = i
        elif i != index_target:
            indices.append(i)
    assert index_complement is not None
    return indices

def select_sample_audit_indices(n_audit_samples, train_mask):
    assert n_audit_samples <= train_mask.shape[0]
    member_indices = mask_to_index(train_mask)
    non_member_indices = mask_to_index(~train_mask)
    if 2 * member_indices.shape[0] < n_audit_samples:
        n_in_samples = member_indices.shape[0]
        n_out_samples = n_audit_samples - n_in_samples
    elif 2 * non_member_indices.shape[0] < n_audit_samples:
        n_out_samples = non_member_indices.shape[0]
        n_in_samples = n_audit_samples - n_out_samples
    else:
        n_in_samples = n_audit_samples // 2
        n_out_samples = n_audit_samples // 2 + n_audit_samples % 2
    rand_mask = torch.randperm(member_indices.shape[0])
    selected_members = member_indices[rand_mask][:n_in_samples]
    rand_mask = torch.randperm(non_member_indices.shape[0])
    selected_non_members = non_member_indices[rand_mask][:n_out_samples]
    audit_indices = torch.cat((selected_members, selected_non_members)).sort()[0]
    return audit_indices

def select_entitywise_audit_indices(
    entity_indices,
    train_mask,
    mode,
    min_samples_per_entity=0,
    max_samples_per_entity=None,
    hold_out_frac=0.0,
):
    entity_indices = torch.as_tensor(entity_indices)
    train_indices = entity_indices[train_mask[entity_indices]]
    train_indices = train_indices[torch.randperm(train_indices.shape[0])]
    non_train_indices = entity_indices[~train_mask[entity_indices]]
    non_train_indices = non_train_indices[torch.randperm(non_train_indices.shape[0])]
    is_member_entity = train_indices.shape[0] > 0
    if max_samples_per_entity is None:
        max_samples_per_entity = entity_indices.shape[0]
    if mode == "all":
        if is_member_entity:
            n = min(train_indices.shape[0], non_train_indices.shape[0])
            selected_indices = torch.empty_like(entity_indices, dtype=torch.long)
            selected_indices[: 2 * n: 2] = train_indices[:n]
            selected_indices[1: 2 * n: 2] = non_train_indices[:n]
            if n < train_indices.shape[0]:
                selected_indices[2 * n:] = train_indices[n:]
            elif n < non_train_indices.shape[0]:
                selected_indices[2 * n:] = non_train_indices[n:]
            selected_indices = selected_indices[:max_samples_per_entity].tolist()
        else:
            selected_indices = non_train_indices[:max_samples_per_entity].tolist()
    elif mode == "max_one_train":
        selected_indices = non_train_indices.tolist()
        if is_member_entity:
            selected_indices.append(train_indices[0].item())
        selected_indices = selected_indices[-max_samples_per_entity:]
    elif mode == "exclude_train":
        selected_indices = non_train_indices[:max_samples_per_entity].tolist()
    elif mode == "hold_out":
        n = int(len(entity_indices) * (1.0 - hold_out_frac))
        selected_indices = entity_indices[n:].tolist()[:max_samples_per_entity]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    n_selected = len(selected_indices)
    if n_selected < min_samples_per_entity:
        selected_indices = [] # Not enough indices to reach minimum number of samples per entity so select nothing
    selected_indices.sort()
    return selected_indices, is_member_entity

def select_entity_audit_indices(
    entity_index_table: dict,
    train_mask: torch.Tensor,
    mode: str,
    min_samples_per_entity: int = None,
    max_samples_per_entity: int = None,
    hold_out_frac: float = 0.0,
    balance_in_and_out: bool = True,
):
    train_entity_ids = set()
    selected_index_table = {}
    for entity_id, indices in entity_index_table.items():
        selected_indices, is_member_entity = select_entitywise_audit_indices(
            entity_indices=indices,
            train_mask=train_mask,
            mode=mode,
            min_samples_per_entity=min_samples_per_entity,
            max_samples_per_entity=max_samples_per_entity,
            hold_out_frac=hold_out_frac,
        )
        if selected_indices:
            selected_index_table[entity_id] = selected_indices
        if is_member_entity:
            train_entity_ids.add(entity_id)

    if balance_in_and_out:
        in_entity_ids = [entity_id for entity_id in selected_index_table.keys() if entity_id in train_entity_ids]
        out_entity_ids = [entity_id for entity_id in selected_index_table.keys() if entity_id not in train_entity_ids]
        if not in_entity_ids or not out_entity_ids:
            raise RuntimeError("Failed to include both target and non-target entities in the audit table.")

        keep_per_class = min(len(in_entity_ids), len(out_entity_ids))
        selected_entity_ids = in_entity_ids[:keep_per_class] + out_entity_ids[:keep_per_class]
        audit_table = {entity_id: selected_index_table[entity_id] for entity_id in selected_entity_ids}
    else:
        audit_table = selected_index_table
    return audit_table

def load_loss_signals(loss_path):
    '''
    Load one target checkpoint's audit loss signals and train mask.
    Args:
        loss_path (str | Path): Loss-signal pickle path.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Averaged 1D loss signals and membership mask.
    '''
    with open(loss_path, "rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Loss-signal pickle must contain keys 'loss_sigs' and 'train_mask'.")
    if "loss_sigs" not in payload or "train_mask" not in payload:
        raise ValueError("Loss-signal pickle must contain keys 'loss_sigs' and 'train_mask'.")
    loss_sigs = torch.tensor(payload["loss_sigs"], dtype=torch.float32)
    train_mask = torch.tensor(payload["train_mask"], dtype=torch.bool)
    if train_mask.ndim != 1:
        raise ValueError(f"Loss-signal pickle must store 1D train_mask tensor: {loss_path}")
    if loss_sigs.ndim == 2:
        loss_sigs = loss_sigs.mean(dim=1) # TODO: utilize full distribution rather than only empirical mean.
    elif loss_sigs.ndim != 1:
        raise ValueError(f"Unsupported format for loss signal tensor in pickle file: {loss_path}")
    if len(loss_sigs) != len(train_mask):
        raise ValueError(f"Loss-signal length and train mask length mismatch in {loss_path}.")
    return loss_sigs, train_mask
