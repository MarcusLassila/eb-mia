from data.dataset_metadata import entity_index_table_from_entity_ids, load_dataset_metadata
from check_train_splits import expected_count_options
from utils import get_train_indices, parse_properties_from_checkpoint_path

import argparse

import torch


def normalize_train_indices(train_indices):
    '''
    Convert checkpoint train indices to a Python list of ints.
    Args:
        train_indices (torch.Tensor | Sequence[int]): Loaded checkpoint indices.
    Returns:
        list[int]: Normalized training indices.
    '''
    index_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    return index_tensor.tolist()


def validate_index_properties(indices, n_samples, checkpoint_path):
    '''
    Validate basic index properties for one checkpoint.
    Args:
        indices (list[int]): Training indices loaded from the checkpoint.
        n_samples (int): Number of dataset samples in metadata.
        checkpoint_path (str): Checkpoint path used for error messages.
    Returns:
        None
    '''
    if indices != sorted(indices):
        raise ValueError(f"Train indices are not sorted in {checkpoint_path}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Train indices are not unique in {checkpoint_path}")
    if not indices:
        return
    smallest_index = min(indices)
    largest_index = max(indices)
    if smallest_index < 0 or largest_index >= n_samples:
        raise ValueError(f"Train indices are out of bounds in {checkpoint_path}")


def expected_selected_counts(total_count, fraction, complement):
    '''
    Return allowed selection counts for one split specification.
    Args:
        total_count (int): Number of available items.
        fraction (float): Fraction stored in the checkpoint path.
        complement (bool): Whether the checkpoint path represents a complement split.
    Returns:
        set[int]: Allowed selected counts.
    '''
    allowed_counts = expected_count_options(total_count, fraction)
    if not complement:
        return allowed_counts
    return {total_count - count for count in allowed_counts}


def validate_sample_split(properties, indices, metadata, checkpoint_path):
    '''
    Validate a sample-based checkpoint against dataset metadata.
    Args:
        properties (dict): Parsed checkpoint path properties.
        indices (list[int]): Training indices loaded from the checkpoint.
        metadata (dict): Dataset metadata.
        checkpoint_path (str): Checkpoint path used for error messages.
    Returns:
        dict: Summary statistics for reporting.
    '''
    n_samples = metadata["n_samples"]
    if "first_n_samples" in properties:
        expected_indices = list(range(properties["first_n_samples"]))
        if indices != expected_indices:
            raise ValueError(f"First-n sample split mismatch in {checkpoint_path}")
        return {
            "training_samples": len(indices),
            "hold_out_samples": 0,
            "selected_entities": None,
        }
    allowed_counts = expected_selected_counts(
        total_count=n_samples,
        fraction=properties["fraction"],
        complement=properties["complement"],
    )
    if len(indices) not in allowed_counts:
        raise ValueError(f"Unexpected sample count in {checkpoint_path}")
    return {
        "training_samples": len(indices),
        "hold_out_samples": 0,
        "selected_entities": None,
    }


def validate_entity_split(properties, indices, metadata, checkpoint_path):
    '''
    Validate an entity-based checkpoint against dataset metadata.
    Args:
        properties (dict): Parsed checkpoint path properties.
        indices (list[int]): Training indices loaded from the checkpoint.
        metadata (dict): Dataset metadata.
        checkpoint_path (str): Checkpoint path used for error messages.
    Returns:
        dict: Summary statistics for reporting.
    '''
    if "entity_ids" not in metadata or "n_entities" not in metadata:
        raise ValueError(f"Entity metadata is missing for {checkpoint_path}")
    entity_ids = metadata["entity_ids"]
    entity_index_table = entity_index_table_from_entity_ids(entity_ids)
    selected_entity_ids = sorted({entity_ids[index] for index in indices})
    allowed_entity_counts = expected_selected_counts(
        total_count=metadata["n_entities"],
        fraction=properties["entity_fraction"],
        complement=properties["complement"],
    )
    if len(selected_entity_ids) not in allowed_entity_counts:
        raise ValueError(f"Unexpected entity count in {checkpoint_path}")
    index_set = set(indices)
    hold_out_count = 0
    for entity_id in selected_entity_ids:
        entity_indices = entity_index_table[entity_id]
        effective_count = len(entity_indices)
        hold_out_fraction = properties["per_entity_hold_out"]
        if hold_out_fraction > 0.0:
            effective_count = int(effective_count * (1.0 - hold_out_fraction))
        eligible_indices = set(entity_indices[:effective_count])
        hold_out_indices = set(entity_indices[effective_count:])
        selected_indices = index_set & set(entity_indices)
        hold_out_count += len(hold_out_indices)
        if selected_indices - eligible_indices:
            raise ValueError(f"Hold-out mismatch in {checkpoint_path}")
        allowed_counts = expected_count_options(
            total_count=effective_count,
            fraction=properties["per_entity_fraction"],
        )
        if len(selected_indices) not in allowed_counts:
            raise ValueError(f"Unexpected per-entity count in {checkpoint_path}")
    return {
        "training_samples": len(indices),
        "hold_out_samples": hold_out_count,
        "selected_entities": len(selected_entity_ids),
    }


def validate_checkpoint_against_metadata(checkpoint_path, metadata_dir=None):
    '''
    Validate checkpoint train indices against dataset metadata.
    Args:
        checkpoint_path (str): Checkpoint path to inspect.
        metadata_dir (str | None): Optional dataset metadata directory.
    Returns:
        dict: Validation summary.
    '''
    properties = parse_properties_from_checkpoint_path(checkpoint_path)
    metadata = load_dataset_metadata(properties["dataset"], metadata_dir=metadata_dir)
    train_indices = get_train_indices(checkpoint_path)
    indices = normalize_train_indices(train_indices)
    validate_index_properties(
        indices=indices,
        n_samples=metadata["n_samples"],
        checkpoint_path=checkpoint_path,
    )
    if properties["split_mode"] == "sample":
        summary = validate_sample_split(
            properties=properties,
            indices=indices,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
        )
    else:
        summary = validate_entity_split(
            properties=properties,
            indices=indices,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
        )
    summary["checkpoint_path"] = checkpoint_path
    summary["dataset"] = properties["dataset"]
    summary["split_mode"] = properties["split_mode"]
    summary["complement"] = properties["complement"]
    return summary


def parse_args(argv=None):
    '''
    Parse CLI arguments for checkpoint metadata verification.
    Args:
        argv (list[str] | None): Optional CLI argument list.
    Returns:
        argparse.Namespace: Parsed arguments.
    '''
    parser = argparse.ArgumentParser(description="Validate checkpoint train indices against dataset metadata.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--metadata-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    '''
    Validate one checkpoint and print a short summary.
    Args:
        argv (list[str] | None): Optional CLI argument list.
    Returns:
        dict: Validation summary.
    '''
    args = parse_args(argv)
    summary = validate_checkpoint_against_metadata(
        checkpoint_path=args.checkpoint_path,
        metadata_dir=args.metadata_dir,
    )
    print(f"Checkpoint: {summary['checkpoint_path']}")
    print(f"Dataset: {summary['dataset']}")
    print(f"Split mode: {summary['split_mode']}")
    print(f"Complement split: {summary['complement']}")
    print(f"Training samples: {summary['training_samples']}")
    if summary["selected_entities"] is not None:
        print(f"Selected entities: {summary['selected_entities']}")
    print(f"Hold-out samples excluded: {summary['hold_out_samples']}")
    print("Status: OK")
    return summary


if __name__ == "__main__":
    main()
