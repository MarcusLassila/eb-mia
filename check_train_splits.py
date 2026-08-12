from itertools import combinations
from pathlib import Path
import argparse
import math
import re

import numpy as np
import torch

from data import datasets as data_datasets
from data.utils import load_dataset
from training import train_split


def _parse_fraction(token):
    '''
    Convert a path fraction token into a float.
    Args:
        token (str): Fraction token such as `0p5`.
    Returns:
        float: Parsed fraction.
    '''
    return float(token.replace("p", "."))


def parse_train_split_stem(stem):
    '''
    Parse a train-split stem into structured metadata.
    Args:
        stem (str): Split filename without suffix.
    Returns:
        dict: Parsed split metadata.
    '''
    sample_match = re.match(
        r"^(?P<dataset>.+)-smpl-f(?P<fraction>\d+p\d+)-s(?P<seed>\d+)(?P<complement>-comp)?$",
        stem,
    )
    if sample_match is not None:
        return {
            "dataset": sample_match.group("dataset"),
            "split_mode": "sample",
            "fraction": _parse_fraction(sample_match.group("fraction")),
            "seed": int(sample_match.group("seed")),
            "complement": sample_match.group("complement") is not None,
        }
    entity_match = re.match(
        r"^(?P<dataset>.+)-ent-f(?P<entity_fraction>\d+p\d+)-p(?P<per_entity_fraction>\d+p\d+)(?:-h(?P<per_entity_hold_out>\d+p\d+))?-s(?P<seed>\d+)(?P<complement>-comp)?$",
        stem,
    )
    if entity_match is not None:
        hold_out_fraction = 0.0
        if entity_match.group("per_entity_hold_out") is not None:
            hold_out_fraction = _parse_fraction(entity_match.group("per_entity_hold_out"))
        return {
            "dataset": entity_match.group("dataset"),
            "split_mode": "entity",
            "entity_fraction": _parse_fraction(entity_match.group("entity_fraction")),
            "per_entity_fraction": _parse_fraction(entity_match.group("per_entity_fraction")),
            "per_entity_hold_out": hold_out_fraction,
            "seed": int(entity_match.group("seed")),
            "complement": entity_match.group("complement") is not None,
        }
    raise ValueError(f"Could not parse train split filename: {stem}")


def build_split_spec(parsed):
    '''
    Return the shared split specification excluding seed and complement.
    Args:
        parsed (dict): Parsed split metadata.
    Returns:
        tuple: Canonical split specification.
    '''
    if parsed["split_mode"] == "sample":
        return (
            parsed["dataset"],
            parsed["split_mode"],
            parsed["fraction"],
        )
    return (
        parsed["dataset"],
        parsed["split_mode"],
        parsed["entity_fraction"],
        parsed["per_entity_fraction"],
        parsed["per_entity_hold_out"],
    )


def expected_count_options(total_count, fraction):
    '''
    Return the allowed selection counts for stochastic rounding.
    Args:
        total_count (int): Number of available items.
        fraction (float): Selection fraction.
    Returns:
        set[int]: Allowed selected counts.
    '''
    expected_count = total_count * fraction
    lower_count = int(math.floor(expected_count))
    upper_count = int(math.ceil(expected_count))
    return {lower_count, upper_count}


def load_split_records(train_splits_dir):
    '''
    Load all split pickle files from a directory.
    Args:
        train_splits_dir (str | Path): Directory containing split files.
    Returns:
        list[dict]: Loaded split records.
    '''
    train_splits_dir = Path(train_splits_dir)
    split_paths = sorted(train_splits_dir.glob("*.pkl"))
    if not split_paths:
        raise ValueError(f"No split files found in {train_splits_dir}")
    split_records = []
    for split_path in split_paths:
        parsed = parse_train_split_stem(split_path.stem)
        indices = train_split.load_indices(split_path)
        split_record = {
            "path": split_path,
            "stem": split_path.stem,
            "parsed": parsed,
            "indices": indices,
        }
        split_records.append(split_record)
    return split_records


def validate_shared_spec(split_records):
    '''
    Validate that all split files share one split specification.
    Args:
        split_records (list[dict]): Loaded split records.
    Returns:
        dict: Parsed metadata shared by the folder.
    '''
    parsed_records = [record["parsed"] for record in split_records]
    datasets = {parsed["dataset"] for parsed in parsed_records}
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset, got: {sorted(datasets)}")
    split_specs = {build_split_spec(parsed) for parsed in parsed_records}
    if len(split_specs) != 1:
        raise ValueError("Expected splits to differ only by seed and -comp flag.")
    return parsed_records[0]


def get_split_entities(dataset, indices):
    '''
    Return the entity ids present in one split.
    Args:
        dataset (data_datasets.EntityDataset): Dataset used by the split.
        indices (list[int]): Training indices.
    Returns:
        set[int]: Entity ids present in the split.
    '''
    index_tensor = torch.tensor(indices, dtype=torch.long)
    entity_ids = torch.unique(dataset.entity_ids[index_tensor], sorted=True)
    return set(entity_ids.tolist())


def validate_index_properties(split_record, dataset):
    '''
    Validate basic index properties for one split file.
    Args:
        split_record (dict): Loaded split record.
        dataset (data_datasets.EntityDataset): Dataset used by the split.
    Returns:
        None
    '''
    indices = split_record["indices"]
    if indices != sorted(indices):
        raise ValueError(f"Indices are not sorted in {split_record['path']}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Indices are not unique in {split_record['path']}")
    len_dataset = len(dataset)
    if indices and (min(indices) < 0 or max(indices) >= len_dataset):
        raise ValueError(f"Indices out of bounds in {split_record['path']}")


def validate_split_matches_spec(split_record, dataset):
    '''
    Validate one split against the counts implied by its path name.
    Args:
        split_record (dict): Loaded split record.
        dataset (data_datasets.EntityDataset): Dataset used by the split.
    Returns:
        None
    '''
    parsed = split_record["parsed"]
    indices = split_record["indices"]
    validate_index_properties(split_record, dataset)
    if parsed["split_mode"] == "sample":
        if not parsed["complement"]:
            allowed_counts = expected_count_options(len(dataset), parsed["fraction"])
            if len(indices) not in allowed_counts:
                raise ValueError(f"Unexpected sample count in {split_record['path']}")
        return
    split_entities = get_split_entities(dataset, indices)
    allowed_entity_counts = expected_count_options(dataset.n_entities, parsed["entity_fraction"])
    if len(split_entities) not in allowed_entity_counts:
        raise ValueError(f"Unexpected entity count in {split_record['path']}")
    entity_index_table = dataset.get_entity_index_table()
    index_set = set(indices)
    for entity_id in split_entities:
        entity_indices = entity_index_table[entity_id]
        effective_count = len(entity_indices)
        hold_out_fraction = parsed["per_entity_hold_out"]
        if hold_out_fraction > 0.0:
            effective_count = int(effective_count * (1.0 - hold_out_fraction))
        allowed_indices = set(entity_indices[:effective_count])
        selected_indices = index_set & set(entity_indices)
        if not selected_indices <= allowed_indices:
            raise ValueError(f"Hold-out mismatch in {split_record['path']}")
        allowed_counts = expected_count_options(effective_count, parsed["per_entity_fraction"])
        if len(selected_indices) not in allowed_counts:
            raise ValueError(f"Unexpected per-entity count in {split_record['path']}")


def pair_seed_key(split_record):
    '''
    Return the pairing key for a split file.
    Args:
        split_record (dict): Loaded split record.
    Returns:
        int | None: Seed used to pair base and complement files.
    '''
    return split_record["parsed"].get("seed")


def build_complement_pairs(split_records):
    '''
    Pair each base split with its complement split.
    Args:
        split_records (list[dict]): Loaded split records.
    Returns:
        list[tuple[dict, dict]]: Base/complement pairs.
    '''
    grouped_records = {}
    for split_record in split_records:
        seed_key = pair_seed_key(split_record)
        grouped_records.setdefault(seed_key, []).append(split_record)
    complement_pairs = []
    for seed_key, group in sorted(grouped_records.items()):
        if len(group) != 2:
            raise ValueError(f"Expected one base/complement pair for seed {seed_key}.")
        base_records = [record for record in group if not record["parsed"]["complement"]]
        comp_records = [record for record in group if record["parsed"]["complement"]]
        if len(base_records) != 1 or len(comp_records) != 1:
            raise ValueError(f"Invalid complement grouping for seed {seed_key}.")
        complement_pairs.append((base_records[0], comp_records[0]))
    return complement_pairs


def validate_complement_pairs(complement_pairs, dataset):
    '''
    Validate that complement pairs are disjoint in datapoints and entities.
    Args:
        complement_pairs (list[tuple[dict, dict]]): Base/complement pairs.
        dataset (data_datasets.EntityDataset): Dataset used by the splits.
    Returns:
        None
    '''
    all_entity_ids = set(range(dataset.n_entities))
    for base_record, comp_record in complement_pairs:
        base_indices = set(base_record["indices"])
        comp_indices = set(comp_record["indices"])
        if base_indices & comp_indices:
            raise ValueError(f"Overlapping datapoints in {base_record['path']} and {comp_record['path']}")
        base_entities = get_split_entities(dataset, base_record["indices"])
        comp_entities = get_split_entities(dataset, comp_record["indices"])
        if base_entities & comp_entities:
            raise ValueError(f"Overlapping entities in {base_record['path']} and {comp_record['path']}")
        if base_record["parsed"]["split_mode"] == "sample":
            union_indices = base_indices | comp_indices
            if len(union_indices) != len(dataset):
                raise ValueError(f"Sample complement pair does not cover the full dataset for seed {pair_seed_key(base_record)}")
        if base_record["parsed"]["split_mode"] == "entity":
            union_entities = base_entities | comp_entities
            if union_entities != all_entity_ids:
                raise ValueError(f"Entity complement pair does not cover all entities for seed {pair_seed_key(base_record)}")


def compute_other_pair_overlaps(split_records, complement_pairs, dataset):
    '''
    Compute overlap percentages for all non-complement pairs.
    Args:
        split_records (list[dict]): Loaded split records.
        complement_pairs (list[tuple[dict, dict]]): Base/complement pairs.
        dataset (data_datasets.EntityDataset): Dataset used by the splits.
    Returns:
        tuple[list[float], list[float]]: Datapoint and entity overlap percentages.
    '''
    excluded_pairs = set()
    for base_record, comp_record in complement_pairs:
        excluded_pairs.add(frozenset({base_record["stem"], comp_record["stem"]}))
    datapoint_overlaps = []
    entity_overlaps = []
    for first_record, second_record in combinations(split_records, 2):
        pair_key = frozenset({first_record["stem"], second_record["stem"]})
        if pair_key in excluded_pairs:
            continue
        first_indices = set(first_record["indices"])
        second_indices = set(second_record["indices"])
        datapoint_overlap = len(first_indices & second_indices) / len(dataset)
        datapoint_overlaps.append(100.0 * datapoint_overlap)
        first_entities = get_split_entities(dataset, first_record["indices"])
        second_entities = get_split_entities(dataset, second_record["indices"])
        entity_overlap = len(first_entities & second_entities) / dataset.n_entities
        entity_overlaps.append(100.0 * entity_overlap)
    return datapoint_overlaps, entity_overlaps


def summarize_percentages(values):
    '''
    Return the mean and standard deviation of percentage values.
    Args:
        values (list[float]): Percentage values.
    Returns:
        tuple[float, float]: Mean and standard deviation.
    '''
    if not values:
        return 0.0, 0.0
    values_array = np.array(values, dtype=float)
    value_mean = float(values_array.mean())
    value_std = float(values_array.std())
    return value_mean, value_std


def parse_args(argv=None):
    '''
    Parse command line arguments for the split checker.
    Args:
        argv (list[str] | None): Optional CLI argument list.
    Returns:
        argparse.Namespace: Parsed arguments.
    '''
    parser = argparse.ArgumentParser(description="Validate train split consistency and pair overlaps.")
    parser.add_argument("--train-splits-dir", required=True)
    parser.add_argument("--data-dir", default="./datasets")
    return parser.parse_args(argv)


def main(argv=None):
    '''
    Validate a folder of train splits and report overlap statistics.
    Args:
        argv (list[str] | None): Optional CLI argument list.
    Returns:
        None
    '''
    args = parse_args(argv)
    split_records = load_split_records(args.train_splits_dir)
    shared_spec = validate_shared_spec(split_records)
    dataset = load_dataset(shared_spec["dataset"], data_dir=args.data_dir)
    if not isinstance(dataset, data_datasets.EntityDataset):
        raise TypeError(f"{shared_spec['dataset']} is not an EntityDataset.")
    for split_record in split_records:
        len_dataset = train_split.load_indices(split_record["path"], len_dataset=len(dataset))
        split_record["indices"] = len_dataset
        validate_split_matches_spec(split_record, dataset)
    complement_pairs = build_complement_pairs(split_records)
    validate_complement_pairs(complement_pairs, dataset)
    datapoint_overlaps, entity_overlaps = compute_other_pair_overlaps(split_records, complement_pairs, dataset)
    datapoint_mean, datapoint_std = summarize_percentages(datapoint_overlaps)
    entity_mean, entity_std = summarize_percentages(entity_overlaps)
    print(f"Dataset: {shared_spec['dataset']}")
    print(f"Split mode: {shared_spec['split_mode']}")
    print(f"Number of split files: {len(split_records)}")
    print(f"Complement pairs checked: {len(complement_pairs)}")
    print(f"Other pairs checked: {len(datapoint_overlaps)}")
    print(f"Mean datapoint overlap (% of dataset): {datapoint_mean:.6f}")
    print(f"Std datapoint overlap (% of dataset): {datapoint_std:.6f}")
    print(f"Mean entity overlap (% of entities): {entity_mean:.6f}")
    print(f"Std entity overlap (% of entities): {entity_std:.6f}")


if __name__ == "__main__":
    main()
