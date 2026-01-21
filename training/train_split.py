import argparse
import pickle
import random
from pathlib import Path

def _format_fraction(fraction):
    '''Format fraction for filenames. Args: fraction (float). Returns: str.'''
    formatted = f"{fraction:.6f}".rstrip("0").rstrip(".")
    return formatted.replace(".", "p")

def _normalize_indices(indices):
    '''Normalize indices into a list of ints. Args: indices (iterable). Returns: list[int].'''
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    indices_list = [int(index) for index in indices]
    return indices_list

def _normalize_labels(labels):
    '''Normalize labels into a list of ints. Args: labels (iterable). Returns: list[int].'''
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    return [int(label) for label in labels]

def _validate_indices(indices, n_items):
    '''Validate index uniqueness and bounds. Args: indices (iterable), n_items (int). Returns: None.'''
    indices_set = set(indices)
    if len(indices_set) != len(indices):
        raise ValueError("indices must be unique")
    if indices_set and (min(indices_set) < 0 or max(indices_set) >= n_items):
        raise ValueError("indices must be within [0, n_items)")

def random_subset_filename(dataset_name, fraction, seed):
    '''Build random subset filename. Args: dataset_name (str), fraction (float), seed (int). Returns: str.'''
    fraction_str = _format_fraction(fraction)
    return f"{dataset_name}-random-frac{fraction_str}-seed{seed}.pkl"

def complement_subset_filename(subset_path):
    '''Build complement subset filename. Args: subset_path (str|Path). Returns: str.'''
    subset_stem = Path(subset_path).stem
    return f"{subset_stem}-complement.pkl"

def save_indices(indices, path):
    '''Save indices to pickle. Args: indices (iterable), path (str|Path). Returns: Path.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = _normalize_indices(indices)
    with open(path, "wb") as file:
        pickle.dump(indices, file)
    return path

def load_indices(path):
    '''Load indices from pickle. Args: path (str|Path). Returns: list[int].'''
    with open(path, "rb") as file:
        indices = pickle.load(file)
    return _normalize_indices(indices)

def entity_subset_filename(dataset_name, entity_fraction, per_entity_fraction, seed):
    '''Build entity subset filename. Args: dataset_name (str), entity_fraction (float), per_entity_fraction (float), seed (int). Returns: str.'''
    entity_frac_str = _format_fraction(entity_fraction)
    per_entity_frac_str = _format_fraction(per_entity_fraction)
    return (
        f"{dataset_name}-entity-frac{entity_frac_str}"
        f"-per{per_entity_frac_str}-seed{seed}.pkl"
    )

def sample_random_fraction_indices(n_items, fraction, seed):
    '''Sample indices uniformly. Args: n_items (int), fraction (float), seed (int). Returns: list[int].'''
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    subset_size = int(n_items * fraction)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n_items), k=subset_size))
    return indices

def sample_entity_fraction_indices(entity_ids, entity_fraction, per_entity_fraction, seed):
    '''Sample entity-based indices. Args: entity_ids (iterable), entity_fraction (float), per_entity_fraction (float), seed (int). Returns: list[int].'''
    entity_ids = _normalize_labels(entity_ids)
    if not 0.0 < entity_fraction <= 1.0:
        raise ValueError("entity_fraction must be in (0, 1]")
    if not 0.0 < per_entity_fraction <= 1.0:
        raise ValueError("per_entity_fraction must be in (0, 1]")
    indices_by_entity = {}
    for index, entity_id in enumerate(entity_ids):
        indices_by_entity.setdefault(entity_id, []).append(index)
    unique_entity_ids = sorted(indices_by_entity)
    subset_size = int(len(unique_entity_ids) * entity_fraction)
    rng = random.Random(seed)
    sampled_entity_ids = sorted(rng.sample(unique_entity_ids, k=subset_size))
    sampled_indices = []
    for entity_id in sampled_entity_ids:
        entity_indices = indices_by_entity[entity_id]
        entity_subset_size = int(len(entity_indices) * per_entity_fraction)
        entity_subset_size = max(1, entity_subset_size)
        entity_subset_size = min(len(entity_indices), entity_subset_size)
        sampled_indices.extend(rng.sample(entity_indices, k=entity_subset_size))
    return sorted(sampled_indices)

def create_random_subset(dataset_name, n_items, fraction, seed, output_dir):
    '''Create and save random subset. Args: dataset_name (str), n_items (int), fraction (float), seed (int), output_dir (str|Path). Returns: Path.'''
    indices = sample_random_fraction_indices(n_items, fraction, seed)
    filename = random_subset_filename(dataset_name, fraction, seed)
    path = Path(output_dir) / filename
    save_indices(indices, path)
    return path

def create_entity_subset(dataset_name, entity_ids, entity_fraction, per_entity_fraction, seed, output_dir):
    '''Create and save entity subset. Args: dataset_name (str), entity_ids (iterable), entity_fraction (float), per_entity_fraction (float), seed (int), output_dir (str|Path). Returns: Path.'''
    indices = sample_entity_fraction_indices(
        entity_ids=entity_ids,
        entity_fraction=entity_fraction,
        per_entity_fraction=per_entity_fraction,
        seed=seed,
    )
    filename = entity_subset_filename(
        dataset_name,
        entity_fraction,
        per_entity_fraction,
        seed,
    )
    path = Path(output_dir) / filename
    save_indices(indices, path)
    return path

def complement_indices(indices, n_items):
    '''Return complement indices. Args: indices (iterable), n_items (int). Returns: list[int].'''
    indices = _normalize_indices(indices)
    _validate_indices(indices, n_items)
    indices_set = set(indices)
    return [index for index in range(n_items) if index not in indices_set]

def create_complement_subset(subset_path, n_items, output_dir=None):
    '''Create and save complement subset. Args: subset_path (str|Path), n_items (int), output_dir (str|Path|None). Returns: Path.'''
    indices = load_indices(subset_path)
    complement = complement_indices(indices, n_items)
    if output_dir is None:
        output_dir = Path(subset_path).parent
    filename = complement_subset_filename(subset_path)
    path = Path(output_dir) / filename
    save_indices(complement, path)
    return path

def _build_parser():
    '''Build CLI argument parser. Args: None. Returns: argparse.ArgumentParser.'''
    parser = argparse.ArgumentParser(description="Create training split index files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--output-dir", default="training/train_splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["random", "entity", "complement"], default="random")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--entity-fraction", type=float, default=1.0)
    parser.add_argument("--per-entity-fraction", type=float, default=1.0)
    parser.add_argument("--subset-path")
    return parser

def main(argv=None):
    '''Run CLI for split creation. Args: argv (list|None). Returns: Path.'''
    parser = _build_parser()
    args = parser.parse_args(argv)
    from data import data as data_module
    dataset = getattr(data_module, args.dataset)(data_dir=args.data_dir, transform=None)
    output_dir = Path(args.output_dir)
    if args.mode == "random":
        return create_random_subset(args.dataset, len(dataset), args.fraction, args.seed, output_dir)
    if args.mode == "entity":
        entity_ids = getattr(dataset, "entity_ids", None)
        if entity_ids is None:
            raise ValueError("dataset must provide entity_ids for entity mode")
        return create_entity_subset(
            dataset_name=args.dataset,
            entity_ids=entity_ids,
            entity_fraction=args.entity_fraction,
            per_entity_fraction=args.per_entity_fraction,
            seed=args.seed,
            output_dir=output_dir,
        )
    if args.subset_path is None:
        raise ValueError("subset_path is required for complement mode")
    return create_complement_subset(args.subset_path, len(dataset), output_dir=output_dir)

if __name__ == "__main__":
    main()
