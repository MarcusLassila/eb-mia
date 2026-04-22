from data import datasets as data_datasets
from data.datasets import EntityDataset
from data.utils import load_dataset
import utils

import argparse
from pathlib import Path
import pickle


def build_dataset_metadata(dataset_name, data_dir):
    '''
    Build serializable population metadata for one dataset.
    Args:
        dataset_name (str): Dataset name passed to `load_dataset`.
        data_dir (str | Path): Dataset cache directory.
    Returns:
        dict: Serializable dataset metadata.
    '''
    dataset = load_dataset(dataset_name=dataset_name, data_dir=data_dir)
    metadata = {
        "version": 1,
        "dataset": dataset_name,
        "n_samples": len(dataset),
    }
    if isinstance(dataset, EntityDataset):
        metadata["entity_ids"] = dataset.entity_ids.tolist()
        metadata["n_entities"] = int(dataset.n_entities)
    return metadata


def save_dataset_metadata(metadata, output_path):
    '''
    Save dataset metadata to a pickle file.
    Args:
        metadata (dict): Serializable dataset metadata.
        output_path (str | Path): Output pickle path.
    Returns:
        Path: Saved pickle path.
    '''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(metadata, file)
    return output_path


def load_dataset_metadata(dataset_name, metadata_dir=None):
    '''
    Load dataset metadata from a pickle file.
    Args:
        dataset_name (str): Dataset name used in the metadata file name.
        metadata_dir (str | Path | None): Directory containing metadata pickle files.
    Returns:
        dict: Loaded dataset metadata.
    '''
    root = utils.get_root()
    if metadata_dir is None:
        metadata_dir = utils.resolve_path(Path("data") / "metadata", root)
    metadata_path = Path(metadata_dir) / f"{dataset_name}.pkl"
    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)
    if metadata.get("dataset") != dataset_name:
        raise ValueError(f"Dataset metadata name mismatch in {metadata_path}")
    if "entity_ids" in metadata and len(metadata["entity_ids"]) != metadata["n_samples"]:
        raise ValueError(f"Entity id length mismatch in {metadata_path}")
    return metadata


def entity_index_table_from_entity_ids(entity_ids):
    '''
    Build an entity index table from entity ids in population order.
    Args:
        entity_ids (list[int]): Entity id for each population sample.
    Returns:
        dict[int, list[int]]: Population indices grouped by entity id.
    '''
    entity_index_table = {}
    for index, entity_id in enumerate(entity_ids):
        if entity_id not in entity_index_table:
            entity_index_table[entity_id] = []
        entity_index_table[entity_id].append(index)
    return entity_index_table


def dataset_names():
    '''
    Return dataset names defined in `data.datasets`.
    Returns:
        tuple[str, ...]: Supported dataset names.
    '''
    names = []
    for name, value in vars(data_datasets).items():
        is_dataset_class = isinstance(value, type)
        if not is_dataset_class:
            continue
        if value.__module__ != "data.datasets":
            continue
        if name in {"Dataset", "EntityDataset"}:
            continue
        names.append(name)
    return tuple(names)


def parse_args(argv=None):
    '''
    Parse CLI arguments for dataset metadata export.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    root = utils.get_root()
    default_output_dir = utils.resolve_path(Path("data") / "metadata", root)
    parser = argparse.ArgumentParser(description="Compute and save dataset population metadata.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name passed to data.utils.load_dataset. When omitted, export metadata for every dataset in data.datasets.",
    )
    parser.add_argument("--data-dir", default="./datasets", help="Dataset cache directory.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Directory used for saved metadata pickle files.")
    return parser.parse_args(argv)


def main(argv=None):
    '''
    Entry point for dataset metadata export.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        Path | list[Path]: Saved pickle path or paths.
    '''
    args = parse_args(argv)
    root = utils.get_root()
    data_dir = utils.resolve_path(args.data_dir, root)
    output_dir = utils.resolve_path(args.output_dir, root)
    selected_dataset_names = [args.dataset] if args.dataset is not None else list(dataset_names())
    saved_paths = []
    for dataset_name in selected_dataset_names:
        metadata = build_dataset_metadata(dataset_name=dataset_name, data_dir=data_dir)
        output_path = output_dir / f"{dataset_name}.pkl"
        saved_path = save_dataset_metadata(metadata, output_path)
        print(f"Saved dataset metadata to {saved_path}")
        saved_paths.append(saved_path)
    if len(saved_paths) == 1:
        return saved_paths[0]
    return saved_paths


if __name__ == "__main__":
    main()
