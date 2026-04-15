from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import datasets as data_datasets
from data.utils import load_dataset


def get_entity_dataset_names():
    '''
    Return the sorted names of datasets that expose entity ids.
    Returns:
        list[str]: Sorted entity-dataset names.
    '''
    entity_dataset_names = []
    for name, dataset_class in vars(data_datasets).items():
        if not isinstance(dataset_class, type):
            continue
        if not issubclass(dataset_class, data_datasets.EntityDataset):
            continue
        if dataset_class is data_datasets.EntityDataset:
            continue
        entity_dataset_names.append(name)
    entity_dataset_names.sort()
    return entity_dataset_names


def get_entity_counts(dataset):
    '''
    Return the number of data points for each entity in a dataset.
    Args:
        dataset (data_datasets.EntityDataset): Dataset with entity ids.
    Returns:
        torch.Tensor: Count for each entity.
    '''
    entity_counts = torch.bincount(dataset.entity_ids)
    return entity_counts.to(dtype=torch.long)


def summarize_counts(entity_counts):
    '''
    Compute summary statistics for entity counts.
    Args:
        entity_counts (torch.Tensor): Number of data points per entity.
    Returns:
        dict: Summary statistics for the counts.
    '''
    entity_counts_float = entity_counts.to(dtype=torch.float32)
    count_mean = float(entity_counts_float.mean().item())
    count_std = float(entity_counts_float.std(unbiased=False).item())
    count_max = int(entity_counts.max().item())
    count_min = int(entity_counts.min().item())
    summary = {
        "mean": count_mean,
        "std": count_std,
        "max": count_max,
        "min": count_min,
    }
    return summary


def save_histogram(entity_counts, dataset_name, output_path):
    '''
    Save a histogram of entity counts to disk.
    Args:
        entity_counts (torch.Tensor): Number of data points per entity.
        dataset_name (str): Dataset name shown in the title.
        output_path (Path): Output PNG path.
    Returns:
        None
    '''
    count_min = int(entity_counts.min().item())
    count_max = int(entity_counts.max().item())
    bin_edges = np.arange(count_min - 0.5, count_max + 1.5, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(entity_counts.tolist(), bins=bin_edges)
    plt.xlabel("Number of data points per entity")
    plt.ylabel("Number of entities")
    plt.title(f"{dataset_name} entity count distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args(argv=None):
    '''
    Parse command line arguments for the entity-count script.
    Args:
        argv (list[str] | None): Optional CLI argument list.
    Returns:
        argparse.Namespace: Parsed arguments.
    '''
    parser = argparse.ArgumentParser(description="Compute per-entity count statistics for an EntityDataset.")
    parser.add_argument("--dataset", required=True, choices=get_entity_dataset_names())
    parser.add_argument("--data-dir", default="./datasets")
    return parser.parse_args(argv)


def main(argv=None):
    '''
    Load a dataset, print entity-count statistics, and save a histogram.
    Args:
        argv (list[str] | None): Optional CLI argument list.
    Returns:
        None
    '''
    args = parse_args(argv)
    dataset = load_dataset(dataset_name=args.dataset, data_dir=args.data_dir)
    if not isinstance(dataset, data_datasets.EntityDataset):
        raise TypeError(f"{args.dataset} is not an EntityDataset.")
    entity_counts = get_entity_counts(dataset)
    summary = summarize_counts(entity_counts)
    output_path = Path("images") / f"{args.dataset}_entity_count_histogram.png"
    save_histogram(entity_counts, args.dataset, output_path)
    print(f"Dataset: {args.dataset}")
    print(f"Entities: {len(entity_counts)}")
    print(f"Mean: {summary['mean']:.6f}")
    print(f"Std: {summary['std']:.6f}")
    print(f"Max: {summary['max']}")
    print(f"Min: {summary['min']}")
    print(f"Saved histogram to: {output_path}")


if __name__ == "__main__":
    main()
