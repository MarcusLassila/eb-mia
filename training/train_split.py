from data import datasets
from data.utils import load_dataset
from utils import index_to_mask, mask_to_index

import argparse
import re
import pickle
import numpy as np
import torch
from math import modf
from pathlib import Path

def load_indices(path, len_dataset=None):
    with open(path, "rb") as file:
        train_split = pickle.load(file)
    assert len_dataset is None or train_split["len_dataset"] == len_dataset
    return train_split["indices"]

def save_indices(indices, len_dataset, output_dir, file_name):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / file_name, "wb") as file:
        pickle.dump({"indices": indices, "len_dataset": len_dataset}, file)

def select_fraction(indices: list, fraction: float, rng: np.random.Generator):
    n = len(indices)
    f, k = modf(n * fraction)
    k = int(k)
    shuffled_indices = rng.permutation(indices)
    selected_indices = shuffled_indices[:k].tolist()
    if k < n and rng.random() < f:
        selected_indices.append(shuffled_indices[k])
    return selected_indices

def sample_split(dataset, rng, args):
    sample_indices = [*range(len(dataset))]
    selected_indices = sorted(select_fraction(sample_indices, args.fraction, rng))
    file_name = f"{args.dataset}-smpl-f{str(args.fraction).replace('.', 'p')}-s{args.seed}.pkl"
    save_indices(selected_indices, len(dataset), args.output_dir, file_name)

def entity_split(dataset: datasets.EntityDataset, rng, args):
    entity_ids = dataset.unique_entity_ids.tolist()
    selected_entities = select_fraction(entity_ids, args.entity_fraction, rng)
    entity_index_table = dataset.get_entity_index_table()
    selected_indices = []
    for entity_id in selected_entities:
        selected_indices.extend(select_fraction(entity_index_table[entity_id], args.per_entity_fraction, rng))
    selected_indices.sort()
    file_name = f"{args.dataset}-ent-f{str(args.entity_fraction).replace('.', 'p')}-p{str(args.per_entity_fraction).replace('.', 'p')}-s{args.seed}.pkl"
    save_indices(selected_indices, len(dataset), args.output_dir, file_name)

def sample_complement(dataset, train_split_path, output_dir):
    train_split_path = Path(train_split_path)
    indices = load_indices(train_split_path, len_dataset=len(dataset))
    mask = index_to_mask(torch.tensor(indices), n_indices=len(dataset))
    complement_indices = mask_to_index(~mask).tolist()
    file_name = f"{train_split_path.stem}-comp.pkl"
    save_indices(complement_indices, len(dataset), output_dir, file_name)

def entity_complement(dataset: datasets.EntityDataset, train_split_path, output_dir, rng):
    train_split_path = Path(train_split_path)
    indices = load_indices(train_split_path, len_dataset=len(dataset))
    entity_ids = torch.unique(dataset.entity_ids[indices], sorted=True)
    complement_entity_ids = mask_to_index(~index_to_mask(entity_ids, dataset.n_entities)).tolist()
    entity_index_table = dataset.get_entity_index_table()
    m = re.search(r"p([01]p\d+)", str(train_split_path))
    assert m is not None
    per_entity_fraction = float(m[1].replace("p", "."))
    selected_indices = []
    for entity_id in complement_entity_ids:
        selected_indices.extend(select_fraction(entity_index_table[entity_id], per_entity_fraction, rng))
    selected_indices.sort()
    file_name = f"{train_split_path.stem}-comp.pkl"
    save_indices(selected_indices, len(dataset), output_dir, file_name)

def _build_parser():
    parser = argparse.ArgumentParser(description="Create training split index files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--output-dir", default="training/train_splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["sample", "entity", "complement", "entity-complement"], default="sample")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--entity-fraction", type=float, default=0.5)
    parser.add_argument("--per-entity-fraction", type=float, default=0.5)
    parser.add_argument("--train-split-path", type=str)
    return parser

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
    )
    rng = np.random.default_rng(seed=args.seed)
    match args.mode:
        case "sample":
            sample_split(dataset, rng, args)
        case "entity":
            assert isinstance(dataset, datasets.EntityDataset)
            entity_split(dataset, rng, args)
        case "complement":
            sample_complement(dataset, args.train_split_path, args.output_dir)
        case "entity-complement":
            entity_complement(dataset, args.train_split_path, args.output_dir, rng)

if __name__ == "__main__":
    main()
