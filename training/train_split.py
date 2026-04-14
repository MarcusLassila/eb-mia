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

def select_fraction(indices: list, fraction: float, rng: np.random.Generator, hold_out_fraction: float = 0.0):
    n = len(indices)
    if hold_out_fraction > 0.0:
        n = int(n * (1.0 - hold_out_fraction))
    f, k = modf(n * fraction)
    k = int(k)
    shuffled_indices = rng.permutation(indices[:n])
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
    entity_ids = list(range(dataset.n_entities))
    selected_entities = select_fraction(entity_ids, args.entity_fraction, rng)
    entity_index_table = dataset.get_entity_index_table()
    selected_indices = []
    for entity_id in selected_entities:
        selected_indices.extend(select_fraction(
            entity_index_table[entity_id],
            args.per_entity_fraction,
            rng,
            hold_out_fraction=args.per_entity_hold_out,
        ))
    selected_indices.sort()
    hold_out_tag = ""
    if args.per_entity_hold_out != 0.0:
        hold_out_tag = f"-h{str(args.per_entity_hold_out).replace('.', 'p')}"
    file_name = (
        f"{args.dataset}-ent"
        f"-f{str(args.entity_fraction).replace('.', 'p')}"
        f"-p{str(args.per_entity_fraction).replace('.', 'p')}"
        f"{hold_out_tag}"
        f"-s{args.seed}.pkl"
    )
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
    match_obj = re.search(r"-p([01]p\d+)(?:-h([01]p\d+))?", train_split_path.stem)
    assert match_obj is not None
    match match_obj.groups():
        case (per_entity_fraction, hold_out_fraction):
            per_entity_fraction = float(per_entity_fraction.replace("p", "."))
            hold_out_fraction = float(hold_out_fraction.replace("p", ".")) if hold_out_fraction is not None else 0.0
        case _:
            raise ValueError(f"Error while parsing train split path.")
    selected_indices = []
    for entity_id in complement_entity_ids:
        selected_indices.extend(select_fraction(
            entity_index_table[entity_id],
            per_entity_fraction,
            rng,
            hold_out_fraction=hold_out_fraction,
        ))
    selected_indices.sort()
    file_name = f"{train_split_path.stem}-comp.pkl"
    save_indices(selected_indices, len(dataset), output_dir, file_name)

def first_n_train_split(dataset, args):
    assert args.first_n_samples <= len(dataset)
    selected_indices = list(range(args.first_n_samples))
    file_name = f"{args.dataset}-smpl-first{args.first_n_samples}.pkl"
    save_indices(selected_indices, len(dataset), args.output_dir, file_name)

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
    parser.add_argument("--per-entity-hold-out", type=float, default=0.0)
    parser.add_argument("--first-n-samples", type=int)
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
            if args.first_n_samples is not None:
                first_n_train_split(dataset, args)
            else:
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
