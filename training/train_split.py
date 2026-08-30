from data import datasets
from data.utils import load_dataset
from utils import index_to_mask, mask_to_index

import argparse
import pickle
import numpy as np
import torch
from math import ceil, modf
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
        n = int(ceil(n * (1.0 - hold_out_fraction)))
    f, k = modf(n * fraction)
    k = int(k)
    shuffled_indices = rng.permutation(indices[:n])
    selected_indices = shuffled_indices[:k].tolist()
    if k < n and rng.random() < f:
        selected_indices.append(shuffled_indices[k])
    return selected_indices

def sample_split(dataset, rng, args, seed):
    sample_indices = [*range(len(dataset))]
    selected_indices = sorted(select_fraction(sample_indices, args.fraction, rng))
    full_output_dir = f"{args.output_dir}/{args.dataset}/sample"
    file_name = f"{args.dataset}-smpl-f{str(args.fraction).replace('.', 'p')}-s{seed}.pkl"
    save_indices(selected_indices, len(dataset), full_output_dir, file_name)
    if args.make_complement_splits:
        mask = index_to_mask(torch.tensor(selected_indices), n_indices=len(dataset))
        comp_indices = mask_to_index(~mask).tolist()
        comp_file_name = f"{Path(file_name).stem}-comp.pkl"
        save_indices(comp_indices, len(dataset), full_output_dir, comp_file_name)

def entity_split(dataset: datasets.EntityDataset, rng, args, seed):
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
        f"-s{seed}.pkl"
    )
    full_output_dir = f"{args.output_dir}/{args.dataset}/entity"
    save_indices(selected_indices, len(dataset), full_output_dir, file_name)
    if args.make_complement_splits:
        entity_complement(dataset, indices=selected_indices,output_dir=full_output_dir, file_name_stem=Path(file_name).stem, rng=rng, args=args)

def entity_complement(dataset: datasets.EntityDataset, indices, output_dir, file_name_stem, rng, args):
    entity_ids = torch.unique(dataset.entity_ids[indices], sorted=True)
    complement_entity_ids = mask_to_index(~index_to_mask(entity_ids, dataset.n_entities)).tolist()
    entity_index_table = dataset.get_entity_index_table()
    selected_indices = []
    for entity_id in complement_entity_ids:
        selected_indices.extend(select_fraction(
            entity_index_table[entity_id],
            args.per_entity_fraction,
            rng,
            hold_out_fraction=args.per_entity_hold_out,
        ))
    selected_indices.sort()
    file_name = f"{file_name_stem}-comp.pkl"
    save_indices(selected_indices, len(dataset), output_dir, file_name)

def first_n_train_split(dataset, args):
    assert args.first_n_samples <= len(dataset)
    selected_indices = list(range(args.first_n_samples))
    full_output_dir = f"{args.output_dir}/{args.dataset}"
    file_name = f"{args.dataset}-smpl-first{args.first_n_samples}.pkl"
    save_indices(selected_indices, len(dataset), full_output_dir, file_name)

def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--output-dir", default="training/train_splits")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--mode", choices=["sample", "entity", "first-n"], default="sample")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--entity-fraction", type=float, default=0.5)
    parser.add_argument("--per-entity-fraction", type=float, default=0.5)
    parser.add_argument("--per-entity-hold-out", type=float, default=0.0)
    parser.add_argument("--first-n-samples", type=int)
    parser.add_argument("--make-complement-splits", action="store_true")
    return parser

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
    )
    if args.mode == "first-n":
        assert args.first_n_samples >= 0
        first_n_train_split(dataset, args)
    else:
        seeds = [int(seed) for seed in args.seeds.split(",")]
        for seed in seeds:
            rng = np.random.default_rng(seed=seed)
            if args.mode == "sample":
                sample_split(dataset, rng, args, seed)
            elif args.mode == "entity":
                assert isinstance(dataset, datasets.EntityDataset)
                entity_split(dataset, rng, args, seed)
            else:
                raise ValueError("Unsupported mode.")

if __name__ == "__main__":
    main()
