import argparse
from pathlib import Path

from data.utils import load_dataset
import utils

import torch


def _canonical_dataset_name(dataset_name):
    '''Map parsed dataset names to load_dataset names. Args: dataset_name (str). Returns: str.'''
    mapping = {
        "mnist": "MNIST",
        "cifar10": "CIFAR10",
        "celeba": "CelebA",
        "celeba2": "CelebA2",
        "celebahq": "CelebAHQ",
        "flowers": "Flowers",
    }
    return mapping.get(dataset_name.lower(), dataset_name)


def _collect_checkpoint_paths(checkpoints_dir, pattern, recursive):
    '''Collect checkpoint paths from a folder. Args: checkpoints_dir (str|Path), pattern (str), recursive (bool). Returns: list[Path].'''
    checkpoints_dir = Path(checkpoints_dir).resolve()
    if not checkpoints_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoints_dir}")
    if not checkpoints_dir.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {checkpoints_dir}")
    if recursive:
        checkpoint_paths = sorted(path for path in checkpoints_dir.rglob(pattern) if path.is_file())
    else:
        checkpoint_paths = sorted(path for path in checkpoints_dir.glob(pattern) if path.is_file())
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {checkpoints_dir} with pattern '{pattern}'.")
    return checkpoint_paths


def verify_shadow_split_balance(checkpoints_dir, data_dir="./datasets", pattern="*.pth", recursive=False):
    '''Verify each dataset sample is in exactly half of shadow model train_indices. Args: checkpoints_dir (str|Path), data_dir (str|Path), pattern (str), recursive (bool). Returns: dict.'''
    checkpoint_paths = _collect_checkpoint_paths(checkpoints_dir, pattern=pattern, recursive=recursive)
    if len(checkpoint_paths) % 2 != 0:
        raise ValueError(f"Expected an even number of checkpoints, got {len(checkpoint_paths)}.")

    parsed = [utils.parse_properties_from_checkpoint_path(path) for path in checkpoint_paths]
    first_meta = parsed[0]
    required_meta = (first_meta["dataset"], first_meta["size"], first_meta["gray"])
    for path, meta in zip(checkpoint_paths[1:], parsed[1:]):
        compare_meta = (meta["dataset"], meta["size"], meta["gray"])
        if compare_meta != required_meta:
            raise ValueError(
                "All checkpoints must share dataset/size/grayscale, "
                f"but {path} has {(meta['dataset'], meta['size'], meta['gray'])} and expected {required_meta}."
            )

    dataset_name = _canonical_dataset_name(first_meta["dataset"])
    dataset = load_dataset(
        dataset_name,
        data_dir=str(data_dir),
        size=first_meta["size"],
        grayscale=first_meta["gray"],
    )
    n_samples = len(dataset)
    expected_inclusions = len(checkpoint_paths) // 2

    in_counts = torch.zeros(n_samples, dtype=torch.long)
    for checkpoint_path in checkpoint_paths:
        train_indices = torch.as_tensor(utils.get_train_indices(str(checkpoint_path)), dtype=torch.long)
        if train_indices.numel() == 0:
            continue
        unique_train_indices = torch.unique(train_indices)
        if int(unique_train_indices.min().item()) < 0 or int(unique_train_indices.max().item()) >= n_samples:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has out-of-range train index for dataset of size {n_samples}."
            )
        in_mask = utils.index_to_mask(unique_train_indices, n_samples)
        in_counts += in_mask.to(dtype=torch.long)

    mismatched_mask = in_counts != expected_inclusions
    mismatched_indices = torch.nonzero(mismatched_mask, as_tuple=True)[0]
    summary = {
        "is_balanced": len(mismatched_indices) == 0,
        "dataset": dataset_name,
        "dataset_size": n_samples,
        "n_checkpoints": len(checkpoint_paths),
        "expected_inclusions_per_sample": expected_inclusions,
        "mismatched_indices": mismatched_indices.tolist(),
    }
    if len(mismatched_indices):
        summary["mismatched_counts"] = {int(idx.item()): int(in_counts[idx].item()) for idx in mismatched_indices}
    return summary


def _build_parser():
    '''Build CLI parser for shadow split balance check. Args: none. Returns: argparse.ArgumentParser.'''
    parser = argparse.ArgumentParser(
        description=(
            "Verify that each dataset sample is included in exactly half of the checkpoints' train_indices."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        required=True,
        help="Directory containing checkpoint files.",
    )
    parser.add_argument(
        "--data-dir",
        default="./datasets",
        help="Dataset cache/root directory used by data loaders.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pth",
        help="Glob pattern for checkpoint files inside checkpoints-dir (default: %(default)s).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively in checkpoints-dir.",
    )
    return parser


def main(argv=None):
    '''Run CLI for shadow split balance verification. Args: argv (list[str]|None). Returns: int.'''
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = verify_shadow_split_balance(
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    print("")
    print("Shadow split balance check")
    print(f"dataset: {summary['dataset']}")
    print(f"dataset_size: {summary['dataset_size']}")
    print(f"n_checkpoints: {summary['n_checkpoints']}")
    print(f"expected_inclusions_per_sample: {summary['expected_inclusions_per_sample']}")
    if summary["is_balanced"]:
        print("status: PASS")
        return 0
    print("status: FAIL")
    print(f"n_mismatched_samples: {len(summary['mismatched_indices'])}")
    preview = summary["mismatched_indices"][:20]
    print(f"mismatched_sample_indices_preview: {preview}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
