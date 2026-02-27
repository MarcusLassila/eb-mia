from data.utils import load_dataset
from training import train_split
from . import path_utils
import utils

import argparse
import pickle
from pathlib import Path

import torch

def _load_scores_list(scores_path):
    '''Load legacy scores as a python list. Args: scores_path (str|Path). Returns: list[float].'''
    with open(scores_path, "rb") as file:
        scores_payload = pickle.load(file)
    if isinstance(scores_payload, dict):
        if "scores" not in scores_payload:
            raise ValueError(f"Invalid score payload in {scores_path}.")
        scores_payload = scores_payload["scores"]
    if hasattr(scores_payload, "tolist"):
        scores_payload = scores_payload.tolist()
    return [float(score) for score in scores_payload]

def _resolve_score_files(scores_path):
    '''Resolve score files from a score file/folder path. Args: scores_path (str|Path). Returns: tuple[list[Path], Path].'''
    scores_path = Path(scores_path)
    if scores_path.is_file():
        return [scores_path.resolve()], scores_path.parent.resolve()
    if scores_path.is_dir():
        score_files = sorted(path.resolve() for path in scores_path.rglob("scores_attack-*_target-*.pkl"))
        return score_files, scores_path.resolve()
    raise ValueError(f"Score path does not exist: {scores_path}")

def _resolve_indices_source(scores_path, indices_source_path):
    '''Resolve checkpoint/split path to recover train indices. Args: scores_path (str|Path), indices_source_path (str|Path). Returns: Path.'''
    indices_source_path = Path(indices_source_path)
    if indices_source_path.is_file():
        return indices_source_path.resolve()
    if not indices_source_path.is_dir():
        raise ValueError(f"Indices source path does not exist: {indices_source_path}")
    target_stem = path_utils.target_stem_from_scores_pickle_path(scores_path)
    target_candidates = sorted(indices_source_path.rglob(f"{target_stem}.pth"))
    if len(target_candidates) == 1:
        return target_candidates[0].resolve()
    if len(target_candidates) > 1:
        raise ValueError(f"Found multiple matching checkpoint paths for {scores_path}.")
    target_props = path_utils.target_properties_from_scores_pickle_path(scores_path)
    split_candidates = sorted(indices_source_path.rglob(f"{target_props['split']}.pkl"))
    if len(split_candidates) == 1:
        return split_candidates[0].resolve()
    if len(split_candidates) > 1:
        raise ValueError(f"Found multiple matching split paths for {scores_path}.")
    raise ValueError(f"Could not find matching checkpoint/split path for {scores_path}.")

def _load_train_indices(indices_source_path):
    '''Load train indices from checkpoint or split file. Args: indices_source_path (str|Path). Returns: torch.Tensor.'''
    indices_source_path = Path(indices_source_path)
    if indices_source_path.suffix == ".pth":
        train_indices = utils.get_train_indices(indices_source_path)
    elif indices_source_path.suffix == ".pkl":
        train_indices = train_split.load_indices(indices_source_path)
    else:
        raise ValueError(f"Unsupported train index source file type: {indices_source_path}")
    return torch.as_tensor(train_indices, dtype=torch.long)

def convert_legacy_scores(scores_path, indices_source_path, data_dir, output_dir=None):
    '''Convert legacy score file(s) to {scores, train_mask} format. Args: scores_path (str|Path), indices_source_path (str|Path), data_dir (str|Path), output_dir (str|Path|None). Returns: list[Path].'''
    score_files, score_root = _resolve_score_files(scores_path)
    if not score_files:
        raise ValueError(f"No score files found in {scores_path}.")
    output_root = None if output_dir is None else Path(output_dir).resolve()
    output_paths = []
    for score_file in score_files:
        scores = _load_scores_list(score_file)
        source_path = _resolve_indices_source(score_file, indices_source_path)
        train_indices = _load_train_indices(source_path)
        target_props = path_utils.target_properties_from_scores_pickle_path(score_file)
        dataset = load_dataset(
            target_props["dataset"],
            data_dir=data_dir,
            size=target_props["size"],
            grayscale=target_props["gray"],
        )
        assert len(scores) == len(dataset)
        train_mask = utils.index_to_mask(train_indices, len(dataset)).to(dtype=torch.long).tolist()
        output_payload = {"scores": scores, "train_mask": train_mask}
        if output_root is None:
            output_path = score_file
        else:
            relative = score_file.relative_to(score_root)
            output_path = output_root / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file:
            pickle.dump(output_payload, file)
        output_paths.append(output_path)
    return output_paths

def parse_args(argv=None):
    '''Parse conversion CLI arguments. Args: argv (list[str]|None). Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Convert legacy score pickle files to include train_mask.")
    parser.add_argument("--scores-path", required=True, help="Legacy score pickle file or directory.")
    parser.add_argument("--indices-source-path", required=True, help="Target checkpoint/split path or directory.")
    parser.add_argument("--data-dir", default="./datasets", help="Dataset directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to in-place overwrite.")
    return parser.parse_args(argv)

def main(argv=None):
    '''Run legacy score conversion CLI. Args: argv (list[str]|None). Returns: None.'''
    args = parse_args(argv)
    output_paths = convert_legacy_scores(
        scores_path=args.scores_path,
        indices_source_path=args.indices_source_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(f"Converted {len(output_paths)} score file(s).")

if __name__ == "__main__":
    main()
