from pathlib import Path
import re

import utils

_SCORES_FILENAME_RE = re.compile(r"^scores_attack-(?P<attack>.+)_target-(?P<target>.+)\.pkl$")
_LOSS_SIGNALS_FILENAME_RE = re.compile(r"^loss_signals-(?P<target>.+)-ls(?P<n_loss_samples>\d+)\.pkl$")
_METRICS_FILENAME_RE = re.compile(r"^metrics_attack-(?P<attack>.+)_target-(?P<target>.+)_mode-(?P<mode>.+)\.pkl$")

def audit_result_name(target_path):
    '''Return canonical audit result folder name without attack prefix. Args: target_path (str|Path). Returns: str.'''
    props = utils.parse_properties_from_checkpoint_path(target_path)
    split_stem = props["split"]
    dataset_prefix = f"{props['dataset']}-"
    assert split_stem.startswith(dataset_prefix)
    split_info_with_seed = split_stem[len(dataset_prefix):]
    split_info = re.sub(r"-s\d+(-comp)?$", "", split_info_with_seed)
    size_part = f"sz{props['size']}"
    if props["gray"]:
        size_part += "-gray"
    epoch_part = "none" if props["epoch"] is None else str(props["epoch"])
    return "-".join([
        props["model"],
        props["dataset"],
        split_info,
        size_part,
        f"epoch{epoch_part}",
    ])

def audit_result_dir(res_dir, target_path):
    '''Return canonical audit result directory without attack prefix. Args: res_dir (str|Path), target_path (str|Path). Returns: Path.'''
    return Path(res_dir) / audit_result_name(target_path)

def scores_dir(res_dir, attack, target_path):
    '''Return sample score directory with attack prefix. Args: res_dir (str|Path), attack (str), target_path (str|Path). Returns: Path.'''
    return audit_result_dir(res_dir, target_path) / f"{attack}-scores"

def scores_pickle_name(target_path, attack):
    '''Return normalized score pickle filename. Args: target_path (str|Path), attack (str). Returns: str.'''
    return "_".join([
        "scores",
        f"attack-{attack}",
        f"target-{Path(target_path).stem}",
    ]) + ".pkl"

def loss_signals_dir(res_dir, target_path):
    '''Return loss-signal directory for one checkpoint. Args: res_dir (str|Path), target_path (str|Path). Returns: Path.'''
    return audit_result_dir(res_dir, target_path) / "loss_signals"

def loss_signals_pickle_name(target_path, n_loss_samples):
    '''Return normalized loss-signal pickle filename. Args: target_path (str|Path), n_loss_samples (int). Returns: str.'''
    return f"loss_signals-{Path(target_path).stem}-ls{n_loss_samples}.pkl"

def _parse_scores_filename(path):
    '''Parse a score pickle filename. Args: path (str|Path). Returns: dict.'''
    match = _SCORES_FILENAME_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse score filename: {path}")
    return {
        "attack": match.group("attack"),
        "target_stem": match.group("target"),
    }

def _parse_loss_signals_filename(path):
    '''Parse a loss-signal pickle filename. Args: path (str|Path). Returns: dict.'''
    match = _LOSS_SIGNALS_FILENAME_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse loss-signal filename: {path}")
    return {
        "target_stem": match.group("target"),
        "n_loss_samples": int(match.group("n_loss_samples")),
    }

def infer_attack_from_score_paths(score_paths):
    '''Infer a unique sample attack from score pickle paths. Args: score_paths (list[Path]). Returns: str.'''
    attacks = sorted({score_attack_from_scores_pickle_path(path) for path in score_paths})
    if len(attacks) != 1:
        raise ValueError(f"Expected score files for one attack, got: {attacks}")
    return attacks[0]

def score_attack_from_scores_pickle_path(path):
    '''Extract attack from score pickle filename. Args: path (str|Path). Returns: str.'''
    return _parse_scores_filename(path)["attack"]

def score_pickle_paths(res_dir, attack):
    '''Return score pickle paths for one attack under res_dir. Args: res_dir (str|Path), attack (str). Returns: list[Path].'''
    res_dir = Path(res_dir)
    pattern = f"**/{attack}-scores/scores_attack-{attack}_target-*.pkl"
    return sorted(res_dir.glob(pattern))

def target_stem_from_loss_signals_pickle_path(path):
    '''Extract target checkpoint stem from a loss-signal pickle filename. Args: path (str|Path). Returns: str.'''
    return _parse_loss_signals_filename(path)["target_stem"]

def n_loss_samples_from_loss_signals_pickle_path(path):
    '''Extract loss-sampling count from a loss-signal pickle filename. Args: path (str|Path). Returns: int.'''
    return _parse_loss_signals_filename(path)["n_loss_samples"]

def target_checkpoint_path_from_loss_signals_pickle_path(path):
    '''Return a synthetic checkpoint path parsed from a loss-signal pickle filename. Args: path (str|Path). Returns: Path.'''
    return Path(f"{target_stem_from_loss_signals_pickle_path(path)}.pth")

def target_properties_from_loss_signals_pickle_path(path):
    '''Parse target model properties from a loss-signal pickle filename. Args: path (str|Path). Returns: dict.'''
    target_path = target_checkpoint_path_from_loss_signals_pickle_path(path)
    return utils.parse_properties_from_checkpoint_path(target_path)

def metrics_dir(res_dir, scores_path, audit_mode, entity_audit_mode=None, n_audit_samples_per_entity=None):
    '''Return audit metrics directory derived from a score path. Args: res_dir (str|Path), scores_path (str|Path), audit_mode (str), entity_audit_mode (str|None), n_audit_samples_per_entity (int|None). Returns: Path.'''
    if audit_mode == "entity":
        assert entity_audit_mode is not None
        folder_name = f"entity-{entity_audit_mode}"
        if n_audit_samples_per_entity is not None:
            folder_name += f"-n{n_audit_samples_per_entity}"
    else:
        folder_name = audit_mode
    score_meta = _parse_scores_filename(scores_path)
    target_path = Path(f"{score_meta['target_stem']}.pth")
    return audit_result_dir(res_dir, target_path) / f"{score_meta['attack']}-{folder_name}"

def target_stem_from_metrics_pickle_path(path):
    '''Extract target checkpoint stem from a metrics pickle filename. Args: path (str|Path). Returns: str.'''
    match = _METRICS_FILENAME_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse target model from metrics filename: {path}")
    return match.group("target")

def target_stem_from_scores_pickle_path(path):
    '''Extract target checkpoint stem from a score pickle filename. Args: path (str|Path). Returns: str.'''
    return _parse_scores_filename(path)["target_stem"]

def target_checkpoint_path_from_scores_pickle_path(path):
    '''Return a synthetic checkpoint path parsed from a score pickle filename. Args: path (str|Path). Returns: Path.'''
    return Path(f"{target_stem_from_scores_pickle_path(path)}.pth")

def target_properties_from_scores_pickle_path(path):
    '''Parse target model properties from a score pickle filename. Args: path (str|Path). Returns: dict.'''
    target_path = target_checkpoint_path_from_scores_pickle_path(path)
    return utils.parse_properties_from_checkpoint_path(target_path)

def metrics_pickle_name(scores_path, audit_mode, min_samples_per_entity=None, max_samples_per_entity=None, n_audit_samples_per_entity=None):
    '''Return metrics filename derived from a score filename. Args: scores_path (str|Path), audit_mode (str), min_samples_per_entity (int|None), max_samples_per_entity (int|None), n_audit_samples_per_entity (int|None). Returns: str.'''
    stem = Path(scores_path).stem
    if not stem.startswith("scores_"):
        raise ValueError(f"Could not parse score filename: {scores_path}")
    metrics_stem = "metrics_" + stem[len("scores_"):]
    metrics_stem += f"_mode-{audit_mode}"
    if audit_mode == "entity":
        if n_audit_samples_per_entity is not None:
            metrics_stem += f"_n-{n_audit_samples_per_entity}"
        min_value = "none" if min_samples_per_entity is None else str(min_samples_per_entity)
        max_value = "none" if max_samples_per_entity is None else str(max_samples_per_entity)
        metrics_stem += f"_min-{min_value}_max-{max_value}"
    return metrics_stem + ".pkl"

def metrics_dir_from_target(res_dir, target_path, attack, audit_mode, entity_audit_mode=None, n_audit_samples_per_entity=None):
    '''Return audit metrics directory derived from target checkpoint metadata. Args: res_dir (str|Path), target_path (str|Path), attack (str), audit_mode (str), entity_audit_mode (str|None), n_audit_samples_per_entity (int|None). Returns: Path.'''
    if audit_mode == "entity":
        assert entity_audit_mode is not None
        folder_name = f"entity-{entity_audit_mode}"
        if n_audit_samples_per_entity is not None:
            folder_name += f"-n{n_audit_samples_per_entity}"
    else:
        folder_name = audit_mode
    return audit_result_dir(res_dir, target_path) / f"{attack}-{folder_name}"

def metrics_pickle_name_from_target(target_path, attack, audit_mode, min_samples_per_entity=None, max_samples_per_entity=None, n_audit_samples_per_entity=None):
    '''Return metrics filename derived from a target checkpoint and attack name. Args: target_path (str|Path), attack (str), audit_mode (str), min_samples_per_entity (int|None), max_samples_per_entity (int|None), n_audit_samples_per_entity (int|None). Returns: str.'''
    metrics_stem = "_".join([
        "metrics",
        f"attack-{attack}",
        f"target-{Path(target_path).stem}",
        f"mode-{audit_mode}",
    ])
    if audit_mode == "entity":
        if n_audit_samples_per_entity is not None:
            metrics_stem += f"_n-{n_audit_samples_per_entity}"
        min_value = "none" if min_samples_per_entity is None else str(min_samples_per_entity)
        max_value = "none" if max_samples_per_entity is None else str(max_samples_per_entity)
        metrics_stem += f"_min-{min_value}_max-{max_value}"
    return metrics_stem + ".pkl"

def _resolve_pickle_paths(res_dir, input_paths, pattern, missing_message, empty_message):
    '''Resolve file and directory inputs to a sorted list of pickle files. Args: res_dir (str|Path), input_paths (list[str]|None), pattern (str), missing_message (str), empty_message (str). Returns: list[Path].'''
    if not input_paths:
        raise ValueError(missing_message)
    input_paths = [utils.resolve_path(path, res_dir) for path in input_paths]
    resolved_files = []
    seen_files = set()
    for input_path in input_paths:
        input_path = Path(input_path)
        if input_path.is_file():
            candidate_paths = [input_path.resolve()]
        elif input_path.is_dir():
            candidate_paths = [path.resolve() for path in sorted(input_path.rglob(pattern))]
        else:
            raise ValueError(f"Path does not exist: {input_path}")
        for candidate_path in candidate_paths:
            if candidate_path not in seen_files:
                seen_files.add(candidate_path)
                resolved_files.append(candidate_path)
    if not resolved_files:
        raise ValueError(empty_message)
    return resolved_files

def resolve_audit_score_paths(config):
    '''Resolve audit score pickle files from configured files/folders. Args: config (Config). Returns: list[Path].'''
    root = utils.get_root()
    res_dir = utils.resolve_path(config.res_dir, root)
    return _resolve_pickle_paths(
        res_dir,
        getattr(config, "score_paths", None),
        "scores_attack-*_target-*.pkl",
        "No score paths specified in config or CLI.",
        "No score pickle files found.",
    )

def resolve_audit_loss_signal_paths(config, key):
    '''Resolve audit loss-signal pickle files from configured files/folders. Args: config (Config), key (str). Returns: list[Path].'''
    root = utils.get_root()
    res_dir = utils.resolve_path(config.res_dir, root)
    return _resolve_pickle_paths(
        res_dir,
        getattr(config, key, None),
        "loss_signals-*.pkl",
        f"No {key} specified in config or CLI.",
        "No loss-signal pickle files found.",
    )

def common_target_meta(target_stems):
    '''Return shared target model metadata. Args: target_stems (iterable[str]). Returns: dict.'''
    target_stems = tuple(target_stems)
    if not target_stems:
        raise ValueError("No target models found.")
    parsed = [utils.parse_properties_from_checkpoint_path(f"{stem}.pth") for stem in target_stems]
    meta_set = {(p["model"], p["dataset"], p["size"], p["gray"]) for p in parsed}
    if len(meta_set) != 1:
        raise ValueError("Target models must share model, dataset and size.")
    model, dataset, size, gray = next(iter(meta_set))
    return {
        "model": model,
        "dataset": dataset,
        "size": size,
        "gray": gray,
    }

def metrics_folder_label(metrics_dir, common_meta):
    '''Return concise metrics folder label excluding model, dataset and size. Args: metrics_dir (str|Path), common_meta (dict). Returns: str.'''
    metrics_dir = Path(metrics_dir)
    parent_name = metrics_dir.parent.name
    folder_name = metrics_dir.name
    folder_match = re.match(r"^(?P<attack>.+)-(?P<mode>sample|entity-.+)$", folder_name)
    assert folder_match is not None
    attack = folder_match.group("attack")
    mode = folder_match.group("mode")
    n_audit_samples_per_entity = None
    entity_n_match = re.match(r"^(entity-.+)-n(?P<n>\d+)$", mode)
    if entity_n_match is not None:
        mode = entity_n_match.group(1)
        n_audit_samples_per_entity = int(entity_n_match.group("n"))
    if mode.startswith("entity-"):
        mode = "ent-" + mode[len("entity-"):]
    mode = mode.replace("max_one_train", "max_one")
    mode = mode.replace("exclude_train", "excl_train")
    size_label = f"sz{common_meta['size']}" + ("-gray" if common_meta["gray"] else "")
    pattern = (
        rf"^{re.escape(common_meta['model'])}-{re.escape(common_meta['dataset'])}-(?P<split_info>.+)-"
        rf"{re.escape(size_label)}-(?P<epoch>epoch.+)$"
    )
    match = re.match(pattern, parent_name)
    assert match is not None
    split_info = match.group("split_info")
    if split_info.startswith("ent-"):
        split_info = split_info[len("ent-"):]
    elif split_info.startswith("rand-"):
        split_info = split_info[len("rand-"):]
    epoch = re.sub(r"^epoch", "e", match.group("epoch"))
    label = f"{attack}-{split_info}-{epoch}-{mode}"
    if mode.startswith("ent-") and n_audit_samples_per_entity is not None:
        label += f"-n{n_audit_samples_per_entity}"
    return label

def resolve_evaluation_paths(config, metrics_folders_override=None):
    '''Resolve output audit dir and metric folder paths. Args: config (Config), metrics_folders_override (list[str]|None). Returns: tuple[Path, list[Path]].'''
    root = utils.get_root()
    output_dir = utils.resolve_path(getattr(config, "output_dir", config.res_dir), root)
    metrics_folders = metrics_folders_override if metrics_folders_override is not None else getattr(config, "metrics_folders", None)
    if not metrics_folders:
        raise ValueError("No metrics folders specified in config or CLI.")
    metrics_dirs = [utils.resolve_path(folder, output_dir) for folder in metrics_folders]
    return output_dir, metrics_dirs
