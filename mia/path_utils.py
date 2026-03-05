from pathlib import Path
import re

import utils

_SCORES_FILENAME_RE = re.compile(r"^scores_attack-(?P<attack>.+)_target-(?P<target>.+)\.pkl$")
_METRICS_FILENAME_RE = re.compile(r"^metrics_attack-(?P<attack>.+)_target-(?P<target>.+)_mode-(?P<mode>.+)\.pkl$")

def attack_name(attack, offline=False):
    '''Return attack name with offline suffix when requested. Args: attack (str), offline (bool). Returns: str.'''
    return f"{attack}-off" if offline else attack

def parse_attack_name(attack):
    '''Split attack name into base name and offline flag. Args: attack (str). Returns: tuple[str, bool].'''
    suffix = "-off"
    if attack.endswith(suffix):
        return attack[:-len(suffix)], True
    return attack, False

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

def _parse_scores_filename(path):
    '''Parse a score pickle filename. Args: path (str|Path). Returns: dict.'''
    match = _SCORES_FILENAME_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse score filename: {path}")
    return {
        "attack": match.group("attack"),
        "target_stem": match.group("target"),
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

def resolve_audit_score_paths(config):
    '''Resolve audit score pickle files from configured files/folders. Args: config (Config). Returns: list[Path].'''
    root = utils.get_root()
    res_dir = utils.resolve_path(config.res_dir, root)
    score_paths = getattr(config, "score_paths", None)
    if not score_paths:
        raise ValueError("No score paths specified in config or CLI.")
    input_paths = [utils.resolve_path(path, res_dir) for path in score_paths]
    score_files = []
    for input_path in input_paths:
        input_path = Path(input_path)
        if input_path.is_file():
            score_files.append(input_path.resolve())
        elif input_path.is_dir():
            score_files.extend(path.resolve() for path in sorted(input_path.rglob("scores_attack-*_target-*.pkl")))
        else:
            raise ValueError(f"Score path does not exist: {input_path}")
    score_files = sorted(set(score_files))
    if not score_files:
        raise ValueError("No score pickle files found.")
    return score_files

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
