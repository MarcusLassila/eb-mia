from pathlib import Path
import re

import utils


_LOSS_SIGNAL_PATTERN = re.compile(
    r"^(?P<signal_type>[A-Za-z0-9-]+)-signals-(?P<target_stem>.+)"
    r"-n(?P<n_loss_samples>\d+)-nl(?P<noise_level>\d+(?:p\d+)?)"
    r"(?:-t-error-step-size(?P<t_error_step_size>\d+))?(?:-dp(?P<n_data_points>\d+))?\.pkl$"
)
_LEGACY_LOSS_SIGNAL_PATTERN = re.compile(
    r"^loss_signals-(?P<target_stem>.+?)(?:-signal-(?P<signal_type>[A-Za-z0-9_]+))?"
    r"-ls(?P<n_loss_samples>\d+)-nl(?P<noise_level>\d+(?:p\d+)?)"
    r"(?:-dp(?P<n_data_points>\d+))?\.pkl$"
)

def loss_signals_dir(res_dir, target_path):
    '''Return the loss-signal directory for a target checkpoint.'''
    properties = utils.parse_properties_from_checkpoint_path(target_path)
    dataset_prefix = f"{properties['dataset']}-"
    split_info = properties["split"][len(dataset_prefix):]
    split_info = re.sub(r"-s\d+(-comp)?$", "", split_info)
    size_part = f"sz{properties['size']}"
    if properties["gray"]:
        size_part += "-gray"
    epoch = "none" if properties["epoch"] is None else properties["epoch"]
    result_name = "-".join([
        properties["model"],
        properties["dataset"],
        split_info,
        size_part,
        f"epoch{epoch}",
    ])
    return Path(res_dir) / result_name

def format_float_filename_token(value):
    '''Format a float as a stable filename token.'''
    token = f"{float(value):.15f}".rstrip("0").rstrip(".")
    return (token or "0").replace(".", "p")

def loss_signals_pickle_name(
    target_path,
    n_samples,
    noise_level=0.1,
    n_data_points=None,
    signal_type="loss",
    t_error_step_size=1,
):
    '''Build the current loss-signal pickle filename.'''
    noise_token = format_float_filename_token(noise_level)
    step_size_token = "" if t_error_step_size == 1 else f"-t-error-step-size{t_error_step_size}"
    data_token = "" if n_data_points is None else f"-dp{n_data_points}"
    signal_token = str(signal_type).replace("_", "-")
    target_stem = Path(target_path).stem
    return f"{signal_token}-signals-{target_stem}-n{n_samples}-nl{noise_token}{step_size_token}{data_token}.pkl"

def parse_loss_signal_path(loss_signal_path):
    '''Parse current or legacy loss-signal metadata from a path.'''
    loss_signal_path = Path(loss_signal_path)
    match = _LOSS_SIGNAL_PATTERN.match(loss_signal_path.name)
    if match is None:
        match = _LEGACY_LOSS_SIGNAL_PATTERN.match(loss_signal_path.name)
    if match is None:
        raise ValueError(f"Invalid loss-signal filename: {loss_signal_path.name}")

    metadata = match.groupdict()
    target_stem = metadata["target_stem"]
    target_properties = utils.parse_properties_from_checkpoint_path(target_stem)
    metadata.update(target_properties)
    metadata["target_stem"] = target_stem
    metadata["target_path"] = Path(f"{target_stem}.pth")
    metadata["n_loss_samples"] = int(metadata["n_loss_samples"])
    metadata["noise_level"] = float(metadata["noise_level"].replace("p", "."))
    metadata["t_error_step_size"] = int(metadata["t_error_step_size"]) if metadata.get("t_error_step_size") is not None else 1
    if metadata["n_data_points"] is not None:
        metadata["n_data_points"] = int(metadata["n_data_points"])
    metadata["signal_type"] = (metadata["signal_type"] or "loss").replace("-", "_")
    return metadata

def resolve_audit_loss_signal_paths(config, key):
    '''Resolve configured loss-signal files or directories to pickle paths.'''
    configured_paths = getattr(config, key, None)
    if not configured_paths:
        raise ValueError(f"Audit config requires {key}.")
    if isinstance(configured_paths, (str, Path)):
        configured_paths = [configured_paths]

    repo_root = Path(utils.get_root())
    resolved_paths = []
    for configured_path in configured_paths:
        candidate = Path(configured_path).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.is_file():
            candidates = [candidate]
        elif candidate.is_dir():
            candidates = candidate.rglob("*.pkl")
        else:
            raise ValueError(f"Path does not exist: {candidate}")
        for loss_signal_path in candidates:
            try:
                parse_loss_signal_path(loss_signal_path)
            except ValueError:
                continue
            resolved_paths.append(loss_signal_path.resolve())
    resolved_paths = sorted(set(resolved_paths))
    if not resolved_paths:
        raise ValueError(f"No loss-signal pickle files found for {key}.")
    return resolved_paths

def audit_results_dir(results_dir_name, results_root=None):
    '''Resolve a named audit output directory under the configured results root.'''
    if Path(results_dir_name).name != results_dir_name:
        raise ValueError("results_dir_name must be a directory name")
    if results_root is None:
        repo_root = Path(utils.get_root())
        results_root = repo_root / "results" / "mia_audit"
    return Path(results_root).expanduser().resolve() / results_dir_name

def metrics_pickle_name(target_stem, attack):
    '''Build a flat metrics filename for one target and attack.'''
    return f"metrics_attack-{attack}_target-{target_stem}.pkl"
