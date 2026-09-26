import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from . import evaluation
from . import path_utils
from . import result_store


def load_batch_config(config_path):
    '''
    Load a multiple-audit YAML configuration.
    Args:
        config_path (str | Path): Batch configuration path.
    Returns:
        dict: Parsed batch configuration.
    '''
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Multiple-audit config must contain a mapping.")
    return config

def build_audit_configs(batch_config):
    '''
    Expand attacks and loss paths into single-audit configurations.
    Args:
        batch_config (dict): Shared settings and attack configurations.
    Returns:
        list[dict]: Single-attack configurations for run_audit.
    '''
    common_config = dict(batch_config)
    attacks = common_config.pop("attacks", None)
    audit_groups = common_config.pop("audit_groups", None)
    audit_mode = common_config.get("audit_mode")
    if audit_mode not in {"sample", "entity"}:
        raise ValueError("Multiple-audit config requires audit_mode sample or entity.")
    results_dir_name = common_config.get("results_dir_name")
    if not isinstance(results_dir_name, str) or Path(results_dir_name).name != results_dir_name:
        raise ValueError("Multiple-audit config requires a results_dir_name.")
    if not isinstance(attacks, list) or not attacks:
        raise ValueError("Multiple-audit config requires attacks.")
    common_config.pop("target_loss_paths", None)
    common_config.pop("shadow_loss_paths", None)
    common_loss_paths = common_config.pop("loss_paths", None)
    common_loss_path_pairs = common_config.pop("loss_path_pairs", None)
    if audit_groups is None:
        audit_groups = [{
            "loss_paths": common_loss_paths,
            "loss_path_pairs": common_loss_path_pairs,
        }]
    elif common_loss_paths is not None or common_loss_path_pairs is not None:
        raise ValueError("Define loss paths in audit_groups or at the top level, not both.")
    audit_configs = []
    attack_names = set()
    for attack_config in attacks:
        attack = attack_config.get("attack")
        attack_name = attack_config.get("name", attack)
        if attack_name in attack_names:
            raise ValueError(f"Attack names must be unique: {attack_name}")
        attack_names.add(attack_name)
    supported_entity_modes = {"all", "max_one_train", "exclude_train", "hold_out"}
    for audit_group in audit_groups:
        group_config = dict(common_config)
        group_config.update(audit_group)
        loss_paths = group_config.pop("loss_paths", None)
        loss_path_pairs = group_config.pop("loss_path_pairs", None)
        if audit_mode == "entity":
            dataset = group_config.get("dataset")
            entity_mode = group_config.get("mode")
            if not isinstance(dataset, str) or not dataset:
                raise ValueError("Entity multiple-audit config requires a dataset.")
            if entity_mode not in supported_entity_modes:
                raise ValueError("Entity multiple-audit config requires a supported entity mode.")
        if loss_paths is not None and loss_path_pairs is not None:
            raise ValueError("Audit group must define loss_paths or loss_path_pairs, not both.")
        if loss_paths is None and loss_path_pairs is None:
            raise ValueError("Audit group requires loss_paths or loss_path_pairs.")
        if loss_paths is not None:
            loss_path_pairs = [
                {
                    "target_loss_paths": [loss_path],
                    "shadow_loss_paths": [loss_path],
                }
                for loss_path in loss_paths
            ]
        for loss_path_pair in loss_path_pairs:
            for attack_config in attacks:
                audit_config = dict(group_config)
                audit_config["target_loss_paths"] = loss_path_pair["target_loss_paths"]
                audit_config["shadow_loss_paths"] = loss_path_pair["shadow_loss_paths"]
                audit_config["reference_protocol"] = loss_path_pairs
                audit_config["attack"] = dict(attack_config)
                audit_config["print_summary"] = True
                audit_configs.append(audit_config)
    return audit_configs

def run_audits(audit_configs):
    '''
    Execute each audit configuration in an isolated subprocess.
    Args:
        audit_configs (list[dict]): Single-attack audit configurations.
    Returns:
        None
    '''
    results_by_directory = {}
    with tempfile.TemporaryDirectory(prefix="eb-mia-audits-") as temp_dir:
        temp_dir = Path(temp_dir)
        for index, audit_config in enumerate(audit_configs, start=1):
            audit_config = dict(audit_config)
            manifest_path = temp_dir / f"audit-{index}.json"
            audit_config["manifest_path"] = str(manifest_path)
            attack_config = audit_config["attack"]
            attack_name = attack_config.get("name", attack_config["attack"])
            loss_path = audit_config["target_loss_paths"][0]
            config_path = temp_dir / f"audit-{index}.yaml"
            with open(config_path, "w") as file:
                yaml.safe_dump(audit_config, file, sort_keys=False)
            print(
                f"Running audit {index}/{len(audit_configs)}: {attack_name} on {loss_path}",
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "mia.run_audit",
                "--config",
                str(config_path),
            ]
            subprocess.run(command, check=True)
            metrics_paths = result_store.load_manifest_paths(manifest_path)
            results_root = audit_config.get("results_root")
            results_dir = path_utils.audit_results_dir(audit_config["results_dir_name"], results_root)
            results_by_directory.setdefault(results_dir, []).extend(metrics_paths)
    for results_dir, metrics_paths in results_by_directory.items():
        resolved_paths = [str(path.resolve()) for path in metrics_paths]
        if len(set(resolved_paths)) != len(resolved_paths):
            raise ValueError(f"Duplicate audit settings in batch results for {results_dir}.")
        manifest = {"schema_version": 2, "metrics_paths": resolved_paths}
        result_store.write_artifact(results_dir / "audit_manifest.json", manifest)

def parse_args(argv=None):
    '''
    Parse command-line arguments.
    Args:
        argv (list[str] | None): Optional argument list.
    Returns:
        argparse.Namespace: Parsed arguments.
    '''
    parser = argparse.ArgumentParser(
        description="Run multiple sample- or entity-level membership audits."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to multiple-audit YAML config.")
    return parser.parse_args(argv)

def main(argv=None):
    '''
    Load, expand, and execute a multiple-audit configuration.
    Args:
        argv (list[str] | None): Optional argument list.
    Returns:
        None
    '''
    args = parse_args(argv)
    batch_config = load_batch_config(args.config)
    audit_configs = build_audit_configs(batch_config)
    run_audits(audit_configs)
    result_dirs = []
    for audit_config in audit_configs:
        results_root = audit_config.get("results_root")
        results_dir = path_utils.audit_results_dir(audit_config["results_dir_name"], results_root)
        if results_dir not in result_dirs:
            result_dirs.append(results_dir)
    for results_dir in result_dirs:
        print(f"\nResults: {results_dir}")
        evaluation.print_grouped_metrics(
            results_dir,
            show_auc=bool(batch_config.get("show_auc", False)),
        )
        evaluation.plot_average_roc_curves(results_dir)

if __name__ == "__main__":
    main()
