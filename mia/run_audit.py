from data.dataset_metadata import entity_index_table_from_entity_ids, load_dataset_metadata
from . import attacks
from . import evaluation
from . import path_utils
from .utils import indices_of_shadow_models, select_sample_audit_indices, select_entity_audit_indices, load_loss_signals
import utils

import argparse
import torch
from tqdm.auto import tqdm

import pickle
import yaml
import numpy as np

def attack_config_from_config(config):
    '''
    Return the attack config as a `Config` object.
    Args:
        config (Config): Audit configuration.
    Returns:
        Config: Attack configuration.
    '''
    attack_config = getattr(config, "attack", None)
    if attack_config is None:
        raise ValueError("Audit config must define attack settings.")
    if isinstance(attack_config, dict):
        return utils.Config(attack_config)
    return attack_config

def get_attacker(attack_config, shadow_loss_sigs, shadow_train_mask):
    '''
    Instantiate the configured attack from shadow loss signals.
    Args:
        attack_config (Config): Attack configuration.
        shadow_loss_sigs (torch.Tensor): Shadow-model loss signals.
        shadow_train_mask (torch.Tensor): Shadow-model membership masks.
    Returns:
        MIA: Instantiated attacker.
    '''
    return getattr(attacks, attack_config.attack)(shadow_loss_sigs, shadow_train_mask, **attack_config.__dict__)

def resolve_target_shadow_paths(config):
    '''
    Resolve per-target shadow loss-signal path groups for auditing.
    Args:
        config (Config): Audit configuration.
    Returns:
        tuple[list[Path], list[list[Path]]]: Target paths and grouped shadow paths.
    '''
    target_loss_paths = path_utils.resolve_audit_loss_signal_paths(config, "target_loss_paths")
    if getattr(config, "round_robin", False):
        target_train_masks = []
        for target_loss_path in target_loss_paths:
            _, train_mask = load_loss_signals(target_loss_path)
            target_train_masks.append(train_mask)
        target_train_masks = torch.stack(target_train_masks)
        shadow_path_groups = []
        for target_idx in range(len(target_loss_paths)):
            shadow_indices = indices_of_shadow_models(target_idx, target_train_masks)
            shadow_path_groups.append([target_loss_paths[idx] for idx in shadow_indices])
        return target_loss_paths, shadow_path_groups
    shadow_loss_paths = path_utils.resolve_audit_loss_signal_paths(config, "shadow_loss_paths")
    return target_loss_paths, [shadow_loss_paths for _ in target_loss_paths]

def load_shadow_signals(shadow_loss_paths, target_len):
    '''
    Load and stack shadow loss signals and masks.
    Args:
        shadow_loss_paths (list[Path]): Shadow loss-signal paths.
        target_len (int): Expected loss-signal length.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Stacked loss signals and masks.
    '''
    shadow_loss_sigs = []
    shadow_train_mask = []
    for shadow_loss_path in shadow_loss_paths:
        loss_sig, train_mask = load_loss_signals(shadow_loss_path)
        if len(loss_sig) != target_len:
            raise ValueError(f"Unexpected loss-signal length in {shadow_loss_path}: got {len(loss_sig)}, expected {target_len}.")
        shadow_loss_sigs.append(loss_sig)
        shadow_train_mask.append(train_mask)
    return torch.stack(shadow_loss_sigs), torch.stack(shadow_train_mask)

def validate_entity_hold_out_indices(entity_index_table, loss_paths, target_len, hold_out_frac):
    '''
    Validate that entity hold-out indices are unused by every model.
    Args:
        entity_index_table (dict[int, list[int]]): Population indices for each entity.
        loss_paths (list[Path]): Loss-signal paths for all audited models.
        target_len (int): Expected number of samples per train mask.
        hold_out_frac (float): Fraction of each entity reserved for hold-out.
    Returns:
        None
    '''
    if hold_out_frac <= 0.0:
        return
    all_train_masks = []
    for loss_path in loss_paths:
        _, train_mask = load_loss_signals(loss_path)
        if len(train_mask) != target_len:
            raise ValueError(f"Unexpected train-mask length in {loss_path}: got {len(train_mask)}, expected {target_len}.")
        all_train_masks.append(train_mask)
    stacked_train_masks = torch.stack(all_train_masks)
    for entity_id, entity_indices in entity_index_table.items():
        n_train_candidates = int(len(entity_indices) * (1.0 - hold_out_frac))
        hold_out_indices = entity_indices[n_train_candidates:]
        if not hold_out_indices:
            continue
        hold_out_train_mask = stacked_train_masks[:, hold_out_indices]
        if torch.any(hold_out_train_mask):
            used_index_mask = hold_out_train_mask.any(dim=0)
            used_model_mask = hold_out_train_mask.any(dim=1)
            used_indices = torch.tensor(hold_out_indices, dtype=torch.long)[used_index_mask].tolist()
            used_paths = [str(loss_paths[idx]) for idx in utils.mask_to_index(used_model_mask).tolist()]
            raise ValueError(
                f"Entity hold-out indices must be unused by all models. "
                f"Entity {entity_id} hold-out indices {used_indices} appear in train masks for {used_paths}."
            )

def print_average_metrics_table(attack, metrics_list):
    '''
    Print mean audit metrics over target models.
    Args:
        attack (str): Attack name shown in the summary.
        metrics_list (list[dict]): Metrics for each target model.
    Returns:
        None
    '''
    mean_auc = float(np.mean([metrics["AUC"] for metrics in metrics_list]))
    mean_tpr_1pct = float(np.mean([metrics["TPR@1%FPR"] for metrics in metrics_list]))
    mean_tpr_0p1pct = float(np.mean([metrics["TPR@0.1%FPR"] for metrics in metrics_list]))
    mean_num_audit_points = float(np.mean([metrics["n_audit_points"] for metrics in metrics_list]))
    print("")
    print(f"Audit summary ({attack})")
    print(f"{'Metric':<16} {'Mean':>10}")
    print(f"{'-' * 16} {'-' * 10}")
    print(f"{'AUC':<16} {mean_auc:>10.4f}")
    print(f"{'TPR@1%FPR':<16} {mean_tpr_1pct:>10.4f}")
    print(f"{'TPR@0.1%FPR':<16} {mean_tpr_0p1pct:>10.4f}")
    print(f"{'n_audit_points':<16} {mean_num_audit_points:>10.4f}")

def run_sample_audit(config):
    attack_config = attack_config_from_config(config)
    attack = getattr(attack_config, "name", attack_config.attack)
    target_loss_paths, shadow_path_groups = resolve_target_shadow_paths(config)
    all_metrics = []
    for target_loss_path, shadow_loss_paths in tqdm(
        zip(target_loss_paths, shadow_path_groups),
        total=len(target_loss_paths),
        desc="Running sample-level audit",
    ):
        target_loss_sigs, target_train_mask = load_loss_signals(target_loss_path)
        n_population = len(target_loss_sigs)
        shadow_loss_sigs, shadow_train_mask = load_shadow_signals(shadow_loss_paths, len(target_loss_sigs))
        attacker = get_attacker(attack_config, shadow_loss_sigs, shadow_train_mask)
        scores = attacker.run_attack(target_loss_sigs)
        audit_indices = select_sample_audit_indices(
            getattr(config, "n_audit_samples", n_population),
            target_train_mask,
        )
        ground_truth = target_train_mask.to(dtype=torch.long)[audit_indices]
        audit_scores = scores[audit_indices]
        metrics = evaluation.evaluate_MIA(score=audit_scores, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__)
        target_path = path_utils.target_checkpoint_path_from_loss_signals_pickle_path(target_loss_path)
        result_metrics_dir = path_utils.metrics_dir_from_target(config.res_dir, target_path, attack, config.audit_mode)
        result_metrics_dir.mkdir(parents=True, exist_ok=True)
        filename = path_utils.metrics_pickle_name_from_target(target_path, attack, config.audit_mode)
        with open(result_metrics_dir / filename, "wb") as f:
            pickle.dump(metrics, f)
    print_average_metrics_table(attack, all_metrics)

def run_entity_audit(config):
    '''
    Run an entity-level membership inference audit.
    Args:
        config (Config): Audit configuration.
    Returns:
        None
    '''
    attack_config = attack_config_from_config(config)
    sample_attack = getattr(attack_config, "name", attack_config.attack)
    target_loss_paths, shadow_path_groups = resolve_target_shadow_paths(config)
    target_properties = path_utils.target_properties_from_loss_signals_pickle_path(target_loss_paths[0])
    hold_out_frac = target_properties["per_entity_hold_out"]
    metadata = load_dataset_metadata(config.dataset)
    entity_ids = torch.tensor(metadata["entity_ids"], dtype=torch.long)
    n_entities = int(metadata["n_entities"])
    n_population = int(metadata["n_samples"])
    entity_index_table = entity_index_table_from_entity_ids(metadata["entity_ids"])
    min_samples_per_entity = getattr(config, "min_samples_per_entity", 0)
    max_samples_per_entity = getattr(config, "max_samples_per_entity", None)
    if hold_out_frac > 0.0:
        all_loss_paths = target_loss_paths + [path for shadow_paths in shadow_path_groups for path in shadow_paths]
        unique_loss_paths = list(dict.fromkeys(all_loss_paths))
        validate_entity_hold_out_indices(
            entity_index_table=entity_index_table,
            loss_paths=unique_loss_paths,
            target_len=n_population,
            hold_out_frac=hold_out_frac,
        )
    all_metrics = []
    for target_loss_path, shadow_loss_paths in tqdm(
        zip(target_loss_paths, shadow_path_groups),
        total=len(target_loss_paths),
        desc="Running entity-level audit",
    ):
        target_loss_sigs, target_train_mask = load_loss_signals(target_loss_path)
        if len(target_loss_sigs) != n_population:
            raise ValueError(f"Unexpected loss-signal length in {target_loss_path}: got {len(target_loss_sigs)}, expected {n_population}.")
        shadow_loss_sigs, shadow_train_mask = load_shadow_signals(shadow_loss_paths, len(target_loss_sigs))
        target_train_index = utils.mask_to_index(target_train_mask)
        train_entity_ids = torch.unique(entity_ids[target_train_index])
        shadow_entity_mask = torch.stack([
            torch.bincount(
                entity_ids[sample_mask],
                minlength=n_entities,
            ) > 0
            for sample_mask in shadow_train_mask
        ])
        audit_table = select_entity_audit_indices(
            entity_index_table=entity_index_table,
            train_mask=target_train_mask,
            mode=config.mode,
            min_samples_per_entity=min_samples_per_entity,
            max_samples_per_entity=max_samples_per_entity,
            hold_out_frac=hold_out_frac,
        )
        audit_entity_ids = torch.tensor(sorted(audit_table.keys()), dtype=torch.long)
        ground_truth = torch.isin(audit_entity_ids, train_entity_ids).to(dtype=torch.long)

        match attack_config.attack:
            case "CompositeBASE":
                score = attacks.CompositeBASE(
                    attack_config=attack_config,
                    shadow_loss_sigs=shadow_loss_sigs,
                    shadow_train_mask=shadow_train_mask,
                ).run_attack(audit_table=audit_table, target_loss_sigs=target_loss_sigs)
            case "CompositeLiRA":
                score = attacks.CompositeLiRA(
                    audit_table=audit_table,
                    shadow_loss_sigs=shadow_loss_sigs,
                    shadow_entity_mask=shadow_entity_mask,
                    offline=attack_config.offline,
                ).run_attack(target_loss_sigs)
            case "JointXGB":
                assert config.mode == "exclude_train", "Other modes are not yet supported"
                assert 0 < min_samples_per_entity == max_samples_per_entity, "Must specifiy a fixed number of audit samples per entity"
                score = attacks.JointXGB(
                    attack_config=attack_config,
                    entity_index_table=entity_index_table,
                    shadow_loss_sigs=shadow_loss_sigs,
                    shadow_train_mask=shadow_train_mask,
                    shadow_entity_mask=shadow_entity_mask,
                    n_features=min_samples_per_entity,
                ).run_attack(audit_table=audit_table, target_loss_sigs=target_loss_sigs)
            case _:
                raise ValueError(f"Unsupported entity-level attack: {attack_config.attack}")
        score = torch.stack([score[entity_id] for entity_id in sorted(audit_table.keys())]).to(dtype=torch.float32)
        assert len(score) == len(ground_truth)
        metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__)
        target_path = path_utils.target_checkpoint_path_from_loss_signals_pickle_path(target_loss_path)
        result_metrics_dir = path_utils.metrics_dir_from_target(
            config.res_dir,
            target_path,
            sample_attack,
            config.audit_mode,
            entity_audit_mode=config.mode,
            min_samples_per_entity=min_samples_per_entity,
            max_samples_per_entity=max_samples_per_entity,
        )
        result_metrics_dir.mkdir(parents=True, exist_ok=True)
        filename = path_utils.metrics_pickle_name_from_target(
            target_path,
            sample_attack,
            config.audit_mode,
            min_samples_per_entity=min_samples_per_entity,
            max_samples_per_entity=max_samples_per_entity,
        )
        with open(result_metrics_dir / filename, "wb") as f:
            pickle.dump(metrics, f)
    print_average_metrics_table(attack_config.name, all_metrics)

def parse_args(argv=None):
    '''
    Parse CLI arguments for audit.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Run membership inference audit.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to audit config yaml file.",
    )
    parser.add_argument("--target-loss-paths", nargs="+", default=None, help="Target loss-signal file/folder paths. Overrides config.")
    parser.add_argument("--shadow-loss-paths", nargs="+", default=None, help="Shadow loss-signal file/folder paths. Overrides config.")
    return parser.parse_args(argv)

def main(argv=None):
    '''
    Entry point for the audit CLI.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        None
    '''
    args = parse_args(argv)
    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)
    config = utils.Config(config_dict)
    seed = getattr(config, "seed", 0)
    utils.set_manual_seed(seed)
    if args.target_loss_paths is not None:
        config.target_loss_paths = args.target_loss_paths
    if args.shadow_loss_paths is not None:
        config.shadow_loss_paths = args.shadow_loss_paths
    if config.audit_mode == "sample":
        run_sample_audit(config=config)
    elif config.audit_mode == "entity":
        run_entity_audit(config=config)
    else:
        raise ValueError(f"Unknown audit_mode: {config.audit_mode}")

if __name__ == "__main__":
    main()
