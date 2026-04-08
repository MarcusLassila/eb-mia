from data.datasets import EntityDataset
from data.utils import load_dataset
from . import attacks
from . import evaluation
from . import path_utils
from .utils import indices_of_shadow_models, load_loss_signals
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

def get_audit_indices(n_audit_samples, membership_mask):
    assert n_audit_samples <= membership_mask.shape[0]
    member_indices = utils.mask_to_index(membership_mask)
    non_member_indices = utils.mask_to_index(~membership_mask)
    if 2 * member_indices.shape[0] < n_audit_samples:
        n_in_samples = member_indices.shape[0]
        n_out_samples = n_audit_samples - n_in_samples
    elif 2 * non_member_indices.shape[0] < n_audit_samples:
        n_out_samples = non_member_indices.shape[0]
        n_in_samples = n_audit_samples - n_out_samples
    else:
        n_in_samples = n_audit_samples // 2
        n_out_samples = n_audit_samples // 2 + n_audit_samples % 2
    rand_mask = torch.randperm(member_indices.shape[0])
    selected_members = member_indices[rand_mask][:n_in_samples]
    rand_mask = torch.randperm(non_member_indices.shape[0])
    selected_non_members = non_member_indices[rand_mask][:n_out_samples]
    audit_indices = torch.cat((selected_members, selected_non_members)).sort()[0]
    return audit_indices

def get_entity_audit_table(data_population: EntityDataset, target_train_index, mode, min_samples_per_entity=None, max_samples_per_entity=None, n_audit_samples_per_entity=None):
    '''
    Build an entity-to-indices table with optional size filtering.
    Args:
        data_population (EntityDataset): Full population dataset.
        target_train_index (torch.Tensor): Target training indices.
        mode (str): Entity audit mode.
        min_samples_per_entity (int | None): Optional minimum entity size.
        max_samples_per_entity (int | None): Optional maximum entity size.
        n_audit_samples_per_entity (int | None): Optional fixed per-entity sample count.
    Returns:
        dict[int, list[int]]: Selected indices for each audited entity.
    '''
    table = data_population.get_entity_index_table()
    selected_index_table = {}
    target_entities = set()
    target_train_index = set(target_train_index.tolist())
    for entity_id, indices in table.items():
        indices = set(indices)
        if indices & target_train_index:
            target_entities.add(entity_id)
        match mode:
            case "all":
                selected_indices = indices
            case "max_one_train":
                overlap = sorted(indices & target_train_index)
                if len(overlap) > 1:
                    keep = overlap[0]
                    indices = indices - set(overlap)
                    indices.add(keep)
                selected_indices = indices
                assert len(selected_indices & target_train_index) <= 1
            case "exclude_train":
                selected_indices = indices - target_train_index
            case _:
                raise ValueError(f"Unknown entity audit mode: {mode}")
        n_selected = len(selected_indices)
        if n_selected == 0:
            continue
        if n_audit_samples_per_entity is not None:
            if n_selected < n_audit_samples_per_entity:
                continue
            if n_selected > n_audit_samples_per_entity:
                selected_indices_sorted = sorted(selected_indices)
                if mode == "max_one_train":
                    selected_target_indices = sorted(selected_indices & target_train_index)
                    if selected_target_indices:
                        keep = selected_target_indices[0]
                        selected_indices_sorted = [keep] + [idx for idx in selected_indices_sorted if idx != keep][:n_audit_samples_per_entity - 1]
                    else:
                        selected_indices_sorted = selected_indices_sorted[:n_audit_samples_per_entity]
                else:
                    selected_indices_sorted = selected_indices_sorted[:n_audit_samples_per_entity]
                selected_indices = set(selected_indices_sorted)
                n_selected = len(selected_indices)
        if min_samples_per_entity is not None and n_selected < min_samples_per_entity:
            continue
        if max_samples_per_entity is not None and n_selected > max_samples_per_entity:
            continue
        selected_index_table[entity_id] = sorted(selected_indices)

    entity_ids = sorted(selected_index_table.keys())
    target_entity_ids = [entity_id for entity_id in entity_ids if entity_id in target_entities]
    non_target_entity_ids = [entity_id for entity_id in entity_ids if entity_id not in target_entities]
    if target_entity_ids and non_target_entity_ids:
        n_keep_per_group = min(len(target_entity_ids), len(non_target_entity_ids))
        keep_entity_ids = set(target_entity_ids[:n_keep_per_group] + non_target_entity_ids[:n_keep_per_group])
        entity_ids = [entity_id for entity_id in entity_ids if entity_id in keep_entity_ids]
    else:
        raise RuntimeError("Failed to include both target and non-target entities in the audit table.")

    return {entity_id: selected_index_table[entity_id] for entity_id in entity_ids}

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
        shadow_path_groups = []
        for target_idx in range(len(target_loss_paths)):
            shadow_indices = indices_of_shadow_models(target_idx, len(target_loss_paths))
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
    image_size = path_utils.target_properties_from_loss_signals_pickle_path(target_loss_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    all_metrics = []
    for target_loss_path, shadow_loss_paths in tqdm(
        zip(target_loss_paths, shadow_path_groups),
        total=len(target_loss_paths),
        desc="Running sample-level audit",
    ):
        target_loss_sig, membership_mask = load_loss_signals(target_loss_path)
        if len(target_loss_sig) != len(data_population):
            raise ValueError(f"Unexpected loss-signal length in {target_loss_path}: got {len(target_loss_sig)}, expected {len(data_population)}.")
        shadow_loss_sigs, shadow_train_mask = load_shadow_signals(shadow_loss_paths, len(target_loss_sig))
        attacker = get_attacker(attack_config, shadow_loss_sigs, shadow_train_mask)
        scores = attacker.run_attack(target_loss_sig)
        audit_indices = get_audit_indices(getattr(config, "n_audit_samples", len(data_population)), membership_mask)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
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
    attack_config = attack_config_from_config(config)
    sample_attack = getattr(attack_config, "name", attack_config.attack)
    target_loss_paths, shadow_path_groups = resolve_target_shadow_paths(config)
    n_audit_samples_per_entity = getattr(config, "n_audit_samples_per_entity", None)
    if config.mode == "all" and n_audit_samples_per_entity is not None:
        raise ValueError("mode='all' is incompatible with n_audit_samples_per_entity.")
    image_size = path_utils.target_properties_from_loss_signals_pickle_path(target_loss_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    entity_index_table = data_population.get_entity_index_table()
    assert isinstance(data_population, EntityDataset)
    all_metrics = []
    for target_loss_path, shadow_loss_paths in tqdm(
        zip(target_loss_paths, shadow_path_groups),
        total=len(target_loss_paths),
        desc="Running entity-level audit",
    ):
        target_loss_sigs, train_mask = load_loss_signals(target_loss_path)
        if len(target_loss_sigs) != len(data_population):
            raise ValueError(f"Unexpected loss-signal length in {target_loss_path}: got {len(target_loss_sigs)}, expected {len(data_population)}.")
        shadow_loss_sigs, shadow_train_mask = load_shadow_signals(shadow_loss_paths, len(target_loss_sigs))
        target_train_index = utils.mask_to_index(train_mask)
        train_entity_ids = torch.unique(data_population.entity_ids[target_train_index])
        shadow_entity_mask = torch.stack([
            torch.bincount(
                data_population.entity_ids[sample_mask],
                minlength=data_population.n_entities,
            ) > 0
            for sample_mask in shadow_train_mask
        ])
        audit_table = get_entity_audit_table(
            data_population,
            target_train_index,
            mode=config.mode,
            min_samples_per_entity=getattr(config, "min_samples_per_entity", None),
            max_samples_per_entity=getattr(config, "max_samples_per_entity", None),
            n_audit_samples_per_entity=n_audit_samples_per_entity,
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
                assert hasattr(config, "n_audit_samples_per_entity"), "Must specify a fixed number of audit samples per entity"
                score = attacks.JointXGB(
                    attack_config=attack_config,
                    entity_index_table=entity_index_table,
                    shadow_loss_sigs=shadow_loss_sigs,
                    shadow_train_mask=shadow_train_mask,
                    shadow_entity_mask=shadow_entity_mask,
                    n_features=config.n_audit_samples_per_entity,
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
            n_audit_samples_per_entity=getattr(config, "n_audit_samples_per_entity", None),
        )
        result_metrics_dir.mkdir(parents=True, exist_ok=True)
        filename = path_utils.metrics_pickle_name_from_target(
            target_path,
            sample_attack,
            config.audit_mode,
            min_samples_per_entity=getattr(config, "min_samples_per_entity", None),
            max_samples_per_entity=getattr(config, "max_samples_per_entity", None),
            n_audit_samples_per_entity=n_audit_samples_per_entity,
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
