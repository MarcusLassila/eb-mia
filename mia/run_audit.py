from data.data import EntityDataset
from data.utils import load_dataset
from . import attacks
from . import evaluation
import utils

import argparse
import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from pathlib import Path
import pickle
import yaml
import numpy as np

def load_partition_fn(model_path):
    pathdir = model_path.parent / Path("partition-functions")
    pathname = model_path.name
    with open(pathdir / pathname, "rb") as f:
        log_Z = pickle.load(f)
    return log_Z

def indices_of_shadow_models(index_target, n_models):
    '''
    Return the indices of the shadow models under round robin auditing.
    Due to the way models are trained on pairs of disjoint and complementing data splits,
    the "complement" model to the target model should be filtered out
    since a realistic adversary cannot obtain such a complement model.
    '''
    assert 0 <= index_target < n_models
    if index_target % 2 == 0:
        excluded_indices = {index_target, index_target + 1}
    else:
        excluded_indices = {index_target - 1, index_target}
    index_shadow_models = sorted(set(range(n_models)) - excluded_indices)
    return index_shadow_models

def get_audit_indices(n_audit_samples, membership_mask):
    '''
    Get indices of audit samples with 50% target training member samples.
    '''
    assert n_audit_samples % 2 == 0
    member_indices = utils.mask_to_index(membership_mask)
    non_member_indices = utils.mask_to_index(~membership_mask)
    rand_mask = torch.randperm(member_indices.shape[0])
    selected_members = member_indices[rand_mask][:n_audit_samples // 2]
    rand_mask = torch.randperm(non_member_indices.shape[0])
    selected_non_members = non_member_indices[rand_mask][:n_audit_samples // 2]
    audit_indices = torch.cat((selected_members, selected_non_members)).sort()[0]
    return audit_indices

def get_entity_audit_table(data_population: EntityDataset, target_train_index, mode, min_samples_per_entity=None, max_samples_per_entity=None):
    '''Build an entity -> Subset table with optional size filtering. Args: data_population (EntityDataset), target_train_index (torch.Tensor), mode (str), min_samples_per_entity (int|None), max_samples_per_entity (int|None). Returns: dict[int, Subset].'''
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
            case "max_one_train_sample":
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

    audit_samples = {
        entity_id: Subset(data_population, selected_index_table[entity_id])
        for entity_id in entity_ids
    }
    return audit_samples

def get_attacker(attack_config, batch_size, device, shadow_model_paths):
    match attack_config.attack:
        case "BASE":
            attacker = attacks.BASE(
                batch_size=batch_size,
                device=device,
                shadow_model_paths=shadow_model_paths,
                prior=attack_config.prior,
                n_loss_samples=attack_config.n_loss_samples,
            )
        case "CompositeBASE":
            attacker = attacks.CompositeBASE(
                batch_size=batch_size,
                device=device,
                shadow_model_paths=shadow_model_paths,
                prior=attack_config.prior,
                n_loss_samples=attack_config.n_loss_samples,
            )
        case _:
            raise ValueError(f"No attack: {attack_config.attack}")
    return attacker

def metrics_pickle_name(target_path, attack, audit_mode, min_samples_per_entity=None, max_samples_per_entity=None):
    '''Return normalized metrics pickle filename. Args: target_path (str|Path), attack (str), audit_mode (str), min_samples_per_entity (int|None), max_samples_per_entity (int|None). Returns: str.'''
    parts = [
        "metrics",
        f"attack-{attack}",
        f"target-{Path(target_path).stem}",
        f"mode-{audit_mode}",
    ]
    if audit_mode == "entity":
        min_value = "none" if min_samples_per_entity is None else str(min_samples_per_entity)
        max_value = "none" if max_samples_per_entity is None else str(max_samples_per_entity)
        parts.append(f"min-{min_value}")
        parts.append(f"max-{max_value}")
    return "_".join(parts) + ".pkl"

def print_average_metrics_table(attack, metrics_list):
    '''Print mean audit metrics over target models. Args: attack (str), metrics_list (list[dict]). Returns: None.'''
    mean_auc = float(np.mean([metrics["AUC"] for metrics in metrics_list]))
    mean_tpr_1pct = float(np.mean([metrics["TPR@1%FPR"] for metrics in metrics_list]))
    mean_tpr_0p1pct = float(np.mean([metrics["TPR@0.1%FPR"] for metrics in metrics_list]))
    print("")
    print(f"Audit summary ({attack})")
    print(f"{'Metric':<16} {'Mean':>10}")
    print(f"{'-' * 16} {'-' * 10}")
    print(f"{'AUC':<16} {mean_auc:>10.4f}")
    print(f"{'TPR@1%FPR':<16} {mean_tpr_1pct:>10.4f}")
    print(f"{'TPR@0.1%FPR':<16} {mean_tpr_0p1pct:>10.4f}")

def run_audit(config, device, audit_config=None):
    image_size = utils.parse_properties_from_checkpoint_path(config.target_model_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    target_model_paths = list(map(Path, config.target_model_paths))
    attack_config = utils.Config(config.attack)
    attack = attack_config.attack
    all_metrics = []
    if not config.round_robin:
        shadow_model_paths = list(map(Path, config.shadow_model_paths))
    for target_idx, target_path in tqdm(enumerate(target_model_paths), total=len(target_model_paths), desc="Running audit"):
        if config.round_robin:
            shadow_model_indices = indices_of_shadow_models(target_idx, len(target_model_paths))
            shadow_model_paths = [model_path for i, model_path in enumerate(target_model_paths) if i in shadow_model_indices]
        target_train_indices = utils.get_train_indices(target_path)
        membership_mask = utils.index_to_mask(target_train_indices, len(data_population))
        audit_indices = get_audit_indices(config.n_audit_samples, membership_mask)
        audit_samples = Subset(data_population, audit_indices)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
        attacker = get_attacker(
            attack_config=attack_config,
            batch_size=config.batch_size,
            device=device,
            shadow_model_paths=shadow_model_paths,
        )
        score = attacker.run_attack(audit_samples, target_path)
        metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__) if audit_config is None else dict(audit_config)
        Path(f"{config.res_dir}/{attack}").mkdir(parents=True, exist_ok=True)
        filename = metrics_pickle_name(target_path, attack, config.audit_mode)
        with open(f"{config.res_dir}/{attack}/{filename}", "wb") as f:
            pickle.dump(metrics, f)
    print_average_metrics_table(attack, all_metrics)

def run_entity_audit(config, device, audit_config=None):
    image_size = utils.parse_properties_from_checkpoint_path(config.target_model_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    assert isinstance(data_population, EntityDataset)
 
    target_model_paths = list(map(Path, config.target_model_paths))
    attack_config = utils.Config(config.attack)
    attack = attack_config.attack
    all_metrics = []
    if not config.round_robin:
        shadow_model_paths = list(map(Path, config.shadow_model_paths))
    for target_idx, target_path in enumerate(target_model_paths):
        print(f"Running audit number {target_idx}")
        if config.round_robin:
            shadow_model_indices = indices_of_shadow_models(target_idx, len(target_model_paths))
            shadow_model_paths = [model_path for i, model_path in enumerate(target_model_paths) if i in shadow_model_indices]

        target_train_index = utils.get_train_indices(target_path)
        train_entity_ids = torch.unique(data_population.entity_ids[target_train_index])
        audit_table = get_entity_audit_table(
            data_population,
            target_train_index,
            mode=config.entity_audit_mode,
            min_samples_per_entity=getattr(config, "entity_audit_min_samples_per_entity", None),
            max_samples_per_entity=getattr(config, "entity_audit_max_samples_per_entity", None),
        )

        ground_truth = {entity_id: 0 for entity_id in audit_table.keys()}
        for entity_id in train_entity_ids:
            entity_id = entity_id.item()
            if entity_id in ground_truth:
                ground_truth[entity_id] = 1
        ground_truth = torch.tensor(list(ground_truth.values()), dtype=torch.long)

        attacker = get_attacker(
            attack_config=attack_config,
            batch_size=config.batch_size,
            device=device,
            shadow_model_paths=shadow_model_paths,
        )
        score = attacker.run_attack(audit_table, target_path)
        score = torch.tensor(list(score.values()), dtype=torch.float32)
        metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__) if audit_config is None else dict(audit_config)
        Path(f"{config.res_dir}/{attack}").mkdir(parents=True, exist_ok=True)
        filename = metrics_pickle_name(
            target_path,
            attack,
            config.audit_mode,
            min_samples_per_entity=getattr(config, "entity_audit_min_samples_per_entity", None),
            max_samples_per_entity=getattr(config, "entity_audit_max_samples_per_entity", None),
        )
        with open(f"{config.res_dir}/{attack}/{filename}", "wb") as f:
            pickle.dump(metrics, f)
    print_average_metrics_table(attack, all_metrics)

def default_config_path():
    '''Return default audit config path. Args: None. Returns: str.'''
    root = utils.get_root()
    if root is None:
        return "mia/configs/config_audit.yaml"
    return f"{root}/mia/configs/config_audit.yaml"

def parse_args(argv=None):
    '''Parse CLI arguments for audit. Args: argv (list[str]|None). Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Run membership inference audit.")
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="Path to audit config yaml file.",
    )
    return parser.parse_args(argv)

def main(argv=None):
    '''Entry point for audit CLI. Args: argv (list[str]|None). Returns: None.'''
    args = parse_args(argv)
    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)
    config = utils.Config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.audit_mode == "sample":
        run_audit(config=config, device=device, audit_config=config_dict)
    elif config.audit_mode == "entity":
        run_entity_audit(config=config, device=device, audit_config=config_dict)
    else:
        raise ValueError(f"Unknown audit_mode: {config.audit_mode}")

if __name__ == "__main__":
    main()
