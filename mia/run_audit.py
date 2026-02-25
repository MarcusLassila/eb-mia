from data.data import EntityDataset
from data.utils import load_dataset
from . import attacks
from . import evaluation
from .run_mia import scores_pickle_name
import utils

import argparse
import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from pathlib import Path
import pickle
import yaml
import numpy as np

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

def get_attacker(attack_config):
    match attack_config.attack:
        case "CompositeBASE":
            attacker = attacks.CompositeBASE(prior=attack_config.prior)
        case _:
            raise ValueError(f"No composite MIA: {attack_config.attack}")
    return attacker

def metrics_pickle_name(target_path, attack, audit_mode, min_samples_per_entity=None, max_samples_per_entity=None):
    '''Return normalized metrics pickle filename. Args: target_path (str|Path), attack (str), audit_mode (str), min_samples_per_entity (int|None), max_samples_per_entity (int|None). Returns: str.'''
    return scores_pickle_name(
        target_path,
        attack,
        audit_mode,
        min_samples_per_entity=min_samples_per_entity,
        max_samples_per_entity=max_samples_per_entity
    ).replace("scores", "metrics", 1)

def load_scores(res_dir, attack, target_path):
    '''Load sample-level MIA scores for a target model. Args: res_dir (str|Path), attack (str), target_path (str|Path). Returns: torch.Tensor.'''
    pathname = Path(res_dir) / attack / scores_pickle_name(target_path, attack, "sample")
    with open(pathname, "rb") as file:
        scores = pickle.load(file)
    return torch.tensor(scores, dtype=torch.float32)

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

def run_sample_audit(config):
    image_size = utils.parse_properties_from_checkpoint_path(config.target_model_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    target_model_paths = list(map(Path, config.target_model_paths))
    attack_config = utils.Config(config.attack)
    attack = attack_config.attack
    all_metrics = []
    for _, target_path in tqdm(enumerate(target_model_paths), total=len(target_model_paths), desc="Running audit"):
        target_train_indices = utils.get_train_indices(target_path)
        membership_mask = utils.index_to_mask(target_train_indices, len(data_population))
        audit_indices = get_audit_indices(config.n_audit_samples, membership_mask)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
        all_scores = load_scores(config.res_dir, attack, target_path)
        score = all_scores[audit_indices]
        metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__)
        Path(f"{config.res_dir}/{attack}").mkdir(parents=True, exist_ok=True)
        filename = metrics_pickle_name(target_path, attack, config.audit_mode)
        with open(f"{config.res_dir}/{attack}/{filename}", "wb") as f:
            pickle.dump(metrics, f)
    print_average_metrics_table(attack, all_metrics)

def run_entity_audit(config):
    image_size = utils.parse_properties_from_checkpoint_path(config.target_model_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    assert isinstance(data_population, EntityDataset)
 
    target_model_paths = list(map(Path, config.target_model_paths))
    attack_config = utils.Config(config.attack)
    attack = attack_config.attack
    attacker = get_attacker(attack_config)
    all_metrics = []
    for target_idx, target_path in enumerate(target_model_paths):
        print(f"Running audit number {target_idx}")
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

        sample_scores = load_scores(config.res_dir, "BASE", target_path)
        entity_sample_scores = {
            entity_id: sample_scores[audit_samples.indices]
            for entity_id, audit_samples in audit_table.items()
        }

        score = attacker.run_attack(entity_sample_scores)
        score = torch.stack([score[entity_id] for entity_id in audit_table.keys()]).to(dtype=torch.float32)
        metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__)
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

def parse_args(argv=None):
    '''Parse CLI arguments for audit. Args: argv (list[str]|None). Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Run membership inference audit.")
    default_config_path = str(utils.resolve_path(Path("mia") / "configs" / "config_audit.yaml", utils.get_root()))
    parser.add_argument(
        "--config",
        default=default_config_path,
        help="Path to audit config yaml file.",
    )
    return parser.parse_args(argv)

def main(argv=None):
    '''Entry point for audit CLI. Args: argv (list[str]|None). Returns: None.'''
    args = parse_args(argv)
    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)
    config = utils.Config(config_dict)
    if config.audit_mode == "sample":
        run_sample_audit(config=config)
    elif config.audit_mode == "entity":
        run_entity_audit(config=config)
    else:
        raise ValueError(f"Unknown audit_mode: {config.audit_mode}")

if __name__ == "__main__":
    main()
