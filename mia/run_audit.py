from data.data import EntityDataset
from data.utils import load_dataset
from . import attacks
from . import evaluation
from . import path_utils
import utils

import argparse
import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

import pickle
import yaml
import numpy as np

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
    '''Build an entity -> Subset table with optional size filtering. Args: data_population (EntityDataset), target_train_index (torch.Tensor), mode (str), min_samples_per_entity (int|None), max_samples_per_entity (int|None), n_audit_samples_per_entity (int|None). Returns: dict[int, Subset].'''
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

    audit_samples = {
        entity_id: Subset(data_population, selected_index_table[entity_id])
        for entity_id in entity_ids
    }
    return audit_samples

def load_scores(scores_path):
    '''Load sample-level MIA scores and train mask from a score pickle. Args: scores_path (str|Path). Returns: tuple[torch.Tensor, torch.Tensor].'''
    with open(scores_path, "rb") as file:
        scores_payload = pickle.load(file)
    if not isinstance(scores_payload, dict):
        raise ValueError("Scores pickle must contain keys 'scores' and 'train_mask'.")
    if "scores" not in scores_payload or "train_mask" not in scores_payload:
        raise ValueError("Scores pickle must contain keys 'scores' and 'train_mask'.")
    scores = torch.tensor(scores_payload["scores"], dtype=torch.float32)
    train_mask = torch.tensor(scores_payload["train_mask"], dtype=torch.bool)
    if len(scores) != len(train_mask):
        raise ValueError(f"Score length and train mask length mismatch in {scores_path}.")
    return scores, train_mask

def print_average_metrics_table(attack, metrics_list):
    '''Print mean audit metrics over target models. Args: attack (str), metrics_list (list[dict]). Returns: None.'''
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
    score_paths = path_utils.resolve_audit_score_paths(config)
    attack = path_utils.infer_attack_from_score_paths(score_paths)
    image_size = path_utils.target_properties_from_scores_pickle_path(score_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    all_metrics = []
    for scores_path in tqdm(score_paths, total=len(score_paths), desc="Running sample-level audit"):
        scores, membership_mask = load_scores(scores_path)
        if len(scores) != len(data_population):
            raise ValueError(f"Unexpected score length in {scores_path}: got {len(scores)}, expected {len(data_population)}.")
        audit_indices = get_audit_indices(getattr(config, "n_audit_samples", len(data_population)), membership_mask)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
        audit_scores = scores[audit_indices]
        metrics = evaluation.evaluate_MIA(score=audit_scores, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__)
        result_metrics_dir = path_utils.metrics_dir(config.res_dir, scores_path, config.audit_mode)
        result_metrics_dir.mkdir(parents=True, exist_ok=True)
        filename = path_utils.metrics_pickle_name(scores_path, config.audit_mode)
        with open(result_metrics_dir / filename, "wb") as f:
            pickle.dump(metrics, f)
    print_average_metrics_table(attack, all_metrics)

def run_entity_audit(config):
    score_paths = path_utils.resolve_audit_score_paths(config)
    sample_attack, is_offline_attack = path_utils.parse_attack_name(path_utils.infer_attack_from_score_paths(score_paths))
    n_audit_samples_per_entity = getattr(config, "n_audit_samples_per_entity", None)
    if config.mode == "all" and n_audit_samples_per_entity is not None:
        raise ValueError("mode='all' is incompatible with n_audit_samples_per_entity.")
    image_size = path_utils.target_properties_from_scores_pickle_path(score_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    assert isinstance(data_population, EntityDataset)

    composite_attack = f"Composite{sample_attack}"
    attacker_cls = getattr(attacks, composite_attack, None)
    if attacker_cls is None:
        raise ValueError(f"No composite MIA: {composite_attack}")
    attacker = attacker_cls()
    attack = composite_attack + ("-off" if is_offline_attack else "")
    all_metrics = []
    for scores_path in tqdm(score_paths, total=len(score_paths), desc="Running entity-level audit"):
        sample_scores, train_mask = load_scores(scores_path)
        if len(sample_scores) != len(data_population):
            raise ValueError(f"Unexpected score length in {scores_path}: got {len(sample_scores)}, expected {len(data_population)}.")
        target_train_index = utils.mask_to_index(train_mask)
        train_entity_ids = torch.unique(data_population.entity_ids[target_train_index])
        audit_table = get_entity_audit_table(
            data_population,
            target_train_index,
            mode=config.mode,
            min_samples_per_entity=getattr(config, "min_samples_per_entity", None),
            max_samples_per_entity=getattr(config, "max_samples_per_entity", None),
            n_audit_samples_per_entity=n_audit_samples_per_entity,
        )

        ground_truth = {entity_id: 0 for entity_id in audit_table.keys()}
        for entity_id in train_entity_ids:
            entity_id = entity_id.item()
            if entity_id in ground_truth:
                ground_truth[entity_id] = 1
        ground_truth = torch.tensor([ground_truth[entity_id] for entity_id in sorted(ground_truth.keys())], dtype=torch.long)

        entity_sample_scores = {
            entity_id: sample_scores[audit_samples.indices]
            for entity_id, audit_samples in audit_table.items()
        }

        score = attacker.run_attack(entity_sample_scores)
        score = torch.stack([score[entity_id] for entity_id in sorted(audit_table.keys())]).to(dtype=torch.float32)
        assert len(score) == len(ground_truth)
        metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        all_metrics.append(metrics)
        metrics["audit_config"] = dict(config.__dict__)
        result_metrics_dir = path_utils.metrics_dir(
            config.res_dir,
            scores_path,
            config.audit_mode,
            entity_audit_mode=config.mode,
            n_audit_samples_per_entity=getattr(config, "n_audit_samples_per_entity", None),
        )
        result_metrics_dir.mkdir(parents=True, exist_ok=True)
        filename = path_utils.metrics_pickle_name(
            scores_path,
            config.audit_mode,
            min_samples_per_entity=getattr(config, "min_samples_per_entity", None),
            max_samples_per_entity=getattr(config, "max_samples_per_entity", None),
            n_audit_samples_per_entity=n_audit_samples_per_entity,
        )
        with open(result_metrics_dir / filename, "wb") as f:
            pickle.dump(metrics, f)
    print_average_metrics_table(attack, all_metrics)

def parse_args(argv=None):
    '''Parse CLI arguments for audit. Args: argv (list[str]|None). Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Run membership inference audit.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to audit config yaml file.",
    )
    parser.add_argument(
        "--score-paths",
        nargs="+",
        default=None,
        help="Score pickle file/folder paths. Overrides config.",
    )
    return parser.parse_args(argv)

def main(argv=None):
    '''Entry point for audit CLI. Args: argv (list[str]|None). Returns: None.'''
    args = parse_args(argv)
    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)
    config = utils.Config(config_dict)
    if args.score_paths is not None:
        config.score_paths = args.score_paths
    if config.audit_mode == "sample":
        run_sample_audit(config=config)
    elif config.audit_mode == "entity":
        run_entity_audit(config=config)
    else:
        raise ValueError(f"Unknown audit_mode: {config.audit_mode}")

if __name__ == "__main__":
    main()
