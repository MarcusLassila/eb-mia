from data.dataset_metadata import entity_index_table_from_entity_ids, load_dataset_metadata
from . import attacks_entity
from . import attacks_sample
from . import evaluation
from . import path_utils
from . import result_store
from .utils import (
    entitiy_train_mask,
    indices_of_shadow_models,
    select_sample_audit_indices,
    select_entity_audit_indices,
    load_loss_signals,
    subsample_loss_signals,
    standardize_signals,
)
import utils

import argparse
import torch
from tqdm.auto import tqdm

import yaml

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

def get_attacker(
    attack_config,
    shadow_loss_sigs,
    shadow_train_mask=None,
    audit_table=None,
    shadow_entity_mask=None,
):
    '''
    Instantiate a configured sample- or entity-level attack.
    Args:
        attack_config (Config): Attack configuration.
        shadow_loss_sigs (torch.Tensor): Reference-model loss signals.
        shadow_train_mask (torch.Tensor | None): Reference sample memberships.
        audit_table (dict | None): Audited entity indices.
        shadow_entity_mask (torch.Tensor | None): Reference entity memberships.
    Returns:
        object: Instantiated attacker.
    '''
    attacker = None
    match attack_config.attack:
        case "GlobalThreshold":
            attacker = attacks_sample.GlobalThreshold(
                transformation=getattr(attack_config, "transformation", "neg_mean"),
            )
        case "BASE":
            attacker = attacks_sample.BASE(
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                prior=getattr(attack_config, "prior", 0.5),
                apply_sigmoid=getattr(attack_config, "apply_sigmoid", False),
            )
        case "NormalBASE":
            attacker = attacks_sample.NormalBASE(
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                prior=getattr(attack_config, "prior", 0.5),
                use_global_var=getattr(attack_config, "use_global_dispersion", True),
                apply_sigmoid=getattr(attack_config, "apply_sigmoid", False),
            )
        case "LiRA_alt":
            attacker = attacks_sample.LiRA_alt(
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                use_global_var=getattr(attack_config, "use_global_dispersion", False),
                loss_transformation=getattr(attack_config, "loss_transformation", "none"),
            )
        case "LiRA_legacy":
            attacker = attacks_sample.LiRA_legacy(
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                use_global_var=getattr(attack_config, "use_global_dispersion", None),
                loss_transformation=getattr(attack_config, "loss_transformation", "none"),
            )
        case "LiRA":
            attacker = attacks_sample.LiRA(
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                use_global_var=attack_config.use_global_dispersion,
                share_variance=attack_config.share_variance,
                loss_transformation=getattr(attack_config, "loss_transformation", "none"),
            )
        case "HG_LiRA":
            attacker = attacks_sample.HG_LiRA(
                ref_sigs=shadow_loss_sigs,
                ref_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                sig_transformation=getattr(attack_config, "loss_transformation", "none"),
                n_gibbs_samples=getattr(attack_config, "n_gibbs_samples", 128),
                n_gibbs_warmup=getattr(attack_config, "n_gibbs_warmup", 64),
                n_quadrature=getattr(attack_config, "n_quadrature", 50),
            )
        case "HG_LiRA_r":
            attacker = attacks_sample.HG_LiRA_r(
                ref_sigs=shadow_loss_sigs,
                ref_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                sig_transformation=getattr(attack_config, "loss_transformation", "none"),
                n_gibbs_samples=getattr(attack_config, "n_gibbs_samples", 128),
                n_gibbs_warmup=getattr(attack_config, "n_gibbs_warmup", 64),
                n_quadrature=getattr(attack_config, "n_quadrature", 50),
            )
        case "HG_LiRA_r_shared":
            attacker = attacks_sample.HG_LiRA_r_shared(
                ref_sigs=shadow_loss_sigs,
                ref_train_mask=shadow_train_mask,
                offline=getattr(attack_config, "offline", False),
                sig_transformation=getattr(attack_config, "loss_transformation", "none"),
                n_gibbs_samples=getattr(attack_config, "n_gibbs_samples", 128),
                n_gibbs_warmup=getattr(attack_config, "n_gibbs_warmup", 64),
                n_quadrature=getattr(attack_config, "n_quadrature", 50),
                random_seed=getattr(attack_config, "random_seed", 42),
            )
        case "CompositeBASE":
            attacker = attacks_entity.CompositeBASE(
                attack_config=attack_config,
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_train_mask=shadow_train_mask,
            )
        case "CompositeLiRA_legacy":
            attacker = attacks_entity.CompositeLiRA_legacy(
                audit_table=audit_table,
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_entity_mask=shadow_entity_mask,
                offline=getattr(attack_config, "offline", False),
                use_full_cov=getattr(attack_config, "use_full_cov", False),
            )
        case "CompositeLiRA":
            if audit_table is None or shadow_entity_mask is None:
                raise ValueError("CompositeLiRA requires audit_table and shadow_entity_mask.")
            attacker = attacks_entity.CompositeLiRA(
                audit_table=audit_table,
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_entity_mask=shadow_entity_mask,
                offline=getattr(attack_config, "offline", False),
                covariance=getattr(attack_config, "covariance", "spherical"),
                use_global_dispersion=getattr(attack_config, "use_global_dispersion", True),
                share_variance=getattr(attack_config, "share_variance", False),
                covariance_rank=getattr(attack_config, "covariance_rank", 2),
                covariance_shrinkage=getattr(attack_config, "covariance_shrinkage", 0.1),
                n_qmc_samples=getattr(attack_config, "n_qmc_samples", 1024),
                random_seed=getattr(attack_config, "random_seed", 0),
                loss_transformation=getattr(attack_config, "loss_transformation", "none"),
            )
        case "HBE_Simple":
            if audit_table is None or shadow_entity_mask is None:
                raise ValueError("HBE_Simple requires audit_table and shadow_entity_mask.")
            if not getattr(attack_config, "offline", True):
                raise ValueError("HBE_Simple supports offline inference only.")
            attacker = attacks_entity.HBE_Simple(
                audit_table=audit_table,
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_entity_mask=shadow_entity_mask,
                loss_transformation=getattr(attack_config, "loss_transformation", "none"),
                n_gibbs_samples=getattr(attack_config, "n_gibbs_samples", 128),
                n_gibbs_warmup=getattr(attack_config, "n_gibbs_warmup", 64),
                random_seed=getattr(attack_config, "random_seed", 0),
                min_variance=getattr(attack_config, "min_variance", 1e-9),
            )
        case "HBE_GlobalLatent":
            if audit_table is None or shadow_entity_mask is None:
                raise ValueError("HBE_GlobalLatent requires audit_table and shadow_entity_mask.")
            if not getattr(attack_config, "offline", True):
                raise ValueError("HBE_GlobalLatent supports offline inference only.")
            attacker = attacks_entity.HBE_GlobalLatent(
                audit_table=audit_table,
                shadow_loss_sigs=shadow_loss_sigs,
                shadow_entity_mask=shadow_entity_mask,
                n_folds=getattr(attack_config, "n_folds", 5),
                quadrature_nodes=getattr(attack_config, "quadrature_nodes", 24),
                max_hyper_units=getattr(attack_config, "max_hyper_units", 4096),
                factor_steps=getattr(attack_config, "factor_steps", 50),
                factor_tolerance=getattr(attack_config, "factor_tolerance", 1e-7),
                min_variance=getattr(attack_config, "min_variance", 1e-9),
            )
        case _:
            raise ValueError(f"Unsupported attack: {attack_config.attack}")
    assert attacker is not None
    return attacker

def resolve_target_shadow_paths(config, entity_ids=None):
    '''
    Resolve per-target shadow loss-signal path groups for auditing.
    Args:
        config (Config): Audit configuration.
    Returns:
        tuple[list[Path], list[list[Path]]]: Target paths and grouped shadow paths.
    '''
    target_loss_paths = path_utils.resolve_audit_loss_signal_paths(config, "target_loss_paths")
    if getattr(config, "round_robin", False):
        shadow_loss_paths = target_loss_paths
    else:
        shadow_loss_paths = path_utils.resolve_audit_loss_signal_paths(config, "shadow_loss_paths")
    all_paths = sorted(set(target_loss_paths + shadow_loss_paths))
    train_masks = []
    for loss_path in all_paths:
        _, train_mask = load_loss_signals(loss_path)
        if train_masks and len(train_mask) != len(train_masks[0]):
            raise ValueError(f"Unexpected train-mask length in {loss_path}.")
        train_masks.append(train_mask)
    train_masks = torch.stack(train_masks)
    if config.audit_mode == "entity":
        if entity_ids is None:
            metadata = load_dataset_metadata(config.dataset)
            entity_ids = torch.tensor(metadata["entity_ids"], dtype=torch.long)
        if len(entity_ids) != train_masks.shape[1]:
            raise ValueError("Entity metadata must match the loss-signal population.")
        unique_entities, entity_ids = torch.unique(entity_ids, return_inverse=True)
        train_masks = entitiy_train_mask(entity_ids, len(unique_entities), train_masks)
    candidate_paths = set(shadow_loss_paths)
    shadow_path_groups = []
    for target_loss_path in target_loss_paths:
        target_index = all_paths.index(target_loss_path)
        shadow_indices = indices_of_shadow_models(target_index, train_masks)
        filtered_paths = [all_paths[index] for index in shadow_indices if all_paths[index] in candidate_paths]
        if not filtered_paths:
            raise ValueError(f"No shadow models remain after excluding target and complement splits for {target_loss_path}.")
        shadow_path_groups.append(filtered_paths)
    return target_loss_paths, shadow_path_groups

def load_shadow_loss_signals(shadow_loss_paths, target_len):
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

def normalize_loss_signals(target_loss_sigs, shadow_loss_sigs, normalization, min_std=1e-12):
    '''
    Normalize target and shadow loss signals before attack fitting.
    Args:
        target_loss_sigs (torch.Tensor): Target losses shaped as data points, loss samples.
        shadow_loss_sigs (torch.Tensor): Shadow losses shaped as models, data points, loss samples.
        normalization (str): Normalization mode, either none or log_standardized.
        min_std (float): Minimum standard deviation used for numerical stability.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized target and shadow loss signals.
    '''
    if normalization in (None, "none"):
        return target_loss_sigs, shadow_loss_sigs
    if normalization != "log_standardized":
        raise ValueError(f"Unsupported loss normalization: {normalization}")
    target_loss_sigs = standardize_signals(target_loss_sigs, "log", min_std)
    shadow_loss_sigs = standardize_signals(shadow_loss_sigs, "log", min_std)
    return target_loss_sigs, shadow_loss_sigs

def _validate_entity_hold_out_indices(entity_index_table, loss_paths, target_len, hold_out_frac):
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

def run_sample_audit(config):
    '''
    Run a sample-level membership inference audit.
    Args:
        config (Config): Audit configuration.
    Returns:
        None
    '''
    attack_config = attack_config_from_config(config)
    attack = getattr(attack_config, "name", attack_config.attack)
    target_loss_paths, shadow_path_groups = resolve_target_shadow_paths(config)
    n_loss_samples = getattr(config, "n_loss_samples", None)
    sampling_seed = int(getattr(config, "seed", 0))
    metrics_paths = []
    target_shadow_pairs = zip(target_loss_paths, shadow_path_groups)
    progress = tqdm(
        target_shadow_pairs,
        total=len(target_loss_paths),
        desc="Running sample-level audit",
    )
    for target_loss_path, shadow_loss_paths in progress:
        target_loss_sigs, target_train_mask = load_loss_signals(target_loss_path)
        n_population = len(target_loss_sigs)
        shadow_loss_sigs, shadow_train_mask = load_shadow_loss_signals(shadow_loss_paths, len(target_loss_sigs))
        target_seed = path_utils.audit_seed(target_loss_path, sampling_seed)
        if n_loss_samples is not None:
            generator = torch.Generator().manual_seed(target_seed)
            target_loss_sigs = subsample_loss_signals(target_loss_sigs, n_loss_samples, generator)
            shadow_loss_sigs = subsample_loss_signals(shadow_loss_sigs, n_loss_samples, generator)
        loss_normalization = getattr(config, "loss_normalization", "log_standardized")
        target_loss_sigs, shadow_loss_sigs = normalize_loss_signals(
            target_loss_sigs,
            shadow_loss_sigs,
            loss_normalization,
        )
        attacker = get_attacker(attack_config, shadow_loss_sigs, shadow_train_mask)
        scores = attacker.run_attack(target_loss_sigs)
        audit_generator = torch.Generator().manual_seed(target_seed)
        audit_indices = select_sample_audit_indices(
            getattr(config, "n_audit_samples", n_population),
            target_train_mask,
            generator=audit_generator,
        )
        ground_truth = target_train_mask.to(dtype=torch.long)[audit_indices]
        audit_scores = scores[audit_indices]
        metrics = evaluation.evaluate_MIA(score=audit_scores, ground_truth=ground_truth)
        metrics["audit_indices"] = audit_indices.tolist()
        metrics_path = result_store.save_audit_metrics(config, metrics, target_loss_path, shadow_loss_paths, attack)
        metrics_paths.append(metrics_path)
    manifest_path = result_store.write_audit_manifest(config, metrics_paths)
    if getattr(config, "print_summary", True):
        evaluation.print_grouped_metrics(
            metrics_paths[0].parent,
            show_auc=getattr(config, "show_auc", False),
            manifest_path=manifest_path,
        )

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
    metadata = load_dataset_metadata(config.dataset)
    entity_ids = torch.tensor(metadata["entity_ids"], dtype=torch.long)
    target_loss_paths, shadow_path_groups = resolve_target_shadow_paths(config, entity_ids)
    n_entities = int(metadata["n_entities"])
    n_population = int(metadata["n_samples"])
    entity_index_table = entity_index_table_from_entity_ids(metadata["entity_ids"])
    min_samples_per_entity = getattr(config, "min_samples_per_entity", 0)
    max_samples_per_entity = getattr(config, "max_samples_per_entity", None)
    audit_random_seed = int(getattr(config, "audit_random_seed", 0))
    n_loss_samples = getattr(config, "n_loss_samples", None)
    metrics_paths = []
    target_shadow_pairs = zip(target_loss_paths, shadow_path_groups)
    progress = tqdm(
        target_shadow_pairs,
        total=len(target_loss_paths),
        desc="Running entity-level audit",
    )
    for target_loss_path, shadow_loss_paths in progress:
        hold_out_frac = path_utils.parse_loss_signal_path(target_loss_path)["per_entity_hold_out"]
        _validate_entity_hold_out_indices(
            entity_index_table=entity_index_table,
            loss_paths=[target_loss_path, *shadow_loss_paths],
            target_len=n_population,
            hold_out_frac=hold_out_frac,
        )
        target_loss_sigs, target_train_mask = load_loss_signals(target_loss_path)
        if len(target_loss_sigs) != n_population:
            raise ValueError(f"Unexpected loss-signal length in {target_loss_path}: got {len(target_loss_sigs)}, expected {n_population}.")
        shadow_loss_sigs, shadow_train_mask = load_shadow_loss_signals(shadow_loss_paths, len(target_loss_sigs))
        target_seed = path_utils.audit_seed(target_loss_path, audit_random_seed)
        if n_loss_samples is not None:
            generator = torch.Generator().manual_seed(target_seed)
            target_loss_sigs = subsample_loss_signals(target_loss_sigs, n_loss_samples, generator)
            shadow_loss_sigs = subsample_loss_signals(shadow_loss_sigs, n_loss_samples, generator)
        loss_normalization = getattr(config, "loss_normalization", "log_standardized")
        target_loss_sigs, shadow_loss_sigs = normalize_loss_signals(
            target_loss_sigs,
            shadow_loss_sigs,
            loss_normalization,
        )
        target_train_index = utils.mask_to_index(target_train_mask)
        train_entity_ids = torch.unique(entity_ids[target_train_index])
        shadow_entity_mask = entitiy_train_mask(entity_ids=entity_ids, n_entities=n_entities, sample_train_mask=shadow_train_mask)
        audit_generator = torch.Generator().manual_seed(target_seed)
        audit_table = select_entity_audit_indices(
            entity_index_table=entity_index_table,
            train_mask=target_train_mask,
            mode=config.mode,
            min_samples_per_entity=min_samples_per_entity,
            max_samples_per_entity=max_samples_per_entity,
            hold_out_frac=hold_out_frac,
            generator=audit_generator,
        )
        audit_entity_ids = torch.tensor(sorted(audit_table.keys()), dtype=torch.long)
        ground_truth = torch.isin(audit_entity_ids, train_entity_ids).to(dtype=torch.long)

        attacker = get_attacker(
            attack_config=attack_config,
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_train_mask=shadow_train_mask,
            audit_table=audit_table,
            shadow_entity_mask=shadow_entity_mask,
        )
        match attack_config.attack:
            case "CompositeBASE":
                score = attacker.run_attack(audit_table=audit_table, target_loss_sigs=target_loss_sigs)
            case "CompositeLiRA_legacy" | "CompositeLiRA" | "HBE_Simple" | "HBE_GlobalLatent":
                score = attacker.run_attack(target_loss_sigs)
            case _:
                raise ValueError(f"Unsupported entity-level attack: {attack_config.attack}")
        score = torch.stack([score[entity_id] for entity_id in sorted(audit_table.keys())]).to(dtype=torch.float32)
        assert len(score) == len(ground_truth)
        metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        if hasattr(attacker, "calibration_diagnostics"):
            metrics["calibration_diagnostics"] = attacker.calibration_diagnostics
        if hasattr(attacker, "fit_diagnostics"):
            metrics["fit_diagnostics"] = attacker.fit_diagnostics
        metrics["audit_entity_ids"] = audit_entity_ids.tolist()
        metrics["audit_table"] = audit_table
        metrics_path = result_store.save_audit_metrics(config, metrics, target_loss_path, shadow_loss_paths, sample_attack)
        metrics_paths.append(metrics_path)
    manifest_path = result_store.write_audit_manifest(config, metrics_paths)
    if getattr(config, "print_summary", True):
        evaluation.print_grouped_metrics(
            metrics_paths[0].parent,
            show_auc=getattr(config, "show_auc", False),
            manifest_path=manifest_path,
        )

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
