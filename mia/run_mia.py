from data.utils import load_dataset
from . import attacks
import utils

import argparse
import torch
from tqdm.auto import tqdm

from pathlib import Path
import pickle
import yaml

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
        case _:
            raise ValueError(f"No MIA: {attack_config.attack}")
    return attacker

def scores_pickle_name(target_path, attack, audit_mode, min_samples_per_entity=None, max_samples_per_entity=None, n_audit_samples_per_entity=None):
    '''Return normalized score pickle filename. Args: target_path (str|Path), attack (str), audit_mode (str), min_samples_per_entity (int|None), max_samples_per_entity (int|None), n_audit_samples_per_entity (int|None). Returns: str.'''
    parts = [
        "scores",
        f"attack-{attack}",
        f"target-{Path(target_path).stem}",
        f"mode-{audit_mode}",
    ]
    if audit_mode == "entity":
        if n_audit_samples_per_entity is not None:
            parts.append(f"n-{n_audit_samples_per_entity}")
        min_value = "none" if min_samples_per_entity is None else str(min_samples_per_entity)
        max_value = "none" if max_samples_per_entity is None else str(max_samples_per_entity)
        parts.append(f"min-{min_value}")
        parts.append(f"max-{max_value}")
    return "_".join(parts) + ".pkl"

def parse_args(argv=None):
    '''Parse CLI arguments for sample-level MIA scoring. Args: argv (list[str]|None). Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Run sample-level membership inference and save scores.")
    default_config_path = str(utils.resolve_path(Path("mia") / "configs" / "config_mia.yaml", utils.get_root()))
    parser.add_argument(
        "--config",
        default=default_config_path,
        help="Path to audit config yaml file.",
    )
    return parser.parse_args(argv)

def run_mia(config, device):
    '''Run sample-level MIA on the full dataset and save score lists. Args: config (Config), device (torch.device)'''
    image_size = utils.parse_properties_from_checkpoint_path(config.target_model_paths[0])["size"]
    data_population = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
    target_model_paths = list(map(Path, config.target_model_paths))
    attack_config = utils.Config(config.attack)
    attack = attack_config.attack
    if not config.round_robin:
        shadow_model_paths = list(map(Path, config.shadow_model_paths))
    for target_idx, target_path in enumerate(target_model_paths):
        if config.round_robin:
            shadow_model_indices = indices_of_shadow_models(target_idx, len(target_model_paths))
            shadow_model_paths = [model_path for i, model_path in enumerate(target_model_paths) if i in shadow_model_indices]
        attacker = get_attacker(
            attack_config=attack_config,
            batch_size=config.batch_size,
            device=device,
            shadow_model_paths=shadow_model_paths,
        )
        score = attacker.run_attack(data_population, target_path)
        assert len(score) == len(data_population)
        Path(f"{config.res_dir}/{attack}").mkdir(parents=True, exist_ok=True)
        filename = scores_pickle_name(target_path, attack, "sample")
        with open(f"{config.res_dir}/{attack}/{filename}", "wb") as file:
            pickle.dump(score.tolist(), file)

def main(argv=None):
    '''Entry point for sample-level MIA CLI. Args: argv (list[str]|None). Returns: None.'''
    args = parse_args(argv)
    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)
    config = utils.Config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_mia(config=config, device=device)

if __name__ == "__main__":
    main()
