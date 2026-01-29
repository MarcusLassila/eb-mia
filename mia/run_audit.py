from data.data import EntityDataset
from data.utils import load_dataset
from . import attacks
from . import evaluation
import utils

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from pathlib import Path
import pickle
import yaml

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

def get_entity_audit_table(data_population: EntityDataset):
    table = data_population.get_entity_index_table()
    audit_samples = {}
    for entity_id, indices in table.items():
        audit_samples[entity_id] = Subset(data_population, indices)

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

def run_audit(config, device):
    data_population = load_dataset(config.dataset)
    target_model_paths = list(map(Path, config.target_model_paths))
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
        for attack, attack_dict in config.attacks.items():
            attacker = get_attacker(
                attack_config=utils.Config(attack_dict),
                batch_size=config.batch_size,
                device=device,
                shadow_model_paths=shadow_model_paths,
            )
            score = attacker.run_attack(audit_samples, target_path)
            metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
            Path(f"{config.res_dir}/{attack}").mkdir(parents=True, exist_ok=True)
            model_id = target_path.stem
            with open(f"{config.res_dir}/{attack}/metrics_{model_id}.pkl", "wb") as f:
                pickle.dump(metrics, f)

def run_entity_audit(config, device):
    data_population = load_dataset(config.dataset)
    assert isinstance(data_population, EntityDataset)
 
    target_model_paths = list(map(Path, config.target_model_paths))
    if not config.round_robin:
        shadow_model_paths = list(map(Path, config.shadow_model_paths))
    for target_idx, target_path in tqdm(enumerate(target_model_paths), total=len(target_model_paths), desc="Running audit"):
        if config.round_robin:
            shadow_model_indices = indices_of_shadow_models(target_idx, len(target_model_paths))
            shadow_model_paths = [model_path for i, model_path in enumerate(target_model_paths) if i in shadow_model_indices]
        target_train_index = utils.get_train_indices(target_path)
        train_entity_ids = torch.unique(data_population.entity_ids[target_train_index])
        audit_table = get_entity_audit_table()
        ground_truth = {entity_id: 0 for entity_id in audit_table.keys()}
        for entity_id in train_entity_ids:
            entity_id = entity_id.item()
            if entity_id in ground_truth:
                ground_truth[entity_id] = 1
        ground_truth = torch.tensor(ground_truth.values(), dtype=torch.long)

        for attack, attack_dict in config.attacks.items():
            attacker = get_attacker(
                attack_config=utils.Config(attack_dict),
                batch_size=config.batch_size,
                device=device,
                shadow_model_paths=shadow_model_paths,
            )
            score = attacker.run_attack(audit_table, target_path)
            score = torch.tensor(score.values(), dtype=torch.float32)
            metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
            Path(f"{config.res_dir}/{attack}").mkdir(parents=True, exist_ok=True)
            model_id = target_path.stem
            with open(f"{config.res_dir}/{attack}/metrics_{model_id}.pkl", "wb") as f:
                pickle.dump(metrics, f)

if __name__ == "__main__":
    root = utils.get_root()
    with open(f"{root}/mia/configs/config_audit.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_audit(
        config=config,
        device=device,
    )
