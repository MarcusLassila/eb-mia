from data import data
import attacks
import evaluation
import utils

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from collections import defaultdict
from statistics import mean, stdev
import yaml

def indices_of_ref_models(index_target, n_ref_models):
    assert 0 <= index_target < n_ref_models
    if index_target % 2 == 0:
        excluded_indices = {index_target, index_target + 1}
    else:
        excluded_indices = {index_target - 1, index_target}
    index_ref_models = sorted(set(range(n_ref_models)) - excluded_indices)
    return index_ref_models

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

def get_attacker(attack_config, dataset_name, model_type, batch_size, device, index_ref_models):
    match attack_config.attack:
        case "GlobalLossAttack":
            attacker = attacks.GlobalLossAttack(
                dataset_name=dataset_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
            )
        case "UncalibratedBASE":
            attacker = attacks.UncalibratedBASE(
                dataset_name=dataset_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                index_ref_models=index_ref_models,
                prior=attack_config.prior,
            )
        case _:
            raise ValueError(f"No attack:: {attack_config.attack}")
    return attacker

def run_audit(
        dataset_name,
        model_type,
        attack_config,
        batch_size,
        n_audits,
        n_audit_samples,
        device,
    ):
    data_population = getattr(data, dataset_name)()
    aurocs = defaultdict(list)
    for index_target in tqdm(range(n_audits), desc="Running audit"):
        target_train_indices = utils.get_train_indices(dataset_name, model_type, index_target)
        membership_mask = utils.index_to_mask(target_train_indices, len(data_population))
        audit_indices = get_audit_indices(n_audit_samples, membership_mask)
        audit_samples = Subset(data_population, audit_indices)
        index_ref_models = indices_of_ref_models(index_target, n_audits)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
        for attack, attack_dict in attack_config.items():
            attacker = get_attacker(
                attack_config=utils.Config(attack_dict),
                dataset_name=dataset_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                index_ref_models=index_ref_models,
            )
            score = attacker.run_attack(audit_samples, index_target)
            auc = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
            aurocs[attack].append(auc)
    for attack, auc in aurocs.items():
        print(f"{attack}: {mean(auc):.5f}, {stdev(auc):.5f}")

if __name__ == "__main__":
    root = utils.get_root()
    with open(f"{root}/mia/config_audit.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_audit(
        dataset_name=config.dataset,
        model_type=config.model,
        attack_config=config.attacks,
        batch_size=config.batch_size,
        n_audits=config.n_audits,
        n_audit_samples=config.n_audit_samples,
        device=device,
    )
