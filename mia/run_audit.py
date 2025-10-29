from data import data
import attacks
import evaluation
import utils

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from pathlib import Path
import pickle
import yaml

N_MODELS = 10

def load_partition_fn(model_path):
    pathdir = model_path.parent / Path("partition-functions")
    pathname = model_path.name
    with open(pathdir / pathname, "rb") as f:
        log_Z = pickle.load(f)
    return log_Z

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

def get_attacker(attack_config, dataset_name, model_type, batch_size, device, ref_model_paths, partition_fns):
    match attack_config.attack:
        # case "ClassifierAttack":
        #     attacker = attacks.ClassifierAttack(
        #         dataset_name=dataset_name,
        #         model_type=model_type,
        #         batch_size=batch_size,
        #         device=device,
        #         index_ref_models=index_ref_models[:1], # just one for now
        #         n_loss_samples=attack_config.n_loss_samples,
        #         classifier=attack_config.classifier,
        #     )
        case "GlobalLossAttack":
            attacker = attacks.GlobalLossAttack(
                dataset_name=dataset_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                n_loss_samples=attack_config.n_loss_samples,
            )
        case "BASE":
            attacker = attacks.BASE(
                dataset_name=dataset_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                ref_model_paths=ref_model_paths,
                partition_fns=partition_fns,
                prior=attack_config.prior,
                n_loss_samples=attack_config.n_loss_samples,
            )
        case _:
            raise ValueError(f"No attack: {attack_config.attack}")
    return attacker

def run_audit(config, device):
    root = utils.get_root()
    resdir = f"{root}/mia/results/{config.dataset}-{config.model_type}/"
    data_population = getattr(data, config.dataset_name)()
    partition_fns = {}
    for model_path in config.target_model_paths + config.ref_model_paths:
        log_Z = load_partition_fn(model_path)
        partition_fns[model_path] = log_Z
    for target_model_path in tqdm(config.target_model_paths, desc="Running audit"):
        target_train_indices = utils.get_train_indices(target_model_path, config.model_type)
        membership_mask = utils.index_to_mask(target_train_indices, len(data_population))
        audit_indices = get_audit_indices(config.n_audit_samples, membership_mask)
        audit_samples = Subset(data_population, audit_indices)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
        for attack, attack_dict in config.attacks.items():
            attacker = get_attacker(
                attack_config=utils.Config(attack_dict),
                dataset_name=config.dataset_name,
                model_type=config.model_type,
                batch_size=config.batch_size,
                device=device,
                ref_model_paths=config.ref_model_paths,
                partition_fns=partition_fns,
            )
            score = attacker.run_attack(audit_samples, target_model_path)
            metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
            Path(f"{resdir}/{attack}").mkdir(parents=True, exist_ok=True)
            model_id = target_model_path.stem
            with open(f"{resdir}/{attack}/metrics_{model_id}.pkl", "wb") as f:
                pickle.dump(metrics, f)

if __name__ == "__main__":
    root = utils.get_root()
    with open(f"{root}/mia/config_audit.yaml", "r") as file:
        config = yaml.safe_load(file)
    _, params = next(iter(config.items()))
    config = utils.Config(params)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_audit(
        config=config,
        device=device,
    )
