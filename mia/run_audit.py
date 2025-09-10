from data import data
import ais
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

def compute_partition_functions(n_models, dataset_name, model_type, device):
    data_shape = getattr(data, dataset_name)()[0].shape
    dim = torch.tensor(data_shape).prod()
    beta_schedule = torch.linspace(0, 1, steps=1000, device=device)
    partition_fns = []
    for index in range(n_models):
        model, _ = utils.load_model(
            dataset=dataset_name,
            model_type=model_type,
            index_model=index,
            device=device,
        )
        log_p1 = ais.unnormalized_log_prob(model.per_sample_loss, data_shape)
        sampler = ais.AnnealedImportanceSampling(
            dim=dim,
            beta_schedule=beta_schedule,
            log_p1=log_p1,
            device=device,
        )
        log_Z = sampler.run()["log_Z"]
        partition_fns.append(log_Z)
    savedir = f"{utils.get_root()}/trained_models/partition_functions/{model_type}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    with open(f"{savedir}/{dataset_name}_partition_functions.pkl", "wb") as f:
        pickle.dump(partition_fns, f)
    print("Partition functions computed:")
    for i, log_Z in enumerate(partition_fns):
        print(f"{i}: {log_Z}")
    return partition_fns

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

def get_attacker(attack_config, dataset_name, model_type, batch_size, device, index_ref_models, partition_fns):
    match attack_config.attack:
        case "ClassifierAttack":
            attacker = attacks.ClassifierAttack(
                dataset_name=dataset_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                index_ref_models=index_ref_models[:1], # just one for now
                n_loss_samples=attack_config.n_loss_samples,
                classifier=attack_config.classifier,
            )
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
                index_ref_models=index_ref_models,
                partition_fns=partition_fns if attack_config.calibrated else [0.0 for _ in range(N_MODELS)],
                prior=attack_config.prior,
                n_loss_samples=attack_config.n_loss_samples,
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
    root = utils.get_root()
    resdir = f"{root}/mia/results/{dataset_name}-{model_type}/"
    data_population = getattr(data, dataset_name)()
    partition_fns = compute_partition_functions(N_MODELS, dataset_name, model_type, device)
    for index_target in tqdm(range(n_audits), desc="Running audit"):
        target_train_indices = utils.get_train_indices(dataset_name, model_type, index_target)
        membership_mask = utils.index_to_mask(target_train_indices, len(data_population))
        audit_indices = get_audit_indices(n_audit_samples, membership_mask)
        audit_samples = Subset(data_population, audit_indices)
        index_ref_models = indices_of_ref_models(index_target, N_MODELS)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
        for attack, attack_dict in attack_config.items():
            attacker = get_attacker(
                attack_config=utils.Config(attack_dict),
                dataset_name=dataset_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                index_ref_models=index_ref_models,
                partition_fns=partition_fns,
            )
            score = attacker.run_attack(audit_samples, index_target)
            metrics = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
            Path(f"{resdir}/{attack}").mkdir(parents=True, exist_ok=True)
            with open(f"{resdir}/{attack}/metrics_{index_target}.pkl", "wb") as f:
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
        dataset_name=config.dataset,
        model_type=config.model,
        attack_config=config.attacks,
        batch_size=config.batch_size,
        n_audits=config.n_audits,
        n_audit_samples=config.n_audit_samples,
        device=device,
    )
