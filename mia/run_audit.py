from data import data
from vae import vae
import attacks
import evaluation
import utils

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm
from statistics import mean, stdev

DATASET = "CIFAR10"
MODEL_TYPE = "VAE"
NUM_AUDITS = 10
NUM_AUDIT_SAMPLES = 2000

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

def run_audit():
    data_population = getattr(data, DATASET)()
    aurocs = []
    for index_target in tqdm(range(NUM_AUDITS), desc="Running audit"):
        target_train_indices = utils.get_train_indices(DATASET, MODEL_TYPE, index_target)
        membership_mask = utils.index_to_mask(target_train_indices, len(data_population))
        audit_indices = get_audit_indices(NUM_AUDIT_SAMPLES, membership_mask)
        audit_samples = Subset(data_population, audit_indices)
        index_ref_models = indices_of_ref_models(index_target, NUM_AUDITS)
        attacker = attacks.UncalibratedBASE(
            model_type=MODEL_TYPE,
            device=torch.device("cpu"),
            prior=0.5,
        )
        score = attacker.run_attack(audit_samples, index_target_model=index_target, index_ref_models=index_ref_models)
        ground_truth = membership_mask.to(dtype=torch.long)[audit_indices]
        auc = evaluation.evaluate_MIA(score=score, ground_truth=ground_truth)
        aurocs.append(auc)
    print(mean(aurocs), stdev(aurocs))

if __name__ == "__main__":
    run_audit()
