from abc import ABC, abstractmethod
import numpy as np
import torch
from scipy.stats import multivariate_normal, norm

class MIA(ABC):

    @abstractmethod
    def run_attack(self, target_loss_sigs: torch.Tensor):
        raise NotImplementedError

class BASE(MIA):

    def __init__(self, shadow_loss_sigs: torch.Tensor, shadow_train_mask: torch.Tensor, offline=True, prior=0.5):
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_train_mask = shadow_train_mask
        self.offline = offline
        self.prior = prior
        if offline:
            self.shadow_loss_sigs = self.shadow_loss_sigs.masked_fill(shadow_train_mask, torch.inf)
            n_shadow_models = (~shadow_train_mask).to(torch.int32).sum(dim=0)
        else:
            n_shadow_models = torch.tensor([self.shadow_loss_sigs.shape[0]], torch.int32)
        self.ref = torch.logsumexp(-self.shadow_loss_sigs, dim=0) - torch.log(n_shadow_models)
        self.t_l = np.log(self.prior / (1 - self.prior))

    def run_attack(self, target_loss_sigs):
        score = -target_loss_sigs - self.ref + self.t_l
        return score.sigmoid()

class NormalBASE(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=True,
        prior=0.5,
        use_global_var=True,
    ):
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_train_mask = shadow_train_mask
        self.offline = offline
        self.prior = prior
        if offline:
            self.shadow_loss_sigs = self.shadow_loss_sigs.masked_fill(shadow_train_mask, 0)
            n_shadow_models = (~shadow_train_mask).to(torch.int32).sum(dim=0)
        else:
            n_shadow_models = torch.tensor([self.shadow_loss_sigs.shape[0]], torch.int32)
        mean = torch.sum(-self.shadow_loss_sigs, dim=0) / n_shadow_models
        if use_global_var:
            var = self.shadow_loss_sigs[~shadow_train_mask].var() if offline else self.shadow_loss_sigs.var()
        else:
            var = (self.shadow_loss_sigs ** 2).sum(dim=0) / n_shadow_models - mean ** 2
        self.ref = mean + 0.5 * var
        self.t_l = np.log(self.prior / (1 - self.prior))

    def run_attack(self, target_loss_sigs):
        score = -target_loss_sigs - self.ref + self.t_l
        return score.sigmoid()

class LiRA(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=True,
        use_global_var=True,
        loss_transformation="logit_scaling",
    ):
        self.offline = offline
        self.use_global_var = use_global_var
        self.loss_transformation = loss_transformation
        shadow_phi = self.transform_loss_values(shadow_loss_sigs)
        if self.offline:
            self.mean_in = None
            self.std_in = None
        else:
            self.mean_in, self.std_in = self._mean_and_std(shadow_phi, shadow_train_mask)
        self.mean_out, self.std_out = self._mean_and_std(shadow_phi, ~shadow_train_mask)

    def transform_loss_values(self, loss_sigs):
        match self.loss_transformation:
            case "logit_scaling":
                rescaled_sigs = -loss_sigs - torch.log1p(-torch.exp(-loss_sigs))
            case "nll":
                rescaled_sigs = torch.exp(-loss_sigs)
            case "none":
                rescaled_sigs = -loss_sigs # Assuming loss is minimized but larger scores are more likely members
            case _:
                raise ValueError(f"Unavailable transformation {self.loss_transformation}")
        return rescaled_sigs

    def _mean_and_std(self, phi, mask):
        n = mask.sum(dim=0)
        assert torch.all(n)
        phi_m = phi * mask
        mean = phi_m.sum(dim=0) / n
        if self.use_global_var:
            std = phi[mask].std()
        else:
            var = (phi_m ** 2).sum(dim=0) / n - mean ** 2
            std = var.clamp_min(0.0).sqrt()
        return mean.cpu().numpy(), std.cpu().numpy() + 1e-11

    def run_attack(self, target_loss_sigs):
        phi = self.transform_loss_values(target_loss_sigs).cpu().numpy()
        if self.offline:
            score = norm.logcdf(
                phi,
                loc=self.mean_out,
                scale=self.std_out,
            )
            score = torch.tensor(score)
        else:
            p_in = norm.logpdf(
                phi,
                loc=self.mean_in,
                scale=self.std_in,
            )
            p_out = norm.logpdf(
                phi,
                loc=self.mean_out,
                scale=self.std_out,
            )
            score = torch.tensor(p_in - p_out)
        return score

def composite_BASE(sample_scores_by_entity: dict):
    assert isinstance(sample_scores_by_entity, dict)
    score = {}
    for entity_id, sample_scores in sample_scores_by_entity.items():
        assert isinstance(sample_scores, torch.Tensor)
        base_probs = sample_scores.to(dtype=torch.float32).clamp(min=0.0, max=1.0 - 1e-12)
        # Numerically stable implementation of 1 - (1 - base_probs).prod()
        score[entity_id] = -torch.expm1(torch.log1p(-base_probs).sum())
    return score

def composite_LiRA(audit_table, loss_sigs, shadow_entity_mask, offline=True):
    phi = -loss_sigs
    phi_target, phi_shadow = phi[0], phi[1:]
    mean_in = {}
    mean_out = {}
    phi_in = {}
    phi_out = {}
    for entity_id, indices in audit_table.items():
        phi_in[entity_id] = phi_shadow[shadow_entity_mask[:, entity_id]][:, indices]
        phi_out[entity_id] = phi_shadow[~shadow_entity_mask[:, entity_id]][:, indices]
        mean_in[entity_id] = phi_in[entity_id].mean(dim=0)
        mean_out[entity_id] = phi_out[entity_id].mean(dim=0)
    var_in = torch.cat([*phi_in.values()], dim=0).var().clamp_min(1e-9)
    var_out = torch.cat([*phi_out.values()], dim=0).var().clamp_min(1e-9)
    score = {}
    for entity_id, indices in audit_table.items():
        cov_in = np.eye(len(indices), dtype=np.float32) * float(var_in)
        cov_out = np.eye(len(indices), dtype=np.float32) * float(var_out)
        if offline:
            p = multivariate_normal.logcdf(
                phi_target[indices].cpu().numpy(),
                mean=mean_out[entity_id].cpu().numpy(),
                cov=cov_out,
            )
            score[entity_id] = torch.tensor(p, dtype=torch.float32)
        else:
            p_in = multivariate_normal.logpdf(
                phi_target[indices].cpu().numpy(),
                mean=mean_in[entity_id].cpu().numpy(),
                cov=cov_in,
            )
            p_out = multivariate_normal.logpdf(
                phi_target[indices].cpu().numpy(),
                mean=mean_out[entity_id].cpu().numpy(),
                cov=cov_out,
            )
            score[entity_id] = torch.tensor(p_in - p_out, dtype=torch.float32)
    return score
