from .utils import indices_of_shadow_models

from abc import ABC, abstractmethod
import numpy as np
import torch
from scipy.stats import multivariate_normal, norm
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class MIA(ABC):

    @abstractmethod
    def __init__(self, shadow_loss_sigs, shadow_train_mask, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_attack(self, target_loss_sigs: torch.Tensor):
        raise NotImplementedError

class BASE(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=True,
        prior=0.5,
        apply_sigmoid=True,
        **kwargs,
    ):
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_train_mask = shadow_train_mask
        self.offline = offline
        self.prior = prior
        self.apply_sigmoid = apply_sigmoid
        if offline:
            self.shadow_loss_sigs = self.shadow_loss_sigs.masked_fill(shadow_train_mask, torch.inf)
            n_shadow_models = (~shadow_train_mask).to(torch.int32).sum(dim=0)
        else:
            n_shadow_models = torch.tensor([self.shadow_loss_sigs.shape[0]], torch.int32)
        self.ref = torch.logsumexp(-self.shadow_loss_sigs, dim=0) - torch.log(n_shadow_models)
        self.t_l = np.log(self.prior / (1 - self.prior))

    def run_attack(self, target_loss_sigs):
        score = -target_loss_sigs - self.ref + self.t_l
        if self.apply_sigmoid:
            score = score.sigmoid()
        return score

class NormalBASE(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=True,
        prior=0.5,
        use_global_var=True,
        apply_sigmoid=True,
        **kwargs,
    ):
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_train_mask = shadow_train_mask
        self.offline = offline
        self.prior = prior
        self.apply_sigmoid = apply_sigmoid
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
        if self.apply_sigmoid:
            score = score.sigmoid()
        return score

class LiRA(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=True,
        use_global_var=True,
        loss_transformation="logit_scaling",
        **kwargs,
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

class CompositeBASE:

    def __init__(self, attack_config, shadow_loss_sigs, shadow_train_mask):
        self.attack_config = attack_config
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_train_mask = shadow_train_mask
        self.base_attack = BASE(
            shadow_loss_sigs=self.shadow_loss_sigs,
            shadow_train_mask=self.shadow_train_mask,
            offline=attack_config.offline,
            prior=attack_config.prior,
            apply_sigmoid=True,
        )

    def run_attack(self, audit_table, target_loss_sigs):
        sample_scores = self.base_attack.run_attack(target_loss_sigs)
        scores = {}
        for entity_id, indices in audit_table.items():
            base_probs = sample_scores[indices].to(dtype=torch.float32).clamp(min=0.0, max=1.0-1e-12)
            scores[entity_id] = -torch.expm1(torch.log1p(-base_probs).sum())
        return scores

class CompositeLiRA:

    def __init__(self, audit_table, shadow_loss_sigs, shadow_entity_mask, offline=True):
        self.audit_table = audit_table
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_entity_mask = shadow_entity_mask
        self.offline = offline
        self.mean_in, self.var_in, self.mean_out, self.var_out = self.get_mean_and_var()

    def loss_transformation(self, loss_sigs):
        # TODO: Add other loss transformations
        return -loss_sigs

    def get_mean_and_var(self):
        phi_shadow = self.loss_transformation(self.shadow_loss_sigs)
        mean_in = {}
        mean_out = {}
        phi_in = {}
        phi_out = {}
        for entity_id, indices in self.audit_table.items():
            phi_in[entity_id] = phi_shadow[self.shadow_entity_mask[:, entity_id]][:, indices]
            phi_out[entity_id] = phi_shadow[~self.shadow_entity_mask[:, entity_id]][:, indices]
            mean_in[entity_id] = phi_in[entity_id].mean(dim=0)
            mean_out[entity_id] = phi_out[entity_id].mean(dim=0)
        var_in = torch.cat([*phi_in.values()], dim=0).var().clamp_min(1e-9)
        var_out = torch.cat([*phi_out.values()], dim=0).var().clamp_min(1e-9)
        return mean_in, var_in, mean_out, var_out

    def run_attack(self, target_loss_sigs):
        phi_target = self.loss_transformation(target_loss_sigs)
        score = {}
        for entity_id, indices in self.audit_table.items():
            cov_in = np.eye(len(indices), dtype=np.float32) * float(self.var_in)
            cov_out = np.eye(len(indices), dtype=np.float32) * float(self.var_out)
            if self.offline:
                p = multivariate_normal.logcdf(
                    phi_target[indices].cpu().numpy(),
                    mean=self.mean_out[entity_id].cpu().numpy(),
                    cov=cov_out,
                )
                score[entity_id] = torch.tensor(p, dtype=torch.float32)
            else:
                p_in = multivariate_normal.logpdf(
                    phi_target[indices].cpu().numpy(),
                    mean=self.mean_in[entity_id].cpu().numpy(),
                    cov=cov_in,
                )
                p_out = multivariate_normal.logpdf(
                    phi_target[indices].cpu().numpy(),
                    mean=self.mean_out[entity_id].cpu().numpy(),
                    cov=cov_out,
                )
                score[entity_id] = torch.tensor(p_in - p_out, dtype=torch.float32)
        return score

class JointXGB:

    def __init__(self, attack_config, audit_table, shadow_loss_sigs, shadow_train_mask, shadow_entity_mask, sample_level_attack="BASE"):
        self.attack_config = attack_config
        self.audit_table = audit_table
        assert len({len(indices) for indices in audit_table.values()}) == 1, "XGB requires a fixed number of features"
        self.n_shadow_models = shadow_loss_sigs.shape[0]
        assert self.n_shadow_models == shadow_train_mask.shape[0]
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_train_mask = shadow_train_mask
        self.shadow_entity_mask = shadow_entity_mask
        self.sample_level_attack = sample_level_attack
        X_train, X_test, y_train, y_test = self.create_dataset()
        self.model = self.fit_model(X_train, X_test, y_train, y_test)

    def get_sample_scores(self, target_loss_sigs, shadow_loss_sigs, shadow_train_mask):
        config = self.attack_config
        match self.sample_level_attack:
            case "BASE":
                attacker = BASE(
                    shadow_loss_sigs=shadow_loss_sigs,
                    shadow_train_mask=shadow_train_mask,
                    offline=config.offline,
                    prior=config.prior,
                    apply_sigmoid=False,
                )
            case _:
                raise ValueError(f"Unsupported sample-level attack: {self.sample_level_attack}")
        scores = attacker.run_attack(target_loss_sigs)
        scores = scores.to(dtype=torch.float32).cpu().numpy()
        return scores

    def create_dataset(self):
        features, labels = [], []
        for shadow_index in range(self.n_shadow_models):
            simul_shadow_indices = indices_of_shadow_models(shadow_index, self.n_shadow_models)
            sample_scores = self.get_sample_scores(
                self.shadow_loss_sigs[shadow_index],
                self.shadow_loss_sigs[simul_shadow_indices],
                self.shadow_train_mask[simul_shadow_indices],
            )
            for entity_id, indices in self.audit_table.items():
                features.append(sample_scores[indices])
                labels.append(int(self.shadow_entity_mask[shadow_index, entity_id]))
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return train_test_split(features, labels, test_size=0.2, random_state=42)

    def fit_model(self, X_train, X_test, y_train, y_test):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"XGB model fitted. Test accuracy: {accuracy}")
        return model

    def run_attack(self, target_loss_sigs):
        target_sample_scores = self.get_sample_scores(target_loss_sigs, self.shadow_loss_sigs, self.shadow_train_mask)
        score = {}
        for entity_id, indices in self.audit_table.items():
            prob_label_1 = self.model.predict_proba(target_sample_scores[indices].reshape(1, -1))[0, 1]
            score[entity_id] = torch.tensor(prob_label_1, dtype=torch.float32)
        return score
