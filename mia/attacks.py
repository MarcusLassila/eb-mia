from data.utils import load_dataset
from generative_models import VAE
from generative_models.utils import load_model
from utils import index_to_mask

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from scipy.stats import norm

class MIA(ABC):

    def __init__(self):
        self.device = torch.device("cpu")
        self.n_loss_samples = 1

    def load_model(self, path):
        model, train_indices = load_model(
            path=path,
            device=self.device,
        )
        if isinstance(model, VAE):
            model.n_rsamples = self.n_loss_samples
        return model, train_indices

    @abstractmethod
    def run_attack(self, audit_samples, target_model_path):
        raise NotImplementedError

class CompositeMIA(ABC):

    @abstractmethod
    def run_attack(self, sample_scores_by_entity):
        raise NotImplementedError

def compute_averaged_loss(model, samples, n_loss_samples):
    match model.__class__.__name__:
        case "DDPM":
            loss_samples = []
            for _ in range(n_loss_samples):
                t = torch.ones(size=(samples.shape[0],), device=samples.device, dtype=torch.long) * int(model.time_steps * 0.1)
                loss = model.per_sample_loss(samples, t).cpu()
                loss_samples.append(loss)
            avg_loss = torch.stack(loss_samples).mean(dim=0)
        case "VAE":
            avg_loss = model.per_sample_loss(samples).cpu()
        case _:
            raise ValueError("Unavailable class of generative model.")
    return avg_loss

class BASE(MIA):

    def __init__(self, batch_size, device, shadow_model_paths, len_dataset, offline=True, prior=0.5, n_loss_samples=1):
        self.batch_size = batch_size
        self.device = device
        self.shadow_model_paths = shadow_model_paths
        self.len_dataset = len_dataset
        self.offline = offline
        self.prior = prior
        self.n_loss_samples = n_loss_samples

    @torch.inference_mode()
    def loss_signal(self, audit_loader, model):
        sig = []
        for samples in tqdm(audit_loader, total=len(audit_loader), desc=f"Computing loss signal"):
            samples = samples.to(self.device)
            sig.append(compute_averaged_loss(model, samples, self.n_loss_samples))
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    def run_attack(self, audit_samples, target_model_path):
        assert target_model_path not in self.shadow_model_paths, "Should not attack the reference models"
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        target_model, _ = self.load_model(target_model_path)
        sig_target = self.loss_signal(audit_loader, target_model)
        del target_model
        sig_shadow_models = []
        masks = []
        for model_path in self.shadow_model_paths:
            shadow_model, shadow_train_index = self.load_model(model_path)
            masks.append(~index_to_mask(shadow_train_index, self.len_dataset))  # audit_samples is currently assumed to be the entire dataset 
            sig = self.loss_signal(audit_loader, shadow_model)
            sig_shadow_models.append(sig)
        sig_shadow_models = torch.stack(sig_shadow_models, dim=1)
        masks = torch.stack(masks, dim=1)
        if self.offline:
            sig_shadow_models = sig_shadow_models[masks].reshape(-1, len(self.shadow_model_paths) // 2)
        ref = torch.logsumexp(-sig_shadow_models, dim=1) - np.log(sig_shadow_models.shape[1])
        lam = np.log(self.prior / (1 - self.prior))
        score = -sig_target - ref + lam
        return score.sigmoid()

class LiRA(MIA):

    def __init__(self, batch_size, device, shadow_model_paths, len_dataset, offline=True, n_loss_samples=1, eps=1e-11):
        self.batch_size = batch_size
        self.device = device
        self.shadow_model_paths = shadow_model_paths
        self.len_dataset = len_dataset
        self.offline = offline
        self.n_loss_samples = n_loss_samples
        self.eps = eps

    @torch.inference_mode()
    def loss_signal(self, audit_loader, model):
        sig = []
        for samples in tqdm(audit_loader, total=len(audit_loader), desc=f"Computing loss signal"):
            samples = samples.to(self.device)
            sig.append(compute_averaged_loss(model, samples, self.n_loss_samples))
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    @staticmethod
    def logit_scale_transformation(loss_sigs):
        return -loss_sigs - torch.log1p(-torch.exp(-loss_sigs))

    @staticmethod
    def _mean_and_std(phi, mask):
        n = mask.sum(dim=0)
        assert torch.all(n)
        phi = phi * mask
        mean = phi.sum(dim=0) / n
        var = (phi ** 2).sum(dim=0) / n - mean ** 2
        return mean, var.clamp_min(0.0).sqrt()

    def query_shadow_models(self, audit_loader):
        in_mask = []
        loss_sigs = []
        for model_path in self.shadow_model_paths:
            shadow_model, shadow_train_index = self.load_model(model_path)
            train_mask = index_to_mask(shadow_train_index, self.len_dataset)
            in_mask.append(train_mask)
            loss_sigs.append(self.loss_signal(audit_loader, shadow_model))
        loss_sigs = torch.stack(loss_sigs, dim=0)
        phi = self.logit_scale_transformation(loss_sigs) 
        in_mask = torch.stack(in_mask, dim=0)
        out_mask = ~in_mask

        if self.offline:
            mean_in, std_in = None, None
        else:
            mean_in, std_in = self._mean_and_std(phi, in_mask)
        mean_out, std_out = self._mean_and_std(phi, out_mask)
        return mean_in, std_in, mean_out, std_out

    def run_attack(self, audit_samples, target_model_path):
        assert target_model_path not in self.shadow_model_paths, "Should not attack the reference models"
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        mean_in, std_in, mean_out, std_out = self.query_shadow_models(audit_loader)
        target_model, _ = self.load_model(target_model_path)
        loss_sig = self.loss_signal(audit_loader, target_model)
        phi = self.logit_scale_transformation(loss_sig)
        if self.offline:
            score = norm.logcdf(
                phi.cpu().numpy(),
                loc=mean_out.cpu().numpy(),
                scale=std_out.cpu().numpy() + self.eps,
            )
            score = torch.tensor(score)
        else:
            p_in = norm.logcdf(
                phi.cpu().numpy(),
                loc=mean_in.cpu().numpy(),
                scale=std_in.cpu().numpy() + self.eps,
            )
            p_out = norm.logcdf(
                phi.cpu().numpy(),
                loc=mean_out.cpu().numpy(),
                scale=std_out.cpu().numpy() + self.eps,
            )
            score = torch.tensor(p_in - p_out)
        return score

class CompositeBASE(CompositeMIA):

    def __init__(self):
        pass

    def run_attack(self, sample_scores_by_entity):
        assert isinstance(sample_scores_by_entity, dict)
        score = {}
        for entity_id, sample_scores in sample_scores_by_entity.items():
            assert isinstance(sample_scores, torch.Tensor)
            base_probs = sample_scores.to(dtype=torch.float32).clamp(min=0.0, max=1.0 - 1e-12)
            # Numerically stable implementation of 1 - (1 - base_probs).prod()
            score[entity_id] = -torch.expm1(torch.log1p(-base_probs).sum())
        return score
