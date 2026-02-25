from data.utils import load_dataset
from generative_models import VAE
from generative_models.utils import load_model

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader

class MIA(ABC):

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
                t = torch.ones(size=samples.shape[0], device=samples.device, dtype=torch.long) * int(model.time_steps * 0.1)
                loss = model.per_sample_loss(samples, t).cpu()
                loss_samples.append(loss)
            avg_loss = torch.stack(loss_samples).mean(dim=0)
        case "VAE":
            avg_loss = model.per_sample_loss(samples).cpu()
        case _:
            raise ValueError("Unavailable class of generative model.")
    return avg_loss

class BASE(MIA):

    def __init__(self, batch_size, device, shadow_model_paths, prior=0.5, n_loss_samples=1):
        self.batch_size = batch_size
        self.device = device
        self.shadow_model_paths = shadow_model_paths
        self.prior = prior
        self.n_loss_samples = n_loss_samples

    def load_model(self, path):
        model, _ = load_model(
            path=path,
            device=self.device,
        )
        if isinstance(model, VAE):
            model.n_rsamples = self.n_loss_samples
        return model

    @torch.inference_mode()
    def loss_signal(self, audit_loader, model_path):
        model = self.load_model(model_path)
        sig = []
        for samples in audit_loader:
            samples = samples.to(self.device)
            sig.append(compute_averaged_loss(model, samples, self.n_loss_samples))
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    def run_attack(self, audit_samples, target_model_path):
        assert target_model_path not in self.shadow_model_paths, "Should not attack the reference models"
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        sig_target = self.loss_signal(audit_loader, target_model_path)
        sig_shadow_models = []
        for model_path in self.shadow_model_paths:
            sig = self.loss_signal(audit_loader, model_path)
            sig_shadow_models.append(sig)
        sig_shadow_models = torch.stack(sig_shadow_models)
        score = -sig_target - torch.logsumexp(-sig_shadow_models, dim=0) + np.log(self.prior / (1 - self.prior))
        return score.sigmoid()

class CompositeBASE(CompositeMIA):

    def __init__(self, prior=0.5):
        self.prior = prior

    def run_attack(self, sample_scores_by_entity):
        assert isinstance(sample_scores_by_entity, dict)
        score = {}
        for entity_id, sample_scores in sample_scores_by_entity.items():
            assert isinstance(sample_scores, torch.Tensor)
            base_probs = sample_scores.to(dtype=torch.float32).clamp(min=0.0, max=1.0 - 1e-12)
            # Numerically stable implementation of 1 - (1 - base_probs).prod()
            score[entity_id] = -torch.expm1(torch.log1p(-base_probs).sum())
        return score
