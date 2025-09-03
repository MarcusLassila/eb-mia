import utils

import numpy as np
import torch
from torch.utils.data import DataLoader

class GlobalLossAttack:

    def __init__(self, dataset_name, model_type, batch_size, device):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device

    def load_model(self, index):
        model, _ = utils.load_model(
            dataset=self.dataset_name,
            model_type=self.model_type,
            index_model=index,
            device=self.device,
        )
        return model

    @torch.inference_mode()
    def loss_signal(self, audit_loader, index_model):
        model = self.load_model(index_model)
        sig = []
        for samples in audit_loader:
            samples = samples.to(self.device)
            loss = model.per_sample_loss(samples).cpu()
            sig.append(loss)
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    def run_attack(self, audit_samples, index_target_model):
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        return -self.loss_signal(audit_loader, index_target_model)

class UncalibratedBASE:

    def __init__(self, dataset_name, model_type, batch_size, device, index_ref_models, prior=0.5):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        self.index_ref_models = index_ref_models
        self.prior = prior

    def load_model(self, index):
        model, _ = utils.load_model(
            dataset=self.dataset_name,
            model_type=self.model_type,
            index_model=index,
            device=self.device,
        )
        return model

    @torch.inference_mode()
    def loss_signal(self, audit_loader, index_model):
        model = self.load_model(index_model)
        sig = []
        for samples in audit_loader:
            samples = samples.to(self.device)
            loss = model.per_sample_loss(samples).cpu()
            sig.append(loss)
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    def run_attack(self, audit_samples, index_target_model):
        assert index_target_model not in self.index_ref_models, "Should not attack the reference models"
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        sig_target = self.loss_signal(audit_loader, index_target_model)
        sig_ref_models = []
        for idx in self.index_ref_models:
            sig = self.loss_signal(audit_loader, idx)
            sig_ref_models.append(sig)
        sig_ref_models = torch.stack(sig_ref_models)
        score = -sig_target - torch.logsumexp(-sig_ref_models, dim=0) + np.log(self.prior / (1 - self.prior))
        return score.sigmoid()
