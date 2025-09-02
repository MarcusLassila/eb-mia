from vae import vae
import utils

import torch
import numpy as np

class UncalibratedBASE:

    def __init__(self, model_type, device, prior=0.5):
        self.model_type = model_type
        self.device = device
        self.prior = prior

    def loss_signal(self, audit_samples, index_model):
        model, _ = utils.load_model(
            dataset=audit_samples.dataset.__class__.__name__,
            model_type=self.model_type,
            index_model=index_model,
            device=self.device,
        )
        sig = []
        for sample in map(lambda x: x.unsqueeze(dim=0), audit_samples):
            loss = model.loss(sample)
            sig.append(loss)
        return torch.tensor(sig)

    def run_attack(self, audit_samples, index_target_model, index_ref_models):
        sig_target = self.loss_signal(audit_samples, index_target_model)
        sig_ref_models = []
        for idx in index_ref_models:
            sig = self.loss_signal(audit_samples, idx)
            sig_ref_models.append(sig)
        sig_ref_models = torch.stack(sig_ref_models)
        score = -sig_target - torch.logsumexp(-sig_ref_models, dim=0) + np.log(self.prior / (1 - self.prior))
        return score.sigmoid()
