from data import data
import utils

import numpy as np
import torch
import time
import pickle
from pathlib import Path

class AnnealedImportanceSampling:
    '''Hardcoded Metropolis MCMC kernel for now.'''

    def __init__(self, dim, beta_schedule, log_p1, device, n_samples=2048, n_steps_per_sample=20):
        self.dim = dim
        self.beta_schedule = beta_schedule
        self.n_betas = len(self.beta_schedule)
        self.log_p1 = log_p1
        self.device = device
        self.n_samples = n_samples
        self.n_steps_per_sample = n_steps_per_sample
        self.cov_scale_factor = 2.38 ** 2 / dim

    def log_p0(self, x):
        '''
        Log density of standard multivariate normal distribution.
        '''
        return -0.5 * torch.einsum("bi,bi->b", x, x) - 0.5 * self.dim * np.log(2 * np.pi)

    def log_pt(self, x, t):
        '''
        Log density of intermediate distribution.
        The intermediate distribution is a weighted geometric average of the standard normal distribution and the target distribution.
        '''
        beta = self.beta_schedule[t]
        return (1 - beta) * self.log_p0(x) + beta * self.log_p1(x)

    @torch.inference_mode()
    def run(self):
        device = self.device
        X = torch.randn(size=(self.n_samples, self.dim), device=device)
        log_p0_curr = self.log_p0(X)
        log_p1_curr = self.log_p1(X)
        log_w = torch.zeros(self.n_samples, device=device)
        t0 = time.time()
        for t in range(1, self.n_betas):
            beta_prev = self.beta_schedule[t - 1]
            beta = self.beta_schedule[t]
            log_w += (beta - beta_prev) * (log_p1_curr - log_p0_curr)

            X_c = X - X.mean(dim=0, keepdim=True)
            cov = X_c.t() @ X_c / (self.n_samples - 1)
            S = self.cov_scale_factor * (cov + 1e-5 * torch.eye(self.dim, device=device))
            L = torch.linalg.cholesky(S)

            accept_rate = 0.0
            for _ in range(self.n_steps_per_sample):
                z = torch.randn_like(X, device=device)
                X_prop = X + z @ L.t()

                log_p0_prop = self.log_p0(X_prop)
                log_p1_prop = self.log_p1(X_prop)

                log_pt_curr = (1 - beta) * log_p0_curr + beta * log_p1_curr
                log_pt_prop = (1 - beta) * log_p0_prop + beta * log_p1_prop
                log_u = torch.rand(self.n_samples, device=device).log()
                accept_mask = log_u < (log_pt_prop - log_pt_curr)
                n_accept = accept_mask.sum().item()
                accept_rate += n_accept
                if n_accept > 0:
                    X[accept_mask] = X_prop[accept_mask]
                    log_p0_curr[accept_mask] = log_p0_prop[accept_mask]
                    log_p1_curr[accept_mask] = log_p1_prop[accept_mask]
            t1 = time.time()
            accept_rate /= self.n_samples * self.n_steps_per_sample
            log_msg = f"t: {t} | accept rate: {accept_rate:.5f} | time: {t1 - t0:.1f}"
            print(log_msg, flush=True)

        # log(Z) = log(Z0) + LogSumExp(log_w) - log(n_samples) but log(Z0) = 0 since p0 is already normalized
        log_Z = torch.logsumexp(log_w, dim=0) - np.log(self.n_samples)
        return {
            "log_Z": log_Z.item(),
            # Add other quantities of interest here
        }

def unnormalized_log_prob(loss_fn, shape):
    def wrapper(x):
        x = x.view(x.shape[0], *shape)
        loss = loss_fn(x)
        return -loss
    return wrapper

def compute_partition_functions(path, dataset_name, model_type, device):
    data_shape = getattr(data, dataset_name)()[0].shape
    dim = torch.tensor(data_shape).prod()
    beta_schedule = torch.linspace(0, 1, steps=500, device=device)
    model, _ = utils.load_model(
        path=path,
        model_type=model_type,
        device=device,
        n_loss_samples=20,
    )
    log_p1 = unnormalized_log_prob(model.per_sample_loss, data_shape)
    sampler = AnnealedImportanceSampling(
        dim=dim,
        beta_schedule=beta_schedule,
        log_p1=log_p1,
        device=device,
    )
    log_Z = sampler.run()["log_Z"]
    print(f"log(Z) = {log_Z}")
    savedir = path.parent / Path("partition-functions")
    savedir.mkdir(parents=True, exists=True)
    savepath = savedir / path.name
    with open(savepath, "wb") as f:
        pickle.dump(log_Z, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to model"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_partition_functions(
        path=args.path,
        dataset_name=args.dataset,
        model_type=args.model,
        device=device,
    )
