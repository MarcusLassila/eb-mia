from data import data
import utils

import numpy as np
import torch
import torchvision.transforms as T
import time
import pickle
from pathlib import Path

class AnnealedImportanceSampling:
    '''Hardcoded Metropolis MCMC kernel for now.'''

    def __init__(self, dim, beta_schedule, log_p1, device, n_samples=2048, n_steps_per_sample=100):
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

            #X_c = X - X.mean(dim=0, keepdim=True)
            #cov = X_c.t() @ X_c / (self.n_samples - 1)
            #cov = 0.5 * (cov + cov.t())
            #S = self.cov_scale_factor * (cov + 1e-5 * torch.eye(self.dim, device=device))
            diag = torch.var(X, dim=0, unbiased=True) + 1e-6
            S = (2.38**2 / X.shape[1]) * torch.diag(diag)
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

def compute_partition_functions(path, steps, n_loss_samples, device, data_dir):
    path = Path(path)
    dataset_name, model_type, *_ = path.stem.split("-")
    if model_type == "DDPM":
        transform = T.Concat([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ])
    else:
        transform = T.ToTensor()
    data_shape = getattr(data, dataset_name)(transform=transform, data_dir=data_dir)[0].shape
    dim = torch.tensor(data_shape).prod()
    beta_schedule = torch.linspace(0, 1, steps=steps, device=device)
    model, _ = utils.load_model(
        path=path,
        device=device,
        n_loss_samples=n_loss_samples,
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
    savedir.mkdir(parents=True, exist_ok=True)
    savepath = savedir / path.name
    with open(savepath, "wb") as f:
        pickle.dump(log_Z, f)

@torch.inference_mode()
def test_multinormal_partition_fn(steps, n_steps_per_sample, device):
    dim = 50
    mean = torch.normal(mean=10.0, std=3.0, size=(dim,), device=device)
    rand_mat = torch.normal(mean=2.0, std=5.0, size=(dim, dim), device=device)
    cov = rand_mat @ rand_mat.t() + torch.eye(dim, device=device) * 1e-3
    L = torch.linalg.cholesky(cov)

    def log_p(x):
        ''' Unnormalized log pdf '''
        d = x - mean
        # Cholesky solve: LL^T z = d
        z = torch.cholesky_solve(d.unsqueeze(dim=-1), L).squeeze(dim=-1)
        p = -0.5 * torch.sum(d * z, dim=1)
        assert p.shape == (x.shape[0],)
        return p

    beta_schedule = torch.linspace(0, 1, steps=steps, device=device)
    sampler = AnnealedImportanceSampling(
        dim=dim,
        beta_schedule=beta_schedule,
        log_p1=log_p,
        device=device,
        n_steps_per_sample=n_steps_per_sample,
    )
    logZ_est = sampler.run()["log_Z"]
    sign, logdet = torch.slogdet(cov)
    assert torch.all(sign > 0)
    logZ_true = 0.5 * dim * np.log(2 * np.pi) + 0.5 * logdet
    print(f"logZ est: {logZ_est}")
    print(f"logZ true: {logZ_true}")
    print(f"difference: {logZ_est - logZ_true}")

def test_mixture_of_multinormal(steps, n_steps_per_sample, device):
    dim = 100
    n_mixtures = 10
    mean = torch.normal(mean=0.0, std=50.0, size=(n_mixtures, dim), device=device)
    L = torch.randn(n_mixtures, dim, dim, device=device)
    L = torch.tril(L)
    L.diagonal(dim1=1, dim2=2).abs_().add_(1.0)
    mixture_coeffs = torch.rand(n_mixtures, device=device)
    mixture_coeffs /= mixture_coeffs.sum()
    log_mixture_coeffs = mixture_coeffs.log()
    logdet = 2 * L.diagonal(dim1=1, dim2=2).log().sum(dim=1)
    logZ_true = torch.logsumexp(log_mixture_coeffs + 0.5 * dim * np.log(2 * np.pi) + 0.5 * logdet, dim=0)

    def log_p(x):
        ''' Unnormalized log pdf '''
        d = x.unsqueeze(dim=1) - mean
        z = torch.cholesky_solve(d.unsqueeze(dim=-1), L).squeeze(dim=-1)
        p = -0.5 * torch.sum(d * z, dim=2)
        p += log_mixture_coeffs
        assert p.shape == (x.shape[0], n_mixtures)
        p = torch.logsumexp(p, dim=1)
        return p

    beta_schedule = torch.linspace(0, 1, steps=steps, device=device)
    sampler = AnnealedImportanceSampling(
        dim=dim,
        beta_schedule=beta_schedule,
        log_p1=log_p,
        device=device,
        n_steps_per_sample=n_steps_per_sample,
    )
    logZ_est = sampler.run()["logZ"]
    print(f"logZ est: {logZ_est}")
    print(f"logZ true: {logZ_true}")
    print(f"difference: {logZ_est - logZ_true}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./datasets",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        help="Path to model checkpoint. Should be saved in the format dataset-model_type-otherstuff.pth"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--n-loss-samples",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--n-steps-per-sample",
        type=int,
        default=50,
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # compute_partition_functions(
    #     path=args.path,
    #     steps=args.steps,
    #     n_loss_samples=args.n_loss_samples,
    #     device=device,
    #     data_dir=args.data_dir,
    # )
    test_mixture_of_multinormal(args.steps, args.n_steps_per_sample, device=device)
