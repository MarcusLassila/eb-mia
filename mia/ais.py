import numpy as np
import torch
import time

class AIS_MALA:

    def __init__(self, dim, log_p1, beta_schedule, n_particles=2048, n_steps_per_beta=50, device="cpu"):
        self.dim = dim
        self.log_p1 = log_p1
        self.beta_schedule = beta_schedule
        self.n_particles = n_particles
        self.n_steps_per_beta = n_steps_per_beta
        self.device = torch.device(device)
        self.eps = 0.01
        self.jitter = 1e-9
        self.mass_weight = 0.9

    def log_p0(self, x):
        '''
        Log density of standard multivariate normal distribution.
        '''
        return -0.5 * torch.einsum("bi,bi->b", x, x) - 0.5 * self.dim * np.log(2 * np.pi)

    def log_pt(self, x, beta, return_grad=True):
        '''
        Log density of intermediate distribution.
        The intermediate distribution is a weighted geometric average of the standard normal distribution and the target distribution.
        '''
        if return_grad:
            x = x.detach().requires_grad_()
            log_p = (1 - beta) * self.log_p0(x) + beta * self.log_p1(x)
            grad_x, = torch.autograd.grad(log_p.sum(), x)
            return log_p.detach(), grad_x.detach()
        else:
            log_p = (1 - beta) * self.log_p0(x) + beta * self.log_p1(x)
            return log_p

    def diag_mass(self, x):
        mass = x.var(dim=0, unbiased=True) + self.jitter
        mass = mass.clamp(1e-3, 1e3)
        return mass

    def mala_step(self, x, beta, mass):
        sqrt_mass = mass.sqrt()

        # Compute proposal
        log_p_curr, grad_curr = self.log_pt(x, beta, return_grad=True)
        mu_curr = x + 0.5 * self.eps ** 2 * mass * grad_curr
        sigma = self.eps * sqrt_mass * torch.randn_like(x)
        x_prop = mu_curr + sigma

        log_p_prop, grad_prop = self.log_pt(x_prop, beta, return_grad=True)
        mu_prop = x_prop + 0.5 * self.eps ** 2 * mass * grad_prop

        r_forward = (x_prop - mu_curr) / (self.eps * sqrt_mass)
        r_reverse = (x - mu_prop) / (self.eps * sqrt_mass)
        log_q_diff = 0.5 * (r_forward.pow(2).sum(dim=1) - r_reverse.pow(2).sum(dim=1))
        log_alpha = log_p_prop - log_p_curr + log_q_diff
        accept_mask = torch.rand(self.n_particles, device=self.device).log() < log_alpha
        n_accept = accept_mask.sum().item()
        if n_accept:
            x[accept_mask] = x_prop[accept_mask]
        return n_accept

    def run(self):
        device = self.device
        x = torch.randn(size=(self.n_particles, self.dim), device=device)
        log_w = torch.zeros(self.n_particles, device=device)
        mass = self.diag_mass(x)
        t0 = time.time()
        for step in range(1, len(self.beta_schedule)):
            log_p0_curr = self.log_p0(x)
            log_p1_curr = self.log_p1(x)
            beta_prev = self.beta_schedule[step - 1]
            beta = self.beta_schedule[step]
            log_p_ratio = log_p1_curr - log_p0_curr
            log_w += (beta - beta_prev) * log_p_ratio
            mass = self.mass_weight * mass + (1 - self.mass_weight) * self.diag_mass(x)

            accept_rate = 0.0
            for _ in range(self.n_steps_per_beta):
                accept_rate += self.mala_step(x, beta, mass)

            accept_rate /= self.n_particles * self.n_steps_per_beta
            if accept_rate > 0.6:
                self.eps *= 1.05
            elif accept_rate < 0.42:
                self.eps *= 0.95

            t1 = time.time()
            log_msg = f"step: {step} | beta: {beta:.5f} | eps: {self.eps:.5f} | accept rate: {accept_rate:.5f} | time: {t1 - t0:.1f} "
            print(log_msg, flush=True)

        # log(Z) = log(Z0) + LogSumExp(log_w) - log(n_particles) but log(Z0) = 0 since p0 is already normalized
        log_Z = torch.logsumexp(log_w, dim=0) - np.log(self.n_particles)
        return {
            "logZ": log_Z.item(),
            # Add other quantities of interest here
        }

class AIS_RWM:

    def __init__(self, dim, log_p1, beta_schedule, n_particles=2048, n_steps_per_beta=20, device="cpu"):
        self.dim = dim
        self.log_p1 = log_p1
        self.device = torch.device(device)
        self.beta_schedule = beta_schedule
        self.n_particles = n_particles
        self.n_steps_per_beta = n_steps_per_beta
        self.cov_scale_factor = 2.38 ** 2 / dim

    def log_p0(self, x):
        '''
        Log density of standard multivariate normal distribution.
        '''
        return -0.5 * torch.einsum("bi,bi->b", x, x) - 0.5 * self.dim * np.log(2 * np.pi)

    def log_pt(self, x, beta):
        '''
        Log density of intermediate distribution.
        The intermediate distribution is a weighted geometric average of the standard normal distribution and the target distribution.
        '''
        log_p = (1 - beta) * self.log_p0(x) + beta * self.log_p1(x)
        return log_p

    def random_walk_metropolis_step(self, X, L, beta, log_p0_curr, log_p1_curr):
        z = torch.randn_like(X, device=self.device)
        X_prop = X + z @ L.t()

        log_p0_prop = self.log_p0(X_prop)
        log_p1_prop = self.log_p1(X_prop)

        log_pt_curr = (1 - beta) * log_p0_curr + beta * log_p1_curr
        log_pt_prop = (1 - beta) * log_p0_prop + beta * log_p1_prop
        log_u = torch.rand(self.n_particles, device=self.device).log()
        accept_mask = log_u < (log_pt_prop - log_pt_curr)
        n_accept = accept_mask.sum().item()
        if n_accept > 0:
            X[accept_mask] = X_prop[accept_mask]
            log_p0_curr[accept_mask] = log_p0_prop[accept_mask]
            log_p1_curr[accept_mask] = log_p1_prop[accept_mask]
        return n_accept

    def run(self):
        device = self.device
        x = torch.randn(size=(self.n_particles, self.dim), device=device)
        log_w = torch.zeros(self.n_particles, device=device)
        log_p0_curr = self.log_p0(x)
        log_p1_curr = self.log_p1(x)
        t0 = time.time()
        for step in range(1, len(self.beta_schedule)):
            beta_prev = self.beta_schedule[step - 1]
            beta = self.beta_schedule[step]
            log_p_ratio = log_p1_curr - log_p0_curr
            log_w += (beta - beta_prev) * log_p_ratio

            X_c = x - x.mean(dim=0, keepdim=True)
            cov = X_c.t() @ X_c / (self.n_particles - 1)
            cov = 0.5 * (cov + cov.t())
            S = self.cov_scale_factor * (cov + 1e-5 * torch.eye(self.dim, device=device))
            L = torch.linalg.cholesky(S)

            accept_rate = 0.0
            for _ in range(self.n_steps_per_beta):
                accept_rate += self.random_walk_metropolis_step(x, L, beta, log_p0_curr, log_p1_curr)

            accept_rate /= self.n_particles * self.n_steps_per_beta
            t1 = time.time()
            log_msg = f"step: {step} | beta: {beta:.5f} | accept rate: {accept_rate:.5f} | time: {t1 - t0:.1f} "
            print(log_msg, flush=True)

        # log(Z) = log(Z0) + LogSumExp(log_w) - log(n_particles) but log(Z0) = 0 since p0 is already normalized
        log_Z = torch.logsumexp(log_w, dim=0) - np.log(self.n_particles)
        return {
            "logZ": log_Z.item(),
            # Add other quantities of interest here
        }

def create_beta_schedule(steps, gamma=1.0):
    beta_schedule = torch.linspace(0, 1, steps)
    beta_schedule ** gamma
    return beta_schedule

def test_mixture_of_multinormal(ais_cls, steps, n_steps_per_beta, device):
    dim = 50
    n_mixtures = 2
    mean = torch.normal(mean=1.0, std=3.0, size=(n_mixtures, dim), device=device)
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

    beta_schedule = create_beta_schedule(steps, gamma=0.7)
    sampler = ais_cls(
        dim=dim,
        beta_schedule=beta_schedule,
        log_p1=log_p,
        device=device,
        n_steps_per_beta=n_steps_per_beta,
    )
    logZ_est = sampler.run()["logZ"]
    print(f"logZ est: {logZ_est}")
    print(f"logZ true: {logZ_true}")
    print(f"difference: {logZ_est - logZ_true}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--steps-per-beta",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--mcmc-kernel",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match args.mcmc_kernel:
        case "MALA":
            ais_cls = AIS_MALA
        case "RWM":
            ais_cls = AIS_RWM
        case _:
            raise ValueError("Unsupported MCMC kernel")
    print('Running test...')
    test_mixture_of_multinormal(ais_cls, args.steps, args.steps_per_beta, device=device)
