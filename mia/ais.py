import numpy as np
import torch

class AnnealedImportanceSampling:
    '''Hardcoded Metropolis MCMC kernel for now.'''

    def __init__(self, dim, beta_schedule, log_p1, device, n_samples=64, n_steps_per_sample=32, proposal_std=0.1):
        self.dim = dim
        self.beta_schedule = beta_schedule
        self.n_betas = len(self.beta_schedule)
        self.log_p1 = log_p1
        self.device = device
        self.n_samples = n_samples
        self.n_steps_per_sample = n_steps_per_sample
        self.proposal_std = proposal_std

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

    def run(self):
        device = self.device
        X = torch.randn(size=(self.n_samples, self.dim)).to(device)
        log_p0_curr = self.log_p0(X)
        log_p1_curr = self.log_p1(X)
        log_w = torch.zeros(self.n_samples, device=device)

        for t in range(1, self.n_betas):
            beta_prev = self.beta_schedule[t - 1]
            beta = self.beta_schedule[t]
            log_w += (beta - beta_prev) * (log_p1_curr - log_p0_curr)
            for _ in range(self.n_steps_per_sample):
                eps = torch.randn_like(X, device=device)
                X_prop = X + eps * self.proposal_std

                log_p0_prop = self.log_p0(X_prop)
                log_p1_prop = self.log_p1(X_prop)

                log_pt_curr = (1 - beta) * log_p0_curr + beta * log_p1_curr
                log_pt_prop = (1 - beta) * log_p0_prop + beta * log_p1_prop
                log_u = torch.rand(self.n_samples, device=device).log()
                accept_mask = log_u < (log_pt_prop - log_pt_curr)
                if accept_mask.any():
                    X[accept_mask] = X_prop[accept_mask]
                    log_p0_curr[accept_mask] = log_p0_prop[accept_mask]
                    log_p1_curr[accept_mask] = log_p1_prop[accept_mask]

        # log(Z) = log(Z0) + LogSumExp(log_w) - log(n_samples) but log(Z0) = 0 since p0 is already normalized
        log_Z = torch.logsumexp(log_w) - np.log(self.n_samples)
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
