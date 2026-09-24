from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import norm, t as student_t
from scipy.special import logsumexp, gammaln

ArrayF = npt.NDArray[np.float32]
ArrayI = npt.NDArray[np.int32]

class MIA(ABC):

    @abstractmethod
    def __init__(self, shadow_loss_sigs, shadow_train_mask, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_attack(self, target_loss_sigs: torch.Tensor):
        raise NotImplementedError

class GlobalThreshold:

    def __init__(self, transformation: str = "neg_mean"):
        self.transformation = transformation

    def run_attack(self, target_sigs: torch.Tensor):
        match self.transformation:
            case "neg_mean":
                score = - target_sigs.mean(dim=-1)
            case _:
                raise ValueError(f"Unsupported transformation: {self.transformation}")
        return score

class BASE(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=True,
        prior=0.5,
        apply_sigmoid=True,
    ):
        self.shadow_loss_sigs = shadow_loss_sigs.mean(dim=-1)
        self.shadow_train_mask = shadow_train_mask
        self.offline = offline
        self.prior = prior
        self.apply_sigmoid = apply_sigmoid
        if offline:
            self.shadow_loss_sigs = self.shadow_loss_sigs.masked_fill(shadow_train_mask, torch.inf)
            n_shadow_models = (~shadow_train_mask).to(torch.int32).sum(dim=0)
        else:
            n_shadow_models = torch.tensor([self.shadow_loss_sigs.shape[0]], dtype=torch.int32)
        self.ref = torch.logsumexp(-self.shadow_loss_sigs, dim=0) - torch.log(n_shadow_models)
        self.t_l = np.log(self.prior / (1 - self.prior))

    def run_attack(self, target_loss_sigs):
        score = -target_loss_sigs.mean(dim=-1) - self.ref + self.t_l
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
    ):
        self.shadow_loss_sigs = shadow_loss_sigs.mean(dim=-1)
        self.shadow_train_mask = shadow_train_mask
        self.offline = offline
        self.prior = prior
        self.apply_sigmoid = apply_sigmoid
        if offline:
            self.shadow_loss_sigs = self.shadow_loss_sigs.masked_fill(shadow_train_mask, 0)
            n_shadow_models = (~shadow_train_mask).to(torch.int32).sum(dim=0)
        else:
            n_shadow_models = torch.tensor([self.shadow_loss_sigs.shape[0]], dtype=torch.int32)
        mean = torch.sum(-self.shadow_loss_sigs, dim=0) / n_shadow_models
        if use_global_var:
            var = self.shadow_loss_sigs[~shadow_train_mask].var() if offline else self.shadow_loss_sigs.var()
        else:
            var = (self.shadow_loss_sigs ** 2).sum(dim=0) / n_shadow_models - mean ** 2
        self.ref = mean + 0.5 * var
        self.t_l = np.log(self.prior / (1 - self.prior))

    def run_attack(self, target_loss_sigs):
        score = -target_loss_sigs.mean(dim=-1) - self.ref + self.t_l
        if self.apply_sigmoid:
            score = score.sigmoid()
        return score

class LiRA_alt(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=True,
        use_global_var=True,
        loss_transformation="log",
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
            case "log":
                rescaled_sigs = -torch.log(loss_sigs)
            case "logit_scaling":
                rescaled_sigs = -loss_sigs - torch.log1p(-torch.exp(-loss_sigs))
            case "nll":
                rescaled_sigs = torch.exp(-loss_sigs)
            case "none":
                rescaled_sigs = -loss_sigs # Assuming loss is minimized but larger scores are more likely members
            case _:
                raise ValueError(f"Unavailable transformation {self.loss_transformation}")
        transformed_sigs = rescaled_sigs.mean(dim=-1)
        return transformed_sigs

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

class LiRA_legacy(MIA):

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=False,
        use_global_var=None,
        loss_transformation="none",
    ):
        self.N, self.M, self.K = shadow_loss_sigs.shape
        assert shadow_train_mask.shape == (self.N, self.M)
        self.offline = offline
        self.loss_transformation = loss_transformation
        self.shadow_phi = self._transform_loss_values(shadow_loss_sigs.numpy())
        self.shadow_train_mask = shadow_train_mask.numpy()
        if use_global_var is None:
            in_counts = self.shadow_train_mask.sum(axis=0)
            out_counts = (~self.shadow_train_mask).sum(axis=0)
            if self.offline:
                use_global_var = np.any(out_counts <= 16)
            else:
                use_global_var = np.any((in_counts <= 16) | (out_counts <= 16))
        self.use_global_var = bool(use_global_var)
        if self.offline:
            self.mean_in = None
            self.std = self._membership_class_specific_std(~self.shadow_train_mask)
        else:
            self.mean_in = self._membership_class_specific_mean(self.shadow_train_mask)
            self.std = self._membership_class_agnostic_std()
        self.mean_out = self._membership_class_specific_mean(~self.shadow_train_mask)

    def _transform_loss_values(self, loss_sigs):
        match self.loss_transformation:
            case "log":
                rescaled_sigs = -np.log(loss_sigs)
            case "logit":
                rescaled_sigs = -loss_sigs - np.log1p(-np.exp(-loss_sigs))
            case "nll":
                rescaled_sigs = np.exp(-loss_sigs)
            case "none":
                rescaled_sigs = -loss_sigs # Assuming loss is minimized but larger scores are more likely members
            case _:
                raise ValueError(f"Unavailable transformation {self.loss_transformation}")
        rescaled_and_aggregated_sigs = np.mean(rescaled_sigs, axis=-1)
        return rescaled_and_aggregated_sigs

    def _membership_class_specific_mean(self, mask: ArrayI):
        n = mask.sum(axis=0)
        assert np.all(n >= 2)
        mean = (self.shadow_phi * mask).sum(axis=0) / n
        return mean

    def _membership_class_specific_std(self, mask: ArrayI):
        n = mask.sum(axis=0)
        mean = self._membership_class_specific_mean(mask)
        residue = (self.shadow_phi - mean[None, :]) * mask
        dof = n - 1
        if self.use_global_var:
            var_mean = (residue ** 2).sum() / dof.sum()
        else:
            var_mean = (residue ** 2).sum(axis=0) / dof
        std = np.sqrt(var_mean)
        return std

    def _membership_class_agnostic_std(self):
        in_mask = self.shadow_train_mask
        out_mask = ~in_mask
        n_in = in_mask.sum(axis=0)
        n_out = out_mask.sum(axis=0)
        mean_in = self._membership_class_specific_mean(in_mask)
        mean_out = self._membership_class_specific_mean(out_mask)
        residue_in = (self.shadow_phi - mean_in[None, :]) * in_mask
        residue_out = (self.shadow_phi  - mean_out[None, :]) * out_mask
        dof = n_in + n_out - 2
        if self.use_global_var:
            var_mean = ((residue_in ** 2).sum() + (residue_out ** 2).sum()) / dof.sum()
        else:
            var_mean = ((residue_in ** 2).sum(axis=0) + (residue_out ** 2).sum(axis=0)) / dof
        std = np.sqrt(var_mean)
        return std

    def run_attack(self, target_loss_sigs):
        phi = self._transform_loss_values(target_loss_sigs.numpy())
        if self.offline:
            score = norm.logcdf(
                phi,
                loc=self.mean_out,
                scale=self.std,
            )
            score = torch.tensor(score)
        else:
            logp_in = norm.logpdf(
                phi,
                loc=self.mean_in,
                scale=self.std,
            )
            logp_out = norm.logpdf(
                phi,
                loc=self.mean_out,
                scale=self.std,
            )
            score = torch.tensor(logp_in - logp_out)
        return score

class LiRA(MIA):
    '''Fit an online Gaussian likelihood ratio from shadow losses and membership masks.'''

    def __init__(
        self,
        shadow_loss_sigs,
        shadow_train_mask,
        offline=False,
        use_global_var=True,
        share_variance=True,
        loss_transformation="none",
    ):
        '''Estimate IN and OUT means and global or local, shared or separate standard deviations.'''
        if offline:
            raise ValueError("LiRA supports online inference only.")
        self.loss_transformation = loss_transformation
        self.use_global_var = use_global_var
        shadow_phi = self._transform_loss_values(shadow_loss_sigs.numpy())
        in_mask = shadow_train_mask.numpy()
        if in_mask.shape != shadow_phi.shape:
            raise ValueError("Shadow membership mask must match shadow sample dimensions.")
        out_mask = ~in_mask
        n_in = in_mask.sum(axis=0)
        n_out = out_mask.sum(axis=0)
        if np.any(n_in < 2) or np.any(n_out < 2):
            raise ValueError("LiRA requires at least two IN and OUT shadow models per sample.")
        self.mean_in = (shadow_phi * in_mask).sum(axis=0) / n_in
        self.mean_out = (shadow_phi * out_mask).sum(axis=0) / n_out
        residue_in = (shadow_phi - self.mean_in[None, :]) * in_mask
        residue_out = (shadow_phi - self.mean_out[None, :]) * out_mask
        sum_squares_in = (residue_in ** 2).sum(axis=0)
        sum_squares_out = (residue_out ** 2).sum(axis=0)
        dof_in = n_in - 1
        dof_out = n_out - 1
        if share_variance:
            sum_squares = sum_squares_in + sum_squares_out
            dof = dof_in + dof_out
            if use_global_var:
                variance = sum_squares.sum() / dof.sum()
            else:
                variance = sum_squares / dof
            self.std_in = np.sqrt(variance)
            self.std_out = self.std_in
        else:
            if use_global_var:
                variance_in = sum_squares_in.sum() / dof_in.sum()
                variance_out = sum_squares_out.sum() / dof_out.sum()
            else:
                variance_in = sum_squares_in / dof_in
                variance_out = sum_squares_out / dof_out
            self.std_in = np.sqrt(variance_in)
            self.std_out = np.sqrt(variance_out)

    def _transform_loss_values(self, loss_sigs):
        '''Transform query losses and return one mean score per model and sample.'''
        match self.loss_transformation:
            case "log":
                scores = -np.log(loss_sigs)
            case "logit":
                scores = -loss_sigs - np.log1p(-np.exp(-loss_sigs))
            case "nll":
                scores = np.exp(-loss_sigs)
            case "none":
                scores = -loss_sigs
            case _:
                raise ValueError(f"Unavailable transformation {self.loss_transformation}")
        return scores.mean(axis=-1)

    def run_attack(self, target_loss_sigs):
        '''Return the IN versus OUT log density ratio for each target sample.'''
        target_mean = self._transform_loss_values(target_loss_sigs.numpy())
        logp_in = norm.logpdf(target_mean, loc=self.mean_in, scale=self.std_in)
        logp_out = norm.logpdf(target_mean, loc=self.mean_out, scale=self.std_out)
        return torch.tensor(logp_in - logp_out)

class NormalPriorLiRA(MIA):
    '''
    y_1,...,y_n | mu, sigma ~ N(mu, sigma^2)
    mu | m, tau ~ N(m, tau^2)
    (y_1,...,y_n) | m, sigma, tau ~ N(m 1, sigma^2 I + tau^2 1 1^T)
    '''

    def __init__(
        self,
        shadow_loss_sigs: torch.Tensor,
        shadow_train_mask: torch.Tensor,
        offline=False,
        loss_transformation="log",
    ):
        M, _, K = shadow_loss_sigs.shape
        assert M >= 4, "Should have at least 4 shadow models"
        assert K > 1, "Need more than 1 loss sample per model and data point"
        self.M, self.K = M, K
        self.loss_transformation = loss_transformation
        self.shadow_loss_sigs = self._loss_transformation(shadow_loss_sigs.numpy())
        self.shadow_train_mask = shadow_train_mask.numpy()
        self.offline = offline
        self.sigma2, self.tau2 = self._membership_class_agnostic_statistic() 
        if self.offline:
            self.mean_in = None
        else:
            self.mean_in = self._membership_class_specific_statistic(self.shadow_train_mask)
        self.mean_out = self._membership_class_specific_statistic(~self.shadow_train_mask)

    def _loss_transformation(self, loss_sigs: ArrayF):
        match self.loss_transformation:
            case "log":
                return -np.log(loss_sigs)
            case "none":
                return -loss_sigs
            case _:
                raise ValueError(f"Unknown loss transformation: {self.loss_transformation}")

    def _membership_class_agnostic_statistic(self):
        x = self.shadow_loss_sigs
        xbar = x.mean(axis=-1)
        sample_var = x.var(axis=-1, ddof=1)

        sigma2 = np.maximum(sample_var.mean(axis=0), 1e-11)

        in_mask = self.shadow_train_mask
        out_mask = ~self.shadow_train_mask

        n_in = in_mask.sum(axis=0)
        n_out = out_mask.sum(axis=0)
        assert np.all(n_in >= 2)
        assert np.all(n_out >= 2)

        mean_in = (xbar * in_mask).sum(axis=0) / n_in
        mean_out = (xbar * out_mask).sum(axis=0) / n_out

        resid_in = (xbar - mean_in[None, :]) * in_mask
        resid_out = (xbar - mean_out[None, :]) * out_mask

        dof = (n_in - 1) + (n_out - 1)
        var_xbar = ((resid_in ** 2).sum(axis=0) + (resid_out ** 2).sum(axis=0)) / dof

        tau2 = np.maximum(var_xbar - sigma2 / self.K, 0.0)
        return sigma2, tau2

    def _membership_class_specific_statistic(self, mask: ArrayI):
        n = mask.sum(axis=0)
        assert np.all(n >= 2), "Should have at least 2 in (or out) shadow models"
        x = self.shadow_loss_sigs
        mean = x.mean(axis=-1) * mask
        mean_mean = mean.sum(axis=0) / n
        return mean_mean

    def _logpdf(self, x: ArrayF, mean: ArrayF, sigma2: ArrayF, tau2: ArrayF):
        z = x - mean[:, None]
        sum_z = z.sum(axis=1)
        r = tau2 * self.K + sigma2
        logdet = (self.K - 1) * np.log(sigma2) + np.log(r)
        quad = (z ** 2).sum(axis=1) / sigma2 - (sum_z ** 2) * tau2 / (sigma2 * r)
        return -0.5 * (self.K * np.log(2.0 * np.pi) + logdet + quad)

    def run_attack(self, target_loss_sigs):
        x = self._loss_transformation(target_loss_sigs.numpy())
        if self.offline:
            raise ValueError("Offline mode not supported yet!")
        else:
            logp_in = self._logpdf(x=x, mean=self.mean_in, sigma2=self.sigma2, tau2=self.tau2)
            logp_out = self._logpdf(x=x, mean=self.mean_out, sigma2=self.sigma2, tau2=self.tau2)
            score = logp_in - logp_out
        return torch.tensor(score)

class HG_LiRA(MIA):

    @dataclass
    class Prior:
        xi: float
        lam: float
        a_beta: float
        b_beta: float
        a_tau: float
        b_tau: float

    def __init__(
        self,
        ref_sigs: torch.Tensor,
        ref_train_mask: torch.Tensor,
        offline=False,
        sig_transformation="log",
        n_gibbs_samples=64,
        n_gibbs_warmup=64,
        n_quadrature=50,
    ):
        self.n_ref, self.n_data, self.n_sigs = ref_sigs.shape
        self.sig_transformation = sig_transformation
        ref_sigs = self._sig_transformation(ref_sigs.numpy())
        self.ref_train_mask = ref_train_mask.numpy()
        assert self.n_ref >= 4, "Need at least 2 reference models per membership class for variance estimation"
        assert self.n_ref % 2 == 0, "Need an even number of reference models"
        assert np.all(self.ref_train_mask.sum(axis=0) == self.n_ref // 2), "Must have M/2 reference model per membership class for each data point"
        self.ref_sigs_in = self._extract_class_specific_ref_sigs(ref_sigs, self.ref_train_mask)
        self.ref_sigs_out = self._extract_class_specific_ref_sigs(ref_sigs, ~self.ref_train_mask)
        self.offline = offline
        self.n_gibbs_samples = n_gibbs_samples
        self.n_gibbs_warmup = n_gibbs_warmup
        self.n_quadrature = n_quadrature
        self.eps = 1e-9
        self.eta_in = self.estimate_priors(self.ref_sigs_in)
        self.eta_out = self.estimate_priors(self.ref_sigs_out)
        self.rng = np.random.default_rng(seed=42)
        self.Ms_in, self.Vs_in = self.gibbs_sampler(self.ref_sigs_in, self.eta_in)
        self.Ms_out, self.Vs_out = self.gibbs_sampler(self.ref_sigs_out, self.eta_out)

    def _extract_class_specific_ref_sigs(self, ref_sigs: ArrayF, mask: ArrayI):
        return ref_sigs.transpose(1, 0, 2)[mask.T].reshape(self.n_data, self.n_ref // 2, self.n_sigs)

    def _sig_transformation(self, sigs: ArrayF):
        match self.sig_transformation:
            case "log":
                return -np.log(sigs)
            case "none":
                return -sigs
            case _:
                raise ValueError(f"Unknown loss transformation: {self.sig_transformation}")

    def estimate_priors(self, y: ArrayF):
        M = self.n_ref // 2
        n = self.n_sigs
        y_mean = y.mean(axis=2)
        y_var = y.var(axis=2, ddof=1)
        y_mean_mean = y_mean.mean(axis=1)
        y_mean_var = y_var.mean(axis=1)
        y_var_mean = y_mean.var(axis=1)
        u = y_var_mean - y_mean_var / n
        q_beta1 = np.maximum(y_mean_var, self.eps).mean()
        q_tau1 = np.maximum(u, self.eps).mean()
        xi = y_mean_mean.mean()

        # Temporary simplified shape-rate prior:
        a_beta = 2.2
        a_tau = 2.2
        b_beta = q_beta1 * (a_beta - 1)
        b_tau = q_tau1 * (a_tau - 1)

        inv_lam = y_mean_mean.var(ddof=1) - q_tau1 / M - q_beta1 / (n * M)
        lam = 1 / max(inv_lam, self.eps)

        return self.Prior(
            xi=xi,
            lam=lam,
            a_beta=a_beta,
            b_beta=b_beta,
            a_tau=a_tau,
            b_tau=b_tau,
        )

    def _initialize_latent_variables(self, y_mean, S):
        mu = y_mean.copy()
        inner_var = S / (self.n_sigs - 1)
        beta = 1 / np.maximum(inner_var, self.eps)
        nu = y_mean.mean(axis=1)
        tau_inv = y_mean.var(axis=1, ddof=1) - inner_var.mean(axis=1) / self.n_sigs
        tau = 1 / np.maximum(tau_inv, self.eps)
        return mu, beta, nu, tau

    def gibbs_sampler(self, y: ArrayF, eta: Prior):
        y_mean = y.mean(axis=2)
        S = np.sum((y - y_mean[:, :, None]) ** 2, axis=2)
        M, _, n = self.n_ref // 2, self.n_data, self.n_sigs
        xi, lam, a_beta, b_beta, a_tau, b_tau = eta.xi, eta.lam, eta.a_beta, eta.b_beta, eta.a_tau, eta.b_tau

        mu, beta, nu, tau = self._initialize_latent_variables(y_mean, S)

        beta_shape = a_beta + 0.5 * n
        tau_shape = a_tau + 0.5 * M

        M_samples = []
        V_samples = []
        for i in range(self.n_gibbs_warmup + self.n_gibbs_samples):
            # Gibbs update:
            mu_prec = n * beta + tau[:, None]
            mu_center = (n * beta * y_mean + tau[:, None] * nu[:, None]) / mu_prec
            mu = self.rng.normal(loc=mu_center, scale=np.sqrt(1 / mu_prec), size=mu.shape)
            beta_rate = b_beta + 0.5 * (S + n * (y_mean - mu) ** 2)
            beta = self.rng.gamma(shape=beta_shape, scale=1/beta_rate, size=beta.shape)
            nu_prec = M * tau + lam
            nu_center = (tau * mu.sum(axis=1) + lam * xi) / nu_prec
            nu = self.rng.normal(loc=nu_center, scale=np.sqrt(1 / nu_prec), size=nu.shape)
            tau_rate = b_tau + 0.5 * ((mu - nu[:, None]) ** 2).sum(axis=1)
            tau = self.rng.gamma(shape=tau_shape, scale=1/tau_rate, size=tau.shape)

            if i >= self.n_gibbs_warmup:
                m = (tau * mu.sum(axis=1) + lam * xi) / (M * tau + lam)
                v = 1 / tau + 1 / (M * tau + lam)
                M_samples.append(m)
                V_samples.append(v)

        return np.array(M_samples), np.array(V_samples)

    def log_g_func(self, y_target, mu, eta: Prior):
        a = eta.a_beta
        b = eta.b_beta
        n = self.n_sigs

        y_bar = y_target.mean(axis=-1)
        S = ((y_target - y_bar[:, None]) ** 2).sum(axis=-1)
        y_bar = y_bar.reshape((1,) * (mu.ndim - 1) + y_bar.shape)
        S = S.reshape((1,) * (mu.ndim - 1) + S.shape)

        out = gammaln(a + 0.5 * n)
        out -= gammaln(a)
        out += a * np.log(b)
        out -= 0.5 * n * np.log(2 * np.pi)
        out -= (a + 0.5 * n) * np.log(b + 0.5 * (S + n * (y_bar - mu) ** 2))
        return out

    def log_posterior(self, y_target, Ms, Vs, eta: Prior):
        log_post = []
        # Compute it in chunks to not overuse memory
        chunk_size = 10_000_000 // (self.n_quadrature * self.n_gibbs_samples)
        chunk_size = max(chunk_size, 1)
        for start in range(0, self.n_data, chunk_size):
            stop = min(start + chunk_size, self.n_data)
            y_chunk = y_target[start: stop]
            Ms_chunk = Ms[:, start: stop]
            Vs_chunk = Vs[:, start: stop]
            h_chunk = self.h_func(y_chunk, Ms_chunk, Vs_chunk, eta)
            log_post_chunk = logsumexp(h_chunk, axis=0) - np.log(self.n_gibbs_samples)
            log_post.append(log_post_chunk)
        return np.concatenate(log_post)

    def h_func(self, y_target, Ms, Vs, eta: Prior):
        t, w = np.polynomial.hermite.hermgauss(self.n_quadrature)
        mu_quad = Ms[None, :, :] + np.sqrt(2 * Vs)[None, :, :] * t[:, None, None]
        log_w = np.log(w)[:, None, None]
        return logsumexp(log_w + self.log_g_func(y_target, mu_quad, eta), axis=0) - 0.5 * np.log(np.pi)

    def run_attack(self, target_loss_sigs):
        y_target = self._sig_transformation(target_loss_sigs.numpy())
        assert y_target.shape == (self.n_data, self.n_sigs)
        score = None
        if self.offline:
            raise NotImplementedError
        else:
            log_posterior_in = self.log_posterior(y_target, self.Ms_in, self.Vs_in, self.eta_in)
            log_posterior_out = self.log_posterior(y_target, self.Ms_out, self.Vs_out, self.eta_out)
            score = log_posterior_in - log_posterior_out
        return torch.tensor(score)

class HG_LiRA_r(MIA):
    '''
    Fit HG_LiRA with center-dependent local dispersion and first-moment priors.
    Inputs and outputs match HG_LiRA.
    '''

    @dataclass
    class Prior:
        xi: float
        lam: float
        a_beta: float
        b_beta: float
        a_tau: float
        b_tau: float
        r: ArrayF

    def __init__(
        self,
        ref_sigs: torch.Tensor,
        ref_train_mask: torch.Tensor,
        offline=False,
        sig_transformation="log",
        n_gibbs_samples=64,
        n_gibbs_warmup=64,
        n_quadrature=50,
    ):
        self.n_ref, self.n_data, self.n_sigs = ref_sigs.shape
        self.sig_transformation = sig_transformation
        ref_sigs = self._sig_transformation(ref_sigs.numpy())
        self.ref_train_mask = ref_train_mask.numpy()
        assert self.n_ref >= 4, "Need at least 2 reference models per membership class for variance estimation"
        assert self.n_ref % 2 == 0, "Need an even number of reference models"
        assert np.all(self.ref_train_mask.sum(axis=0) == self.n_ref // 2), "Must have M/2 reference model per membership class for each data point"
        self.ref_sigs_in = self._extract_class_specific_ref_sigs(
            ref_sigs,
            self.ref_train_mask,
        )
        self.ref_sigs_out = self._extract_class_specific_ref_sigs(
            ref_sigs,
            ~self.ref_train_mask,
        )
        self.offline = offline
        self.n_gibbs_samples = n_gibbs_samples
        self.n_gibbs_warmup = n_gibbs_warmup
        self.n_quadrature = n_quadrature
        self.eps = 1e-9
        self.eta_in = None
        if not self.offline:
            self.eta_in = self.estimate_priors(self.ref_sigs_in)
        self.eta_out = self.estimate_priors(self.ref_sigs_out)
        self.rng = np.random.default_rng(seed=42)
        self.Ms_in = None
        self.Vs_in = None
        if not self.offline:
            self.Ms_in, self.Vs_in = self.gibbs_sampler(
                self.ref_sigs_in,
                self.eta_in,
            )
        self.Ms_out, self.Vs_out = self.gibbs_sampler(
            self.ref_sigs_out,
            self.eta_out,
        )

    def _extract_class_specific_ref_sigs(self, ref_sigs: ArrayF, mask: ArrayI):
        return ref_sigs.transpose(1, 0, 2)[mask.T].reshape(
            self.n_data,
            self.n_ref // 2,
            self.n_sigs,
        )

    def _sig_transformation(self, sigs: ArrayF):
        match self.sig_transformation:
            case "log":
                return -np.log(sigs)
            case "none":
                return -sigs
            case _:
                raise ValueError(
                    f"Unknown loss transformation: {self.sig_transformation}"
                )

    def _dispersion_scale(self, tau_inverse, y_center):
        '''
        Estimate mean-one center-dependent dispersion scales.
        Inputs are local variance estimates and local centers; output is r_i.
        '''
        positive_tau_inverse = np.maximum(tau_inverse, self.eps)
        log_tau_inverse = np.log(positive_tau_inverse)
        centered_log_scale = log_tau_inverse - np.mean(log_tau_inverse)
        center_scale = max(float(np.std(y_center, ddof=1)), np.sqrt(self.eps))
        standardized_center = (y_center - np.mean(y_center)) / center_scale
        slope_numerator = np.sum(standardized_center * centered_log_scale)
        slope_denominator = np.sum(standardized_center ** 2)
        slope_denominator = max(float(slope_denominator), self.eps)
        slope = slope_numerator / slope_denominator
        r = np.exp(slope * standardized_center)
        r /= np.mean(r)
        return r

    def estimate_priors(self, y: ArrayF):
        '''
        Estimate first-moment priors after removing center-dependent dispersion.
        Input is class-specific reference signals; output is a Prior.
        '''
        M = self.n_ref // 2
        n = self.n_sigs
        y_mean = y.mean(axis=2)
        y_var = y.var(axis=2, ddof=1)
        y_mean_mean = y_mean.mean(axis=1)
        y_mean_var = y_var.mean(axis=1)
        y_var_mean = y_mean.var(axis=1)
        tau_inverse = y_var_mean - y_mean_var / n
        r = self._dispersion_scale(tau_inverse, y_mean_mean)
        adjusted_tau_inverse = tau_inverse / r
        q_beta1 = np.maximum(y_mean_var, self.eps).mean()
        q_tau1 = np.maximum(adjusted_tau_inverse, self.eps).mean()
        xi = y_mean_mean.mean()
        a_beta = 2.2
        a_tau = 2.2
        b_beta = q_beta1 * (a_beta - 1)
        b_tau = q_tau1 * (a_tau - 1)
        inv_lam = y_mean_mean.var(ddof=1) - q_tau1 / M - q_beta1 / (n * M)
        lam = 1 / max(inv_lam, self.eps)
        return self.Prior(
            xi=xi,
            lam=lam,
            a_beta=a_beta,
            b_beta=b_beta,
            a_tau=a_tau,
            b_tau=b_tau,
            r=r,
        )

    def _initialize_latent_variables(self, y_mean, S, r):
        '''
        Initialize latent variables with r-adjusted between-model precision.
        Inputs are model means, sums of squares, and r_i; outputs Gibbs state.
        '''
        mu = y_mean.copy()
        inner_var = S / (self.n_sigs - 1)
        beta = 1 / np.maximum(inner_var, self.eps)
        nu = y_mean.mean(axis=1)
        tau_inv = y_mean.var(axis=1, ddof=1) - inner_var.mean(axis=1) / self.n_sigs
        tau_inv = tau_inv / r
        tau = 1 / np.maximum(tau_inv, self.eps)
        return mu, beta, nu, tau

    def gibbs_sampler(self, y: ArrayF, eta: Prior):
        '''
        Run Gibbs sampling with local precision tau_i / r_i.
        Input is class data and prior; output is posterior mean and variance samples.
        '''
        y_mean = y.mean(axis=2)
        S = np.sum((y - y_mean[:, :, None]) ** 2, axis=2)
        M, _, n = self.n_ref // 2, self.n_data, self.n_sigs
        xi, lam, a_beta, b_beta, a_tau, b_tau = eta.xi, eta.lam, eta.a_beta, eta.b_beta, eta.a_tau, eta.b_tau
        r = eta.r

        mu, beta, nu, tau = self._initialize_latent_variables(y_mean, S, r)

        beta_shape = a_beta + 0.5 * n
        tau_shape = a_tau + 0.5 * M

        M_samples = []
        V_samples = []
        for i in range(self.n_gibbs_warmup + self.n_gibbs_samples):
            local_precision = tau / r
            mu_prec = n * beta + local_precision[:, None]
            mu_center = (n * beta * y_mean + local_precision[:, None] * nu[:, None]) / mu_prec
            mu = self.rng.normal(loc=mu_center, scale=np.sqrt(1 / mu_prec), size=mu.shape)
            beta_rate = b_beta + 0.5 * (S + n * (y_mean - mu) ** 2)
            beta = self.rng.gamma(shape=beta_shape, scale=1 / beta_rate, size=beta.shape)
            nu_prec = M * local_precision + lam
            nu_center = (local_precision * mu.sum(axis=1) + lam * xi) / nu_prec
            nu = self.rng.normal(loc=nu_center, scale=np.sqrt(1 / nu_prec), size=nu.shape)
            tau_residue = ((mu - nu[:, None]) ** 2).sum(axis=1)
            tau_rate = b_tau + 0.5 * tau_residue / r
            tau = self.rng.gamma(shape=tau_shape, scale=1 / tau_rate, size=tau.shape)

            if i >= self.n_gibbs_warmup:
                local_precision = tau / r
                m = (local_precision * mu.sum(axis=1) + lam * xi) / (M * local_precision + lam)
                v = r / tau + 1 / (M * local_precision + lam)
                M_samples.append(m)
                V_samples.append(v)

        return np.array(M_samples), np.array(V_samples)

    def log_g_func(self, y_target, mu, eta: Prior):
        a = eta.a_beta
        b = eta.b_beta
        n = self.n_sigs

        y_bar = y_target.mean(axis=-1)
        S = ((y_target - y_bar[:, None]) ** 2).sum(axis=-1)
        y_bar = y_bar.reshape((1,) * (mu.ndim - 1) + y_bar.shape)
        S = S.reshape((1,) * (mu.ndim - 1) + S.shape)

        out = gammaln(a + 0.5 * n)
        out -= gammaln(a)
        out += a * np.log(b)
        out -= 0.5 * n * np.log(2 * np.pi)
        out -= (a + 0.5 * n) * np.log(
            b + 0.5 * (S + n * (y_bar - mu) ** 2)
        )
        return out

    def log_posterior(self, y_target, Ms, Vs, eta: Prior):
        log_post = []
        chunk_size = 10_000_000 // (
            self.n_quadrature * self.n_gibbs_samples
        )
        chunk_size = max(chunk_size, 1)
        for start in range(0, self.n_data, chunk_size):
            stop = min(start + chunk_size, self.n_data)
            y_chunk = y_target[start:stop]
            Ms_chunk = Ms[:, start:stop]
            Vs_chunk = Vs[:, start:stop]
            h_chunk = self.h_func(y_chunk, Ms_chunk, Vs_chunk, eta)
            log_post_chunk = logsumexp(h_chunk, axis=0)
            log_post_chunk -= np.log(self.n_gibbs_samples)
            log_post.append(log_post_chunk)
        return np.concatenate(log_post)

    def h_func(self, y_target, Ms, Vs, eta: Prior):
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quadrature)
        mu_quad = Ms[None, :, :]
        mu_quad = mu_quad + (
            np.sqrt(2.0 * Vs)[None, :, :] * nodes[:, None, None]
        )
        log_weights = np.log(weights)[:, None, None]
        quadrature_log_density = logsumexp(
            log_weights + self.log_g_func(y_target, mu_quad, eta),
            axis=0,
        )
        quadrature_log_density -= 0.5 * np.log(np.pi)
        return quadrature_log_density

    def log_posterior_mean_cdf(self, y_target, Ms, Vs, eta: Prior):
        '''
        Integrate target-mean lower-tail probabilities over Gibbs states.
        Inputs are target signals, predictive means and variances, and the
        out-class prior; output is one log-CDF score per data point.
        '''
        target_mean = y_target.mean(axis=1)
        degrees_freedom = 2.0 * eta.a_beta
        mean_scale = np.sqrt(
            eta.b_beta / (eta.a_beta * self.n_sigs)
        )
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quadrature)
        log_weights = np.log(weights)[:, None, None]
        chunk_size = 10_000_000 // (
            self.n_quadrature * self.n_gibbs_samples
        )
        chunk_size = max(chunk_size, 1)
        scores = []
        for start in range(0, self.n_data, chunk_size):
            stop = min(start + chunk_size, self.n_data)
            target_chunk = target_mean[start:stop]
            mean_chunk = Ms[:, start:stop]
            variance_chunk = Vs[:, start:stop]
            quadrature_mean = mean_chunk[None, :, :]
            quadrature_mean = quadrature_mean + (
                np.sqrt(2.0 * variance_chunk)[None, :, :]
                * nodes[:, None, None]
            )
            log_cdf = student_t.logcdf(
                target_chunk[None, None, :],
                df=degrees_freedom,
                loc=quadrature_mean,
                scale=mean_scale,
            )
            state_log_cdf = logsumexp(
                log_weights + log_cdf,
                axis=0,
            )
            state_log_cdf -= 0.5 * np.log(np.pi)
            chunk_score = logsumexp(state_log_cdf, axis=0)
            chunk_score -= np.log(self.n_gibbs_samples)
            scores.append(chunk_score)
        return np.concatenate(scores)

    def run_attack(self, target_loss_sigs):
        y_target = self._sig_transformation(target_loss_sigs.numpy())
        assert y_target.shape == (self.n_data, self.n_sigs)
        if self.offline:
            score = self.log_posterior_mean_cdf(
                y_target,
                self.Ms_out,
                self.Vs_out,
                self.eta_out,
            )
            return torch.tensor(score, dtype=torch.float32)
        log_posterior_in = self.log_posterior(
            y_target,
            self.Ms_in,
            self.Vs_in,
            self.eta_in,
        )
        log_posterior_out = self.log_posterior(
            y_target,
            self.Ms_out,
            self.Vs_out,
            self.eta_out,
        )
        score = log_posterior_in - log_posterior_out
        return torch.tensor(score)

class HG_LiRA_2(HG_LiRA):
    '''
    Fit HG_LiRA with second-moment prior estimates and no r_i covariate.
    Inputs and outputs match HG_LiRA.
    '''

    def _gamma_from_inverse_moments(self, first_moment, second_moment):
        '''
        Convert inverse-precision moments into Gamma shape and rate.
        Inputs are first and second inverse moments; output is shape-rate pair.
        '''
        first_moment = max(float(first_moment), self.eps)
        second_moment = max(float(second_moment), first_moment ** 2 + self.eps)
        denominator = second_moment - first_moment ** 2
        if denominator <= self.eps:
            shape = 1e6
        else:
            shape = min(2.0 + first_moment ** 2 / denominator, 1e6)
        rate = max((shape - 1.0) * first_moment, self.eps)
        return float(shape), float(rate)

    def estimate_priors(self, y: ArrayF):
        '''
        Estimate second-moment priors without center-dependent dispersion.
        Input is class-specific reference signals; output is a Prior.
        '''
        M = self.n_ref // 2
        n = self.n_sigs
        y_mean = y.mean(axis=2)
        y_var = y.var(axis=2, ddof=1)
        y_center = y_mean.mean(axis=1)
        beta_var = y_var.reshape(-1)
        beta_first = float(np.mean(beta_var))
        beta_second_terms = (n - 1.0) * beta_var ** 2 / (n + 1.0)
        beta_second = float(np.mean(beta_second_terms))
        a_beta, b_beta = self._gamma_from_inverse_moments(beta_first, beta_second)
        model_var = y_mean.var(axis=1, ddof=1)
        inner_mean_var = y_var.mean(axis=1) / n
        tau_inverse = model_var - inner_mean_var
        tau_first = float(np.mean(tau_inverse))
        tau_second_observed = float(np.mean(tau_inverse ** 2))
        tau_cross_correction = 4.0 * tau_first * beta_first
        tau_cross_correction /= n * (M - 1.0)
        beta_second_correction = 2.0 * beta_second
        beta_second_correction /= M * n * (n - 1.0)
        beta_first_correction = 2.0 * beta_first ** 2
        beta_first_correction /= M * (M - 1.0) * n ** 2
        tau_second = tau_second_observed
        tau_second -= tau_cross_correction
        tau_second -= beta_second_correction
        tau_second -= beta_first_correction
        tau_second *= (M - 1.0) / (M + 1.0)
        a_tau, b_tau = self._gamma_from_inverse_moments(tau_first, tau_second)
        xi = y_center.mean()
        inv_lam = y_center.var(ddof=1) - tau_first / M - beta_first / (n * M)
        lam = 1 / max(inv_lam, self.eps)
        return self.Prior(
            xi=xi,
            lam=lam,
            a_beta=a_beta,
            b_beta=b_beta,
            a_tau=a_tau,
            b_tau=b_tau,
        )
