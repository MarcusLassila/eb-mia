from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import norm
from scipy.special import gammaln, logsumexp, stdtr

ArrayF = npt.NDArray[np.float32]
ArrayI = npt.NDArray[np.int32]

def _gamma_from_inverse_moments(
    first_moment,
    second_moment,
    eps,
    fallback_shape=2.05,
):
    '''
    Convert inverse-precision moments into a finite Gamma shape and rate.
    Inputs are first and second inverse moments and a floor; output is a shape-rate pair.
    '''
    first_moment = float(first_moment)
    second_moment = float(second_moment)
    valid_first_moment = np.isfinite(first_moment) and first_moment > eps
    if not valid_first_moment:
        first_moment = eps
    moment_variance = second_moment - first_moment ** 2
    moment_scale = max(abs(second_moment), first_moment ** 2, eps ** 2)
    tolerance = 32.0 * np.finfo(float).eps * moment_scale
    valid_moment_variance = np.isfinite(moment_variance)
    valid_moment_variance &= moment_variance > tolerance
    if valid_first_moment and valid_moment_variance:
        shape = 2.0 + first_moment ** 2 / moment_variance
        shape = min(shape, 1e6)
    else:
        shape = fallback_shape
    rate = max((shape - 1.0) * first_moment, eps)
    return float(shape), float(rate)

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
        apply_sigmoid=False,
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
        apply_sigmoid=False,
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
        loss_transformation="none",
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
    '''Fit Gaussian membership scores from shadow losses and membership masks.'''

    def __init__(
        self,
        shadow_loss_sigs,
        shadow_train_mask,
        offline=False,
        use_global_var=True,
        share_variance=True,
        loss_transformation="none",
    ):
        '''Estimate reference means and selected standard deviations for online or offline scoring.'''
        self.offline = offline
        self.loss_transformation = loss_transformation
        self.use_global_var = use_global_var
        shadow_phi = self._transform_loss_values(shadow_loss_sigs.numpy())
        in_mask = shadow_train_mask.numpy()
        if in_mask.shape != shadow_phi.shape:
            raise ValueError("Shadow membership mask must match shadow sample dimensions.")
        out_mask = ~in_mask
        n_out = out_mask.sum(axis=0)
        if np.any(n_out < 2):
            raise ValueError("LiRA requires at least two OUT shadow models per sample.")
        self.mean_out = (shadow_phi * out_mask).sum(axis=0) / n_out
        residue_out = (shadow_phi - self.mean_out[None, :]) * out_mask
        sum_squares_out = (residue_out ** 2).sum(axis=0)
        dof_out = n_out - 1
        if offline:
            if use_global_var:
                variance_out = sum_squares_out.sum() / dof_out.sum()
            else:
                variance_out = sum_squares_out / dof_out
            self.mean_in = None
            self.std_in = None
            self.std_out = np.sqrt(variance_out)
            return
        n_in = in_mask.sum(axis=0)
        if np.any(n_in < 2):
            raise ValueError("LiRA requires at least two IN shadow models per sample.")
        self.mean_in = (shadow_phi * in_mask).sum(axis=0) / n_in
        residue_in = (shadow_phi - self.mean_in[None, :]) * in_mask
        sum_squares_in = (residue_in ** 2).sum(axis=0)
        dof_in = n_in - 1
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
        '''Return an OUT-tail score offline or an IN versus OUT density ratio online.'''
        target_mean = self._transform_loss_values(target_loss_sigs.numpy())
        if self.offline:
            out_tail = norm.logcdf(target_mean, loc=self.mean_out, scale=self.std_out)
            return torch.tensor(out_tail)
        logp_in = norm.logpdf(target_mean, loc=self.mean_in, scale=self.std_in)
        logp_out = norm.logpdf(target_mean, loc=self.mean_out, scale=self.std_out)
        return torch.tensor(logp_in - logp_out)

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
        sig_transformation="none",
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
    Fit HG_LiRA with center-dependent dispersion and second-moment priors.
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
        sig_transformation="none",
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
        Estimate second-moment priors after removing center-dependent dispersion.
        Input is class-specific reference signals; output is a Prior.
        '''
        M = self.n_ref // 2
        n = self.n_sigs
        y_mean = y.mean(axis=2)
        y_var = y.var(axis=2, ddof=1)
        y_mean_mean = y_mean.mean(axis=1)
        y_mean_var = y_var.mean(axis=1)
        y_var_mean = y_mean.var(axis=1)
        beta_var = y_var.reshape(-1)
        beta_first = float(np.mean(beta_var))
        beta_second_terms = (n - 1.0) * beta_var ** 2 / (n + 1.0)
        beta_second = float(np.mean(beta_second_terms))
        a_beta, b_beta = _gamma_from_inverse_moments(
            beta_first,
            beta_second,
            self.eps,
        )
        tau_inverse = y_var_mean - y_mean_var / n
        r = self._dispersion_scale(tau_inverse, y_mean_mean)
        adjusted_tau_inverse = tau_inverse / r
        tau_first = float(np.mean(adjusted_tau_inverse))
        tau_second_observed = float(np.mean(adjusted_tau_inverse ** 2))
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
        a_tau, b_tau = _gamma_from_inverse_moments(
            tau_first,
            tau_second,
            self.eps,
        )
        xi = y_mean_mean.mean()
        inv_lam = y_mean_mean.var(ddof=1)
        inv_lam -= tau_first / M
        inv_lam -= beta_first / (n * M)
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
            standardized_target = target_chunk[None, None, :]
            standardized_target = standardized_target - quadrature_mean
            standardized_target /= mean_scale
            log_cdf = np.empty_like(standardized_target)
            lower_tail = standardized_target < 0.0
            lower_probability = stdtr(
                degrees_freedom,
                standardized_target[lower_tail],
            )
            log_cdf[lower_tail] = np.log(lower_probability)
            upper_tail = ~lower_tail
            upper_probability = stdtr(
                degrees_freedom,
                -standardized_target[upper_tail],
            )
            log_cdf[upper_tail] = np.log1p(-upper_probability)
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
        a_beta, b_beta = _gamma_from_inverse_moments(
            beta_first,
            beta_second,
            self.eps,
        )
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
        a_tau, b_tau = _gamma_from_inverse_moments(
            tau_first,
            tau_second,
            self.eps,
        )
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

class HG_LiRA_r_shared(MIA):
    '''Fit an online HG LiRA model with one local residual precision shared by both classes.'''

    @dataclass
    class _ClassPrior:
        '''Hold class-specific center and query-precision prior parameters.'''

        xi: float
        lam: float
        a_beta: float
        b_beta: float

    @dataclass
    class _SharedPrior:
        '''Hold the common residual-precision prior and local scale.'''

        a_tau: float
        b_tau: float
        r: np.ndarray

    def __init__(
        self,
        ref_sigs,
        ref_train_mask,
        offline=False,
        sig_transformation="none",
        n_gibbs_samples=128,
        n_gibbs_warmup=64,
        n_quadrature=50,
        random_seed=42,
    ):
        '''Fit shared priors and Gibbs draws from model, point, and query losses.'''
        if offline:
            raise ValueError("HG_LiRA_r_shared supports online inference only.")
        self.n_ref, self.n_data, self.n_sigs = ref_sigs.shape
        if self.n_ref < 4 or self.n_ref % 2:
            raise ValueError("HG_LiRA_r_shared needs an even number of at least four references.")
        if self.n_sigs < 2:
            raise ValueError("HG_LiRA_r_shared needs at least two queries per point.")
        self.sig_transformation = sig_transformation
        transformed = self._transform_signals(ref_sigs.detach().cpu().numpy())
        self.ref_train_mask = ref_train_mask.detach().cpu().numpy().astype(bool)
        if self.ref_train_mask.shape != (self.n_ref, self.n_data):
            raise ValueError("Reference mask dimensions do not match loss signals.")
        class_size = self.n_ref // 2
        if not np.all(self.ref_train_mask.sum(axis=0) == class_size):
            raise ValueError("Each point needs equally many IN and OUT references.")
        self.reference_in = self._extract_class(transformed, self.ref_train_mask)
        self.reference_out = self._extract_class(transformed, ~self.ref_train_mask)
        self.n_gibbs_samples = n_gibbs_samples
        self.n_gibbs_warmup = n_gibbs_warmup
        self.n_quadrature = n_quadrature
        self.eps = 1e-9
        self.prior_in, self.prior_out, self.prior_shared = self._estimate_priors()
        self.rng = np.random.default_rng(random_seed)
        self.mean_in, self.var_in, self.mean_out, self.var_out = self._sample_posterior()
        self.nodes, weights = np.polynomial.hermite.hermgauss(n_quadrature)
        self.log_weights = np.log(weights)[:, None, None]

    def _transform_signals(self, signals):
        '''Transform query losses to evidence; return a finite NumPy array.'''
        if self.sig_transformation == "log":
            transformed = -np.log(signals)
        elif self.sig_transformation == "none":
            transformed = -signals
        else:
            raise ValueError(f"Unknown loss transformation: {self.sig_transformation}")
        if not np.isfinite(transformed).all():
            raise ValueError("Loss transformation produced non-finite values.")
        return transformed

    def _extract_class(self, signals, mask):
        '''Gather a balanced membership class; return point, model, query values.'''
        class_size = self.n_ref // 2
        point_model_queries = signals.transpose(1, 0, 2)
        return point_model_queries[mask.T].reshape(self.n_data, class_size, self.n_sigs)

    def _dispersion_scale(self, local_variance, center):
        '''Estimate positive mean-one scales from local variance and center.'''
        positive_variance = np.maximum(local_variance, self.eps)
        centered_log_scale = np.log(positive_variance) - np.log(positive_variance).mean()
        center_scale = max(float(np.std(center, ddof=1)), np.sqrt(self.eps))
        standardized_center = (center - np.mean(center)) / center_scale
        denominator = max(float(np.sum(standardized_center ** 2)), self.eps)
        slope = float(np.sum(standardized_center * centered_log_scale)) / denominator
        scale = np.exp(slope * standardized_center)
        return scale / np.mean(scale)

    def _class_statistics(self, reference):
        '''Return class centers, local variances, and query precision moments.'''
        model_mean = reference.mean(axis=2)
        query_variance = reference.var(axis=2, ddof=1)
        center = model_mean.mean(axis=1)
        local_variance = model_mean.var(axis=1, ddof=1)
        local_variance -= query_variance.mean(axis=1) / self.n_sigs
        first_beta = float(query_variance.mean())
        second_beta_terms = (self.n_sigs - 1.0) * query_variance ** 2
        second_beta = float((second_beta_terms / (self.n_sigs + 1.0)).mean())
        beta_shape, beta_rate = _gamma_from_inverse_moments(
            first_beta,
            second_beta,
            self.eps,
        )
        return center, local_variance, first_beta, beta_shape, beta_rate

    def _estimate_priors(self):
        '''Estimate separate center/query priors and a shared local residual prior.'''
        stats_in = self._class_statistics(self.reference_in)
        stats_out = self._class_statistics(self.reference_out)
        pooled_variance = 0.5 * (stats_in[1] + stats_out[1])
        pooled_center = 0.5 * (stats_in[0] + stats_out[0])
        scale = self._dispersion_scale(pooled_variance, pooled_center)
        adjusted_in = stats_in[1] / scale
        adjusted_out = stats_out[1] / scale
        first_tau = float(np.maximum(0.5 * (adjusted_in + adjusted_out), self.eps).mean())
        second_tau = float((adjusted_in * adjusted_out).mean())
        tau_shape, tau_rate = _gamma_from_inverse_moments(
            first_tau,
            second_tau,
            self.eps,
        )
        shared_prior = self._SharedPrior(tau_shape, tau_rate, scale)
        class_size = self.n_ref // 2
        class_priors = []
        for center, _, first_beta, beta_shape, beta_rate in (stats_in, stats_out):
            center_variance = float(center.var(ddof=1))
            center_variance -= first_tau / class_size
            center_variance -= first_beta / (self.n_sigs * class_size)
            lam = 1.0 / max(center_variance, self.eps)
            class_priors.append(self._ClassPrior(float(center.mean()), lam, beta_shape, beta_rate))
        return class_priors[0], class_priors[1], shared_prior

    def _initial_class_state(self, reference):
        '''Return model means, squared deviations, and initial Gibbs state.'''
        model_mean = reference.mean(axis=2)
        sample_ss = np.sum((reference - model_mean[:, :, None]) ** 2, axis=2)
        sample_variance = sample_ss / (self.n_sigs - 1.0)
        mu = model_mean.copy()
        beta = 1.0 / np.maximum(sample_variance, self.eps)
        nu = model_mean.mean(axis=1)
        return [model_mean, sample_ss, mu, beta, nu]

    def _sample_posterior(self):
        '''Run the joint Gibbs chain; return four arrays of predictive draws.'''
        class_size = self.n_ref // 2
        state_in = self._initial_class_state(self.reference_in)
        state_out = self._initial_class_state(self.reference_out)
        local_in = self._class_statistics(self.reference_in)[1]
        local_out = self._class_statistics(self.reference_out)[1]
        initial_variance = 0.5 * (local_in + local_out) / self.prior_shared.r
        tau = 1.0 / np.maximum(initial_variance, self.eps)
        retained = [[], [], [], []]
        total_steps = self.n_gibbs_warmup + self.n_gibbs_samples
        for step in range(total_steps):
            local_precision = tau / self.prior_shared.r
            for state, prior in ((state_in, self.prior_in), (state_out, self.prior_out)):
                model_mean, sample_ss, mu, beta, nu = state
                mu_precision = self.n_sigs * beta + local_precision[:, None]
                mu_center = self.n_sigs * beta * model_mean
                mu_center += local_precision[:, None] * nu[:, None]
                mu_center /= mu_precision
                mu = self.rng.normal(mu_center, np.sqrt(1.0 / mu_precision))
                beta_rate = sample_ss + self.n_sigs * (model_mean - mu) ** 2
                beta_rate = prior.b_beta + 0.5 * beta_rate
                beta_shape = prior.a_beta + 0.5 * self.n_sigs
                beta = self.rng.gamma(beta_shape, scale=1.0 / beta_rate)
                nu_precision = prior.lam + class_size * local_precision
                nu_center = prior.lam * prior.xi + local_precision * mu.sum(axis=1)
                nu_center /= nu_precision
                nu = self.rng.normal(nu_center, np.sqrt(1.0 / nu_precision))
                state[2:] = mu, beta, nu
            in_residual = np.sum((state_in[2] - state_in[4][:, None]) ** 2, axis=1)
            out_residual = np.sum((state_out[2] - state_out[4][:, None]) ** 2, axis=1)
            tau_shape = self.prior_shared.a_tau + class_size
            tau_rate = self.prior_shared.b_tau
            tau_rate += 0.5 * (in_residual + out_residual) / self.prior_shared.r
            tau = self.rng.gamma(tau_shape, scale=1.0 / tau_rate)
            if step < self.n_gibbs_warmup:
                continue
            local_precision = tau / self.prior_shared.r
            for offset, (state, prior) in enumerate(((state_in, self.prior_in), (state_out, self.prior_out))):
                mu = state[2]
                center_precision = prior.lam + class_size * local_precision
                predictive_mean = prior.lam * prior.xi + local_precision * mu.sum(axis=1)
                predictive_mean /= center_precision
                predictive_variance = self.prior_shared.r / tau + 1.0 / center_precision
                retained[2 * offset].append(predictive_mean)
                retained[2 * offset + 1].append(predictive_variance)
        return tuple(np.asarray(values) for values in retained)

    def _log_query_density(self, target, candidate_mean, prior):
        '''Integrate query precision analytically for each candidate mean.'''
        query_mean = target.mean(axis=-1)
        query_ss = np.sum((target - query_mean[:, None]) ** 2, axis=-1)
        broadcast_shape = (1,) * (candidate_mean.ndim - 1) + query_mean.shape
        query_mean = query_mean.reshape(broadcast_shape)
        query_ss = query_ss.reshape(broadcast_shape)
        shape = prior.a_beta
        rate = prior.b_beta
        log_density = gammaln(shape + 0.5 * self.n_sigs) - gammaln(shape)
        log_density += shape * np.log(rate) - 0.5 * self.n_sigs * np.log(2 * np.pi)
        log_density -= (shape + 0.5 * self.n_sigs) * np.log(
            rate + 0.5 * (query_ss + self.n_sigs * (query_mean - candidate_mean) ** 2)
        )
        return log_density

    def _log_posterior(self, target, means, variances, prior):
        '''Integrate predictive means over quadrature nodes and Gibbs draws.'''
        chunk_size = max(10_000_000 // (self.n_quadrature * self.n_gibbs_samples), 1)
        log_posterior = []
        for start in range(0, self.n_data, chunk_size):
            stop = min(start + chunk_size, self.n_data)
            mean_chunk = means[:, start:stop]
            variance_chunk = variances[:, start:stop]
            candidate_mean = mean_chunk[None, :, :]
            candidate_mean = candidate_mean + np.sqrt(2.0 * variance_chunk)[None, :, :] * self.nodes[:, None, None]
            log_density = self._log_query_density(target[start:stop], candidate_mean, prior)
            weighted_density = self.log_weights + log_density
            state_density = logsumexp(weighted_density, axis=0) - 0.5 * np.log(np.pi)
            point_density = logsumexp(state_density, axis=0) - np.log(self.n_gibbs_samples)
            log_posterior.append(point_density)
        return np.concatenate(log_posterior)

    def run_attack(self, target_loss_sigs):
        '''Return per-point online log likelihood ratios from target query losses.'''
        target = self._transform_signals(target_loss_sigs.detach().cpu().numpy())
        if target.shape != (self.n_data, self.n_sigs):
            raise ValueError("Target loss-signal dimensions do not match references.")
        log_in = self._log_posterior(target, self.mean_in, self.var_in, self.prior_in)
        log_out = self._log_posterior(target, self.mean_out, self.var_out, self.prior_out)
        return torch.tensor(log_in - log_out, dtype=torch.float32)
