from dataclasses import dataclass

from numpy.polynomial.hermite import hermgauss
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri_exp
from scipy.stats import chi2, multivariate_normal, norm, qmc
import torch

from .attacks_sample import BASE

class CompositeBASE:

    def __init__(self, attack_config, shadow_loss_sigs, shadow_train_mask):
        self.base_attack = BASE(
            shadow_loss_sigs=shadow_loss_sigs,
            shadow_train_mask=shadow_train_mask,
            offline=attack_config.offline,
            prior=attack_config.prior,
            apply_sigmoid=False,
        )

    def run_attack(self, audit_table, target_loss_sigs):
        sample_scores = self.base_attack.run_attack(target_loss_sigs)
        scores = {}
        for entity_id, indices in audit_table.items():
            entity_scores = sample_scores[indices].to(dtype=torch.float32)
            scores[entity_id] = entity_scores.sum()
        return scores

class CompositeLiRA:

    def __init__(self, audit_table, shadow_loss_sigs, shadow_entity_mask, offline=True, use_full_cov=False):
        self.audit_table = audit_table
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_entity_mask = shadow_entity_mask
        self.offline = offline
        self.use_full_cov = use_full_cov
        if not self.offline:
            self.mean_in, self.cov_in = self.get_mean_and_var(self.shadow_entity_mask)
        self.mean_out, self.cov_out = self.get_mean_and_var(~self.shadow_entity_mask)

    def loss_transformation(self, loss_sigs):
        # TODO: Add other loss transformations
        return -loss_sigs

    def get_mean_and_var(self, entity_mask):
        phi_shadow = self.loss_transformation(self.shadow_loss_sigs)
        mean = {}
        phi = {}
        for entity_id, indices in self.audit_table.items():
            phi[entity_id] = phi_shadow[entity_mask[:, entity_id]][:, indices]
            mean[entity_id] = phi[entity_id].mean(dim=0)
        if self.use_full_cov:
            x = torch.stack([*phi.values()], dim=0)
            x = x.view(x.shape[0] * x.shape[1], -1)
            x_centered = x - x.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (x.shape[0] - 1)
        else:
            cov = torch.cat([*phi.values()], dim=0).var().clamp_min(1e-9)
        return mean, cov

    def run_attack(self, target_loss_sigs):
        phi_target = self.loss_transformation(target_loss_sigs.numpy())
        score = {}
        for entity_id, indices in self.audit_table.items():
            if not self.use_full_cov:
                if not self.offline:
                    cov_in = np.eye(len(indices), dtype=np.float32) * float(self.cov_in)
                cov_out = np.eye(len(indices), dtype=np.float32) * float(self.cov_out)
            if self.offline:
                if self.use_full_cov:
                    log_p = multivariate_normal.logcdf(
                        phi_target[indices],
                        mean=self.mean_out[entity_id].numpy(),
                        cov=self.cov_out.numpy(),
                    )
                else:
                    log_p = -norm.logsf(
                        phi_target[indices],
                        loc=self.mean_out[entity_id].numpy(),
                        scale=np.sqrt(float(self.cov_out)),
                    ).sum()
                score[entity_id] = torch.tensor(log_p, dtype=torch.float32)
            else:
                log_p_in = multivariate_normal.logpdf(
                    phi_target[indices],
                    mean=self.mean_in[entity_id].numpy(),
                    cov=cov_in,
                )
                log_p_out = multivariate_normal.logpdf(
                    phi_target[indices],
                    mean=self.mean_out[entity_id].numpy(),
                    cov=cov_out,
                )
                score[entity_id] = torch.tensor(log_p_in - log_p_out, dtype=torch.float32)
        return score

class CompositeLiRAv2:

    def __init__(
        self,
        audit_table,
        shadow_loss_sigs,
        shadow_entity_mask,
        offline=True,
        use_full_cov=False,
        loss_transformation="none",
        covariance=None,
        use_global_dispersion=True,
        share_variance=False,
        covariance_rank=2,
        covariance_shrinkage=0.1,
        n_qmc_samples=1024,
        random_seed=0,
    ):
        if covariance is None:
            covariance = "full" if use_full_cov else "spherical"
        if covariance not in {"spherical", "diagonal", "low_rank", "full"}:
            raise ValueError(f"Unknown CompositeLiRA covariance: {covariance}")
        if share_variance and (offline or covariance != "spherical"):
            raise ValueError("Shared variance requires online spherical CompositeLiRA.")
        if int(covariance_rank) != covariance_rank or covariance_rank < 1:
            raise ValueError("CompositeLiRA covariance rank must be a positive integer.")
        if not 0.0 <= covariance_shrinkage <= 1.0:
            raise ValueError("CompositeLiRA covariance shrinkage must be in [0, 1].")
        if int(n_qmc_samples) != n_qmc_samples or n_qmc_samples < 2:
            raise ValueError("CompositeLiRA QMC sample count must be an integer of at least two.")
        n_qmc_samples = int(n_qmc_samples)
        if n_qmc_samples & (n_qmc_samples - 1):
            raise ValueError("CompositeLiRA QMC sample count must be a power of two.")
        if not audit_table or any(len(indices) == 0 for indices in audit_table.values()):
            raise ValueError("CompositeLiRA requires nonempty entity audit indices.")
        self.audit_table = audit_table
        self.shadow_loss_sigs = shadow_loss_sigs
        self.shadow_entity_mask = shadow_entity_mask
        self.offline = offline
        self.use_full_cov = covariance in {"low_rank", "full"}
        self.covariance = covariance
        self.use_global_dispersion = use_global_dispersion
        self.share_variance = bool(share_variance)
        self.covariance_rank = int(covariance_rank)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.loss_transformation_name = loss_transformation
        self.mean_in, self.var_in, self.cov_in, self.mean_out, self.var_out, self.cov_out = self.get_mean_and_var()
        if self.share_variance:
            entity_ids = list(self.audit_table)
            entity_mask = self.shadow_entity_mask[:, entity_ids].detach().cpu().to(dtype=torch.bool)
            in_dof = entity_mask.sum(dim=0).to(dtype=torch.float64) - 1.0
            out_dof = (~entity_mask).sum(dim=0).to(dtype=torch.float64) - 1.0
            if self.use_global_dispersion:
                entity_sizes = torch.tensor(
                    [len(self.audit_table[entity_id]) for entity_id in entity_ids],
                    dtype=torch.float64,
                )
                in_weight = (in_dof * entity_sizes).sum()
                out_weight = (out_dof * entity_sizes).sum()
                shared_variance = (
                    in_weight * self.var_in + out_weight * self.var_out
                ) / (in_weight + out_weight)
                self.var_in = shared_variance
                self.var_out = shared_variance
            else:
                for index, entity_id in enumerate(entity_ids):
                    in_weight = in_dof[index]
                    out_weight = out_dof[index]
                    shared_variance = (
                        in_weight * self.var_in[entity_id]
                        + out_weight * self.var_out[entity_id]
                    ) / (in_weight + out_weight)
                    self.var_in[entity_id] = shared_variance
                    self.var_out[entity_id] = shared_variance
        self.log_uniforms = {}
        self.cholesky_out = None
        if self.offline and self.use_full_cov:
            probe_counts = {len(indices) for indices in audit_table.values()}
            for count in sorted(probe_counts):
                sampler = qmc.Sobol(d=count, scramble=True, seed=random_seed)
                uniforms = sampler.random_base2(int(np.log2(n_qmc_samples)))
                uniforms = np.clip(uniforms, np.finfo(float).tiny, np.nextafter(1.0, 0.0))
                self.log_uniforms[count] = np.log(uniforms)
            if self.use_global_dispersion:
                self.cholesky_out = np.linalg.cholesky(self.cov_out.numpy())
            else:
                self.cholesky_out = {
                    entity_id: np.linalg.cholesky(cov.numpy())
                    for entity_id, cov in self.cov_out.items()
                }

    def loss_transformation(self, loss_sigs):
        if not torch.isfinite(loss_sigs).all():
            raise ValueError("CompositeLiRA requires finite loss signals.")
        match self.loss_transformation_name:
            case "log":
                if torch.any(loss_sigs <= 0.0):
                    raise ValueError("CompositeLiRA log transformation requires positive signals.")
                transformed_loss_sigs = -torch.log(loss_sigs)
            case "none":
                transformed_loss_sigs = -loss_sigs
            case _:
                raise ValueError(f"Unknown loss transformation: {self.loss_transformation_name}")
        if transformed_loss_sigs.ndim == 3:
            transformed_loss_sigs = transformed_loss_sigs.mean(dim=-1)
        return transformed_loss_sigs

    def _fit_class(self, phi_shadow, entity_mask):
        '''Fit class means and global or local dispersion from shadow reference vectors.'''
        values = {}
        means = {}
        for entity_id, indices in self.audit_table.items():
            references = phi_shadow[entity_mask[:, entity_id]][:, indices]
            if len(references) < 1:
                raise ValueError("CompositeLiRA requires a reference for each fitted entity/class.")
            values[entity_id] = references
            means[entity_id] = references.mean(dim=0)
        residuals = {
            entity_id: references.to(dtype=torch.float64) - means[entity_id].to(dtype=torch.float64)
            for entity_id, references in values.items()
        }
        degrees_of_freedom = sum(len(references) - 1 for references in values.values())
        if self.use_global_dispersion:
            if degrees_of_freedom < 1:
                raise ValueError("CompositeLiRA global dispersion requires within-entity reference variation.")
            if self.covariance == "spherical":
                sum_squares = sum(residual.square().sum() for residual in residuals.values())
                total_degrees_of_freedom = sum(
                    (len(values[entity_id]) - 1) * residual.shape[1]
                    for entity_id, residual in residuals.items()
                )
                variance = (sum_squares / total_degrees_of_freedom).clamp_min(1e-9)
                return means, variance, None
            probe_counts = {value.shape[1] for value in values.values()}
            if len(probe_counts) != 1:
                raise ValueError("CompositeLiRA global coordinate dispersion requires equal probe counts.")
            pooled_residuals = torch.cat(list(residuals.values()), dim=0)
            if self.covariance == "diagonal":
                sum_squares = pooled_residuals.square().sum(dim=0)
                variance = (sum_squares / degrees_of_freedom).clamp_min(1e-9)
                return means, variance, None
            covariance = self._estimate_covariance(pooled_residuals, degrees_of_freedom)
            return means, None, covariance
        variances = {}
        covariances = {}
        for entity_id, residual in residuals.items():
            references = values[entity_id]
            if len(references) < 2:
                raise ValueError("CompositeLiRA local dispersion requires two references per entity/class.")
            local_degrees_of_freedom = len(references) - 1
            if self.covariance == "spherical":
                variance = residual.square().sum() / (local_degrees_of_freedom * residual.shape[1])
                variances[entity_id] = variance.clamp_min(1e-9)
            elif self.covariance == "diagonal":
                variance = residual.square().sum(dim=0) / local_degrees_of_freedom
                variances[entity_id] = variance.clamp_min(1e-9)
            else:
                covariances[entity_id] = self._estimate_covariance(residual, local_degrees_of_freedom)
        return means, variances, covariances

    def _estimate_covariance(self, residuals, degrees_of_freedom):
        '''Return a regularized covariance from centered vectors and their residual degrees of freedom.'''
        empirical_covariance = (residuals.T @ residuals) / degrees_of_freedom
        diagonal = empirical_covariance.diag().clamp_min(1e-9)
        if self.covariance == "low_rank":
            eigenvalues, eigenvectors = torch.linalg.eigh(empirical_covariance)
            rank = min(self.covariance_rank, residuals.shape[1], degrees_of_freedom)
            eigenvalues = eigenvalues[-rank:].clamp_min(0.0)
            factors = eigenvectors[:, -rank:] * eigenvalues.sqrt()
            covariance = factors @ factors.T
            covariance *= 1.0 - self.covariance_shrinkage
            residual_diagonal = (diagonal - covariance.diag()).clamp_min(1e-9)
            covariance += torch.diag(residual_diagonal)
        else:
            covariance = (1.0 - self.covariance_shrinkage) * empirical_covariance
            covariance += self.covariance_shrinkage * torch.diag(diagonal)
            covariance += 1e-9 * torch.eye(residuals.shape[1], dtype=torch.float64)
        return covariance

    def get_mean_and_var(self):
        phi_shadow = self.loss_transformation(self.shadow_loss_sigs)
        phi_shadow = phi_shadow.detach().cpu()
        entity_mask = self.shadow_entity_mask.detach().cpu().to(dtype=torch.bool)
        mean_in, var_in, cov_in = {}, None, None
        if not self.offline:
            mean_in, var_in, cov_in = self._fit_class(phi_shadow, entity_mask)
        mean_out, var_out, cov_out = self._fit_class(phi_shadow, ~entity_mask)
        return mean_in, var_in, cov_in, mean_out, var_out, cov_out

    def _log_upper_tail(self, residual, cholesky):
        '''Estimate a Gaussian upper-orthant log probability using seeded truncated-normal QMC.'''
        log_uniforms = self.log_uniforms[len(residual)]
        latent = np.zeros_like(log_uniforms)
        log_weights = np.zeros(len(log_uniforms))
        for index in range(len(residual)):
            conditional_mean = np.einsum("ij,j->i", latent[:, :index], cholesky[index, :index])
            threshold = (residual[index] - conditional_mean) / cholesky[index, index]
            log_tail = norm.logsf(threshold)
            log_weights += log_tail
            if index + 1 < len(residual):
                log_probability = log_uniforms[:, index] + log_tail
                latent[:, index] = -ndtri_exp(log_probability)
        return logsumexp(log_weights) - np.log(len(log_weights))

    def run_attack(self, target_loss_sigs):
        phi_target = self.loss_transformation(target_loss_sigs)
        if phi_target.ndim == 2:
            phi_target = phi_target.mean(dim=-1)
        phi_target = phi_target.detach().cpu().numpy()
        scores = {}
        for entity_id, indices in self.audit_table.items():
            target = phi_target[indices]
            mean_out = self.mean_out[entity_id].numpy()
            if self.use_full_cov:
                covariance_out = self.cov_out if self.use_global_dispersion else self.cov_out[entity_id]
                covariance_out = covariance_out.numpy()
                if self.offline:
                    cholesky = self.cholesky_out if self.use_global_dispersion else self.cholesky_out[entity_id]
                    score = -self._log_upper_tail(target - mean_out, cholesky)
                else:
                    covariance_in = self.cov_in if self.use_global_dispersion else self.cov_in[entity_id]
                    log_in = multivariate_normal.logpdf(target, mean=self.mean_in[entity_id].numpy(), cov=covariance_in.numpy())
                    log_out = multivariate_normal.logpdf(target, mean=mean_out, cov=covariance_out)
                    score = log_in - log_out
            else:
                variance_out = self.var_out if self.use_global_dispersion else self.var_out[entity_id]
                scale_out = np.sqrt(variance_out.numpy().astype(np.float64))
                if self.offline:
                    score = -norm.logsf(target, loc=mean_out, scale=scale_out).sum()
                else:
                    variance_in = self.var_in if self.use_global_dispersion else self.var_in[entity_id]
                    scale_in = np.sqrt(variance_in.numpy().astype(np.float64))
                    log_in = norm.logpdf(target, loc=self.mean_in[entity_id].numpy(), scale=scale_in).sum()
                    log_out = norm.logpdf(target, loc=mean_out, scale=scale_out).sum()
                    score = log_in - log_out
            if not np.isfinite(score):
                raise ValueError("CompositeLiRA produced a nonfinite score.")
            scores[entity_id] = torch.tensor(score, dtype=torch.float32)
        return scores

class HBE_Simple:
    '''Fit an O-only entity hierarchy from shadow losses and score target tails.'''

    def __init__(
        self,
        audit_table,
        shadow_loss_sigs,
        shadow_entity_mask,
        loss_transformation="none",
        n_gibbs_samples=128,
        n_gibbs_warmup=64,
        random_seed=0,
        min_variance=1e-9,
    ):
        '''Fit first-moment priors and retain Gaussian predictive components.
        Inputs are audit indices, shadow losses/memberships, and sampler options.
        The fitted state predicts sums of transformed probe means per entity.
        '''
        if shadow_loss_sigs.ndim != 3 or shadow_loss_sigs.shape[2] < 2:
            raise ValueError("HBE_Simple requires models by images by repeated queries, with Q > 1.")
        if shadow_entity_mask.ndim != 2:
            raise ValueError("HBE_Simple requires a models by entities membership mask.")
        if shadow_entity_mask.shape[0] != shadow_loss_sigs.shape[0]:
            raise ValueError("HBE_Simple reference model dimensions do not match.")
        if len(audit_table) < 2:
            raise ValueError("HBE_Simple requires at least two entities.")
        if not np.isfinite(min_variance) or min_variance <= 0.0:
            raise ValueError("HBE_Simple requires a positive finite variance floor.")
        if int(n_gibbs_samples) != n_gibbs_samples or n_gibbs_samples < 1:
            raise ValueError("HBE_Simple requires a positive integer number of retained draws.")
        if int(n_gibbs_warmup) != n_gibbs_warmup or n_gibbs_warmup < 0:
            raise ValueError("HBE_Simple requires a nonnegative integer warmup.")
        if loss_transformation not in {"log", "none"}:
            raise ValueError(f"Unknown loss transformation: {loss_transformation}")
        self.loss_transformation = loss_transformation
        self.n_gibbs_samples = int(n_gibbs_samples)
        self.n_gibbs_warmup = int(n_gibbs_warmup)
        self.random_seed = int(random_seed)
        self.min_variance = float(min_variance)
        self.target_shape = tuple(shadow_loss_sigs.shape[1:])
        self.n_attributes = self.target_shape[1]
        self.entity_ids = sorted(audit_table)
        entity_ids = np.asarray(self.entity_ids)
        if not np.issubdtype(entity_ids.dtype, np.integer):
            raise ValueError("HBE_Simple requires integer entity identifiers.")
        if entity_ids[0] < 0 or entity_ids[-1] >= shadow_entity_mask.shape[1]:
            raise ValueError("HBE_Simple audit entity is absent from the membership mask.")
        entity_indices = []
        for entity_id in self.entity_ids:
            indices = np.asarray(audit_table[entity_id])
            if indices.ndim != 1 or indices.size == 0:
                raise ValueError("HBE_Simple requires a nonempty index vector for each entity.")
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("HBE_Simple requires integer image indices.")
            if np.any(indices < 0) or np.any(indices >= self.target_shape[0]):
                raise ValueError("HBE_Simple audit image index is out of bounds.")
            entity_indices.append(indices)
        self.selected_indices = np.concatenate(entity_indices)
        if np.unique(self.selected_indices).size != self.selected_indices.size:
            raise ValueError("HBE_Simple requires unique audit indices.")
        self.entity_sizes = np.array([len(indices) for indices in entity_indices])
        self.entity_starts = np.cumsum(self.entity_sizes) - self.entity_sizes
        self.unit_entity_indices = np.repeat(np.arange(len(self.entity_ids)), self.entity_sizes)
        entity_mask = shadow_entity_mask[:, self.entity_ids].detach().cpu().numpy()
        reference_mask = ~entity_mask.astype(bool)
        reference_mask = reference_mask[:, self.unit_entity_indices].T
        selected_signals = shadow_loss_sigs[:, self.selected_indices]
        shadow_values = self._transform_signals(selected_signals)
        self.model_mean = shadow_values.mean(axis=2).T
        centered_values = shadow_values - self.model_mean.T[:, :, None]
        self.sample_ss = np.sum(centered_values ** 2, axis=2).T
        summaries = self._reference_summaries(reference_mask)
        self.stats_out = self._fit_gibbs(summaries, reference_mask)

    def _transform_signals(self, signals):
        '''Convert finite Torch signals to NumPy evidence, optionally taking logs.'''
        values = signals.detach().to(device="cpu", dtype=torch.float32)
        values = values.numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("HBE_Simple requires finite loss signals.")
        if self.loss_transformation == "log":
            if np.any(values <= 0.0):
                raise ValueError("HBE_Simple log transformation requires positive signals.")
            values = np.log(values)
        return -values

    def _reference_summaries(self, reference_mask):
        '''Estimate query, image, and shared-entity variance moments from O data.'''
        n_models = reference_mask.sum(axis=1).astype(np.float64)
        if np.any(n_models < 2):
            raise ValueError("HBE_Simple requires two O references per entity.")
        weighted_mean = self.model_mean * reference_mask
        y_bar = weighted_mean.sum(axis=1) / n_models
        residual = (self.model_mean - y_bar[:, None]) * reference_mask
        residual_square = residual ** 2
        model_var = residual_square.sum(axis=1) / (n_models - 1.0)
        sample_var = self.sample_ss / (self.n_attributes - 1.0)
        query_var = (sample_var * reference_mask).sum(axis=1) / n_models
        query_mean_var = query_var / self.n_attributes
        tau_inverse = model_var - query_mean_var
        tau_inverse = np.maximum(tau_inverse, self.min_variance)
        center = float(y_bar.mean())
        tau_first = float(np.mean(tau_inverse))
        beta_first = max(float(sample_var[reference_mask].mean()), self.min_variance)
        center_noise = (tau_first + query_mean_var) / n_models
        center_variance = float(y_bar.var(ddof=1)) - float(center_noise.mean())
        return {
            "n_models": n_models,
            "y_bar": y_bar,
            "sample_var": sample_var,
            "tau_first": tau_first,
            "beta_first": beta_first,
            "center": center,
            "lambda": 1.0 / max(center_variance, self.min_variance),
        }

    def _fit_gibbs(self, summaries, reference_mask):
        '''Use conjugate Gaussian/Gamma updates and return entity predictive draws.'''
        rng = np.random.default_rng(self.random_seed)
        predictive_rng = np.random.default_rng(self.random_seed + 1)
        prior_shape = 1.5
        beta_rate_prior = (prior_shape - 1.0) * summaries["beta_first"]
        tau_rate_prior = (prior_shape - 1.0) * summaries["tau_first"]
        n_models = summaries["n_models"]
        lam = summaries["lambda"]
        center = summaries["center"]
        mu = self.model_mean.copy()
        nu = summaries["y_bar"].copy()
        beta = 1.0 / np.maximum(summaries["sample_var"], self.min_variance)
        tau = np.full(len(nu), 1.0 / summaries["tau_first"])
        predictive_means = np.empty(
            (self.n_gibbs_samples, len(self.entity_ids)),
            dtype=np.float32,
        )
        predictive_variances = np.empty_like(predictive_means)
        total_steps = self.n_gibbs_warmup + self.n_gibbs_samples
        for step in range(total_steps):
            local_precision = tau
            mu_precision = self.n_attributes * beta + local_precision[:, None]
            mu_center = self.n_attributes * beta * self.model_mean
            mu_center += local_precision[:, None] * nu[:, None]
            mu_center /= mu_precision
            mu = rng.normal(mu_center, np.sqrt(1.0 / mu_precision)).astype(
                np.float32,
                copy=False,
            )
            beta_rate = self.sample_ss + self.n_attributes * (self.model_mean - mu) ** 2
            beta_rate = beta_rate_prior + 0.5 * beta_rate
            beta = rng.gamma(
                prior_shape + 0.5 * self.n_attributes,
                scale=1.0 / beta_rate,
            ).astype(np.float32, copy=False)
            centered_mu = mu * reference_mask
            nu_precision = lam + n_models * local_precision
            nu_center = lam * center + local_precision * centered_mu.sum(axis=1)
            nu_center /= nu_precision
            nu = rng.normal(nu_center, np.sqrt(1.0 / nu_precision)).astype(
                np.float32,
                copy=False,
            )
            residual = mu - nu[:, None]
            residual_square = residual ** 2 * reference_mask
            tau_rate = tau_rate_prior + 0.5 * residual_square.sum(axis=1)
            tau = rng.gamma(
                prior_shape + 0.5 * n_models,
                scale=1.0 / tau_rate,
            ).astype(np.float32, copy=False)
            if step < self.n_gibbs_warmup:
                continue
            # Integrate image centers conditionally, then sum the Gaussian means.
            local_precision = tau
            nu_precision = lam + n_models * local_precision
            centered_mu = mu * reference_mask
            nu_center = lam * center + local_precision * centered_mu.sum(axis=1)
            nu_center /= nu_precision
            draw_index = step - self.n_gibbs_warmup
            predictive_means[draw_index] = np.add.reduceat(nu_center, self.entity_starts)
            new_beta = predictive_rng.gamma(
                prior_shape,
                scale=1.0 / beta_rate_prior,
                size=len(nu),
            ).astype(np.float32, copy=False)
            point_variance = 1.0 / nu_precision + 1.0 / local_precision
            point_variance += 1.0 / (self.n_attributes * new_beta)
            entity_variance = np.add.reduceat(point_variance, self.entity_starts)
            predictive_variances[draw_index] = entity_variance
        return {"mean": predictive_means, "variance": predictive_variances}

    def predictive_log_tails(self, target_loss_sigs):
        '''Return aligned log CDF and survival arrays for target entity totals.'''
        if tuple(target_loss_sigs.shape) != self.target_shape:
            raise ValueError("HBE_Simple target dimensions do not match the reference signals.")
        target_values = self._transform_signals(target_loss_sigs[self.selected_indices])
        target_mean = target_values.mean(axis=1)
        target_total = np.add.reduceat(target_mean, self.entity_starts)
        means = self.stats_out["mean"]
        variances = self.stats_out["variance"]
        variances = variances.mean(axis=0) + means.var(axis=0)
        means = means.mean(axis=0)
        log_tail = norm.logsf(target_total, loc=means, scale=np.sqrt(variances))
        log_cdf = norm.logcdf(target_total, loc=means, scale=np.sqrt(variances))
        return log_cdf, log_tail

    def run_attack(self, target_loss_sigs):
        '''Return entity scores from upper tails of summed transformed query means.'''
        _, log_tail = self.predictive_log_tails(target_loss_sigs)
        if not np.all(np.isfinite(log_tail)):
            raise ValueError("HBE_Simple predictive tails must be finite.")
        scores = {}
        for entity_id, value in zip(self.entity_ids, -log_tail):
            scores[entity_id] = torch.tensor(value, dtype=torch.float64)
        return scores

@dataclass
class _LocalCases:
    '''Store padded reference observations and their known query-mean variances.'''

    values: np.ndarray
    query_variances: np.ndarray
    valid: np.ndarray
    prior_centers: np.ndarray
    unit_indices: np.ndarray
    group_indices: np.ndarray


class _HBELocalVariance:
    '''Fit reference-only local predictive moments for HBE_GlobalLatent.'''

    def __init__(
        self,
        audit_table,
        shadow_loss_sigs,
        shadow_entity_mask,
        loss_transformation="none",
        quadrature_nodes=24,
        max_hyper_units=4096,
        min_variance=1e-9,
    ):
        '''Fit pooled local variance shrinkage from O-reference losses.'''
        self._validate_inputs(
            audit_table,
            shadow_loss_sigs,
            shadow_entity_mask,
            loss_transformation,
            quadrature_nodes,
            max_hyper_units,
            min_variance,
        )
        self.loss_transformation = loss_transformation
        self.quadrature_nodes = int(quadrature_nodes)
        self.max_hyper_units = int(max_hyper_units)
        self.min_variance = float(min_variance)
        self.target_shape = tuple(shadow_loss_sigs.shape[1:])
        self.n_queries = self.target_shape[1]
        self.entity_ids = sorted(audit_table)
        entity_indices = [np.asarray(audit_table[entity_id], dtype=int) for entity_id in self.entity_ids]
        self.entity_sizes = np.array([len(indices) for indices in entity_indices], dtype=int)
        self.entity_starts = np.cumsum(self.entity_sizes) - self.entity_sizes
        self.selected_indices = np.concatenate(entity_indices)
        self.unit_entity_indices = np.repeat(np.arange(len(self.entity_ids)), self.entity_sizes)
        self.group_labels = np.array(["pooled"], dtype=object)
        self.model_groups = np.zeros(len(shadow_loss_sigs), dtype=int)
        entity_mask = shadow_entity_mask[:, self.entity_ids].detach().cpu().numpy().astype(bool)
        reference_mask = ~entity_mask[:, self.unit_entity_indices].T
        signals = shadow_loss_sigs[:, self.selected_indices]
        values = self._transform_signals(signals, reference_mask)
        model_means = values.mean(axis=2).T
        query_variances = values.var(axis=2, ddof=1).T / self.n_queries
        cases = self._build_cases(model_means, query_variances, reference_mask)
        prior = self._fit_variance_prior(cases)
        point_means, point_variances = self._posterior_moments(cases, prior)
        self.local_cases = cases
        self.variance_prior = prior
        self.point_means = point_means
        self.point_variances = point_variances
        self.fit_diagnostics = {
            "model": "hbe_local_variance",
            "reference_only": True,
            "target_group_used": False,
            "groups": [str(label) for label in self.group_labels],
            "quadrature_nodes": self.quadrature_nodes,
            "hyper_cases": prior["hyper_cases"],
            "variance_log_location": prior["location"],
            "variance_log_scale": prior["scale"],
            "center_prior_weight": prior["center_weight"],
            "optimizer_success": prior["success"],
            "optimizer_message": prior["message"],
            "objective": prior["objective"],
        }

    def _validate_inputs(
        self,
        audit_table,
        shadow_loss_sigs,
        shadow_entity_mask,
        loss_transformation,
        quadrature_nodes,
        max_hyper_units,
        min_variance,
    ):
        '''Reject incompatible shapes, indices and deterministic integration settings.'''
        if shadow_loss_sigs.ndim != 3 or shadow_loss_sigs.shape[2] < 2:
            raise ValueError("HBE local variance requires models by images by repeated queries, with Q > 1.")
        if shadow_entity_mask.ndim != 2:
            raise ValueError("HBE local variance requires a models by entities membership mask.")
        if shadow_entity_mask.shape[0] != shadow_loss_sigs.shape[0]:
            raise ValueError("HBE local variance reference model dimensions do not match.")
        if loss_transformation not in {"log", "none"}:
            raise ValueError(f"Unknown loss transformation: {loss_transformation}")
        if int(quadrature_nodes) != quadrature_nodes or quadrature_nodes < 8:
            raise ValueError("HBE local variance requires at least eight quadrature nodes.")
        if int(max_hyper_units) != max_hyper_units or max_hyper_units < 32:
            raise ValueError("HBE local variance requires at least 32 hyperparameter-fitting cases.")
        if not np.isfinite(min_variance) or min_variance <= 0.0:
            raise ValueError("HBE local variance requires a positive finite variance floor.")
        if len(audit_table) < 2:
            raise ValueError("HBE local variance requires at least two entities.")
        entity_ids = np.asarray(sorted(audit_table))
        if not np.issubdtype(entity_ids.dtype, np.integer):
            raise ValueError("HBE local variance requires integer entity identifiers.")
        if entity_ids[0] < 0 or entity_ids[-1] >= shadow_entity_mask.shape[1]:
            raise ValueError("HBE local variance audit entity is absent from the membership mask.")
        all_indices = []
        for entity_id in entity_ids:
            indices = np.asarray(audit_table[int(entity_id)])
            if indices.ndim != 1 or not len(indices):
                raise ValueError("HBE local variance requires a nonempty index vector for each entity.")
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("HBE local variance requires integer image indices.")
            if np.any(indices < 0) or np.any(indices >= shadow_loss_sigs.shape[1]):
                raise ValueError("HBE local variance audit image index is out of bounds.")
            all_indices.extend(indices.tolist())
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("HBE local variance requires unique audit indices.")

    def _transform_signals(self, signals, relevant_mask=None):
        '''Convert finite Torch signals to NumPy evidence, optionally taking logs.'''
        values = signals.detach().to(device="cpu", dtype=torch.float64).numpy()
        relevant = values
        if relevant_mask is not None:
            relevant = np.transpose(values, (1, 0, 2))[relevant_mask]
        if not np.all(np.isfinite(relevant)):
            raise ValueError("HBE local variance requires finite loss signals.")
        if self.loss_transformation == "log":
            if np.any(relevant <= 0.0):
                raise ValueError("HBE local variance log transformation requires positive signals.")
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.log(values)
        return -values

    def _build_cases(self, model_means, query_variances, reference_mask):
        '''Create one padded local likelihood case for every image and reference stratum.'''
        n_units, n_models = model_means.shape
        n_groups = len(self.group_labels)
        group_masks = [self.model_groups == group_index for group_index in range(n_groups)]
        counts = np.empty((n_units, n_groups), dtype=int)
        for group_index, group_mask in enumerate(group_masks):
            counts[:, group_index] = np.sum(reference_mask[:, group_mask], axis=1)
        if np.any(counts < 2):
            unit_index, group_index = np.argwhere(counts < 2)[0]
            entity_index = self.unit_entity_indices[unit_index]
            entity_id = self.entity_ids[entity_index]
            group = self.group_labels[group_index]
            raise ValueError(f"HBE local variance requires two O references per entity and stratum; entity={entity_id}, group={group}.")
        n_cases = n_units * n_groups
        width = int(counts.max())
        values = np.zeros((n_cases, width), dtype=float)
        variances = np.zeros_like(values)
        valid = np.zeros_like(values, dtype=bool)
        centers = np.zeros(n_cases, dtype=float)
        unit_indices = np.repeat(np.arange(n_units), n_groups)
        group_indices = np.tile(np.arange(n_groups), n_units)
        for case_index, (unit_index, group_index) in enumerate(zip(unit_indices, group_indices)):
            selected = reference_mask[unit_index] & group_masks[group_index]
            n_selected = int(selected.sum())
            values[case_index, :n_selected] = model_means[unit_index, selected]
            variances[case_index, :n_selected] = query_variances[unit_index, selected]
            valid[case_index, :n_selected] = True
            other = reference_mask[unit_index] & ~group_masks[group_index]
            if np.any(other):
                centers[case_index] = float(model_means[unit_index, other].mean())
            else:
                centers[case_index] = float(model_means[unit_index, selected].mean())
        variances = np.maximum(variances, self.min_variance)
        return _LocalCases(values, variances, valid, centers, unit_indices, group_indices)

    def _quadrature(self, location, scale):
        '''Return log-normal variance nodes and normalized Gauss-Hermite log weights.'''
        standard_nodes, weights = hermgauss(self.quadrature_nodes)
        log_variance = location + np.sqrt(2.0) * scale * standard_nodes
        log_variance = np.clip(log_variance, -30.0, 20.0)
        nodes = np.exp(log_variance)
        log_weights = np.log(weights) - 0.5 * np.log(np.pi)
        return nodes, log_weights

    def _case_log_likelihood(self, cases, variance_nodes, center_weight):
        '''Integrate each Gaussian local center conditionally for every variance node.'''
        local_variance = variance_nodes[:, None, None]
        observation_variance = local_variance + cases.query_variances[None, :, :]
        precision = np.where(cases.valid[None, :, :], 1.0 / observation_variance, 0.0)
        log_variance = np.where(cases.valid[None, :, :], np.log(observation_variance), 0.0)
        precision_sum = precision.sum(axis=2)
        weighted_sum = np.sum(precision * cases.values[None, :, :], axis=2)
        weighted_square = np.sum(precision * cases.values[None, :, :] ** 2, axis=2)
        n_observations = cases.valid.sum(axis=1)[None, :]
        if center_weight > 0.0:
            prior_precision = center_weight / variance_nodes[:, None]
            posterior_precision = precision_sum + prior_precision
            center = cases.prior_centers[None, :]
            posterior_sum = weighted_sum + prior_precision * center
            quadratic = weighted_square + prior_precision * center ** 2
            quadratic -= posterior_sum ** 2 / posterior_precision
            prior_variance = variance_nodes[:, None] / center_weight
            normalization = n_observations * np.log(2.0 * np.pi) + log_variance.sum(axis=2)
            normalization = normalization + np.log(prior_variance)
            normalization = normalization + np.log(posterior_precision)
            return -0.5 * (normalization + quadratic)
        posterior_precision = precision_sum
        quadratic = weighted_square - weighted_sum ** 2 / posterior_precision
        normalization = (n_observations - 1) * np.log(2.0 * np.pi) + log_variance.sum(axis=2)
        normalization = normalization + np.log(posterior_precision)
        return -0.5 * (normalization + quadratic)

    def _initial_hyperparameters(self, cases):
        '''Estimate robust initial log-variance moments from query-corrected local sample variances.'''
        estimates = []
        for values, query, valid in zip(cases.values, cases.query_variances, cases.valid):
            selected_values = values[valid]
            selected_query = query[valid]
            local = np.var(selected_values, ddof=1) - float(selected_query.mean())
            estimates.append(max(local, self.min_variance))
        log_estimates = np.log(estimates)
        location = float(np.median(log_estimates))
        scale = float(np.clip(np.std(log_estimates), 0.2, 1.5))
        return location, scale

    def _fit_variance_prior(self, cases):
        '''Learn variance shrinkage and mixed-center pooling by reference-only marginal likelihood.'''
        n_cases = len(cases.values)
        if n_cases > self.max_hyper_units:
            selected = np.linspace(0, n_cases - 1, self.max_hyper_units, dtype=int)
            fitting = _LocalCases(
                cases.values[selected],
                cases.query_variances[selected],
                cases.valid[selected],
                cases.prior_centers[selected],
                cases.unit_indices[selected],
                cases.group_indices[selected],
            )
        else:
            fitting = cases
        location, scale = self._initial_hyperparameters(fitting)
        use_center_prior = False
        initial = [location, np.log(scale)]
        bounds = [(-20.0, 10.0), (np.log(0.05), np.log(2.5))]
        if use_center_prior:
            initial.append(np.log(0.5))
            bounds.append((np.log(1e-4), np.log(32.0)))

        def objective(parameters):
            '''Return negative reference marginal likelihood for optimizer parameters.'''
            current_scale = np.exp(parameters[1])
            center_weight = np.exp(parameters[2]) if use_center_prior else 0.0
            nodes, log_weights = self._quadrature(parameters[0], current_scale)
            log_likelihood = self._case_log_likelihood(fitting, nodes, center_weight)
            marginal = logsumexp(log_likelihood + log_weights[:, None], axis=0)
            value = -float(marginal.sum())
            if not np.isfinite(value):
                return np.finfo(float).max
            return value

        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80})
        parameters = result.x if np.all(np.isfinite(result.x)) else np.asarray(initial)
        center_weight = float(np.exp(parameters[2])) if use_center_prior else 0.0
        return {
            "location": float(parameters[0]),
            "scale": float(np.exp(parameters[1])),
            "center_weight": center_weight,
            "success": bool(result.success),
            "message": str(result.message),
            "objective": float(objective(parameters)),
            "hyper_cases": len(fitting.values),
        }

    def _posterior_moments(self, cases, prior):
        '''Integrate local center and variance posteriors into pointwise predictive moments.'''
        nodes, log_weights = self._quadrature(prior["location"], prior["scale"])
        center_weight = prior["center_weight"]
        log_likelihood = self._case_log_likelihood(cases, nodes, center_weight)
        log_posterior = log_likelihood + log_weights[:, None]
        log_posterior -= logsumexp(log_posterior, axis=0)[None, :]
        posterior_weights = np.exp(log_posterior)
        observation_variance = nodes[:, None, None] + cases.query_variances[None, :, :]
        precision = np.where(cases.valid[None, :, :], 1.0 / observation_variance, 0.0)
        precision_sum = precision.sum(axis=2)
        weighted_sum = np.sum(precision * cases.values[None, :, :], axis=2)
        if center_weight > 0.0:
            prior_precision = center_weight / nodes[:, None]
            precision_sum += prior_precision
            weighted_sum += prior_precision * cases.prior_centers[None, :]
        conditional_mean = weighted_sum / precision_sum
        conditional_center_variance = 1.0 / precision_sum
        masked_query = np.where(cases.valid, cases.query_variances, 0.0)
        new_query_variance = masked_query.sum(axis=1) / cases.valid.sum(axis=1)
        conditional_variance = nodes[:, None] + new_query_variance[None, :]
        conditional_variance += conditional_center_variance
        predictive_mean = np.sum(posterior_weights * conditional_mean, axis=0)
        predictive_second = conditional_variance + conditional_mean ** 2
        predictive_variance = np.sum(posterior_weights * predictive_second, axis=0)
        predictive_variance -= predictive_mean ** 2
        predictive_variance = np.maximum(predictive_variance, self.min_variance)
        n_units = len(self.selected_indices)
        n_groups = len(self.group_labels)
        means = np.empty((n_groups, n_units), dtype=float)
        variances = np.empty_like(means)
        means[cases.group_indices, cases.unit_indices] = predictive_mean
        variances[cases.group_indices, cases.unit_indices] = predictive_variance
        return means, variances

class HBE_GlobalLatent:
    '''Fit rank-one reference variation and infer one cross-fitted latent coordinate per target model.'''

    def __init__(
        self,
        audit_table,
        shadow_loss_sigs,
        shadow_entity_mask,
        n_folds=5,
        quadrature_nodes=24,
        max_hyper_units=4096,
        factor_steps=50,
        factor_tolerance=1e-7,
        min_variance=1e-9,
    ):
        '''Fit O-only local variance, a fixed rank-one factor, and its residual local model.'''
        self._validate_inputs(
            audit_table,
            shadow_loss_sigs,
            shadow_entity_mask,
            n_folds,
            factor_steps,
            factor_tolerance,
            min_variance,
        )
        self.n_folds = int(n_folds)
        self.factor_steps = int(factor_steps)
        self.factor_tolerance = float(factor_tolerance)
        self.min_variance = float(min_variance)
        self.target_shape = tuple(shadow_loss_sigs.shape[1:])
        self.entity_ids = sorted(audit_table)
        entity_indices = [np.asarray(audit_table[entity_id], dtype=int) for entity_id in self.entity_ids]
        self.entity_sizes = np.array([len(indices) for indices in entity_indices], dtype=int)
        self.entity_starts = np.cumsum(self.entity_sizes) - self.entity_sizes
        self.selected_indices = np.concatenate(entity_indices)
        self.unit_entity_indices = np.repeat(np.arange(len(self.entity_ids)), self.entity_sizes)
        entity_mask = shadow_entity_mask[:, self.entity_ids].detach().cpu().numpy().astype(bool)
        self.reference_mask = ~entity_mask[:, self.unit_entity_indices].T
        selected = shadow_loss_sigs[:, self.selected_indices]
        values = -selected.detach().to(device="cpu", dtype=torch.float64).numpy()
        relevant = np.transpose(values, (1, 0, 2))[self.reference_mask]
        if not np.all(np.isfinite(relevant)):
            raise ValueError("HBE_GlobalLatent requires finite O-reference signals.")
        self.reference_means = values.mean(axis=2).T
        local_options = {
            "audit_table": audit_table,
            "shadow_loss_sigs": shadow_loss_sigs,
            "shadow_entity_mask": shadow_entity_mask,
            "loss_transformation": "none",
            "quadrature_nodes": quadrature_nodes,
            "max_hyper_units": max_hyper_units,
            "min_variance": min_variance,
        }
        initial_local = _HBELocalVariance(
            **local_options,
        )
        initial_loading, initial_coordinates, _ = self._fit_factor(initial_local)
        adjusted_signals = shadow_loss_sigs.clone()
        contribution = initial_loading[:, None] * initial_coordinates[None, :]
        selected_adjustment = torch.tensor(
            contribution.T,
            dtype=adjusted_signals.dtype,
            device=adjusted_signals.device,
        )
        adjusted_signals[:, self.selected_indices] += selected_adjustment[:, :, None]
        local_options["shadow_loss_sigs"] = adjusted_signals
        self.local_model = _HBELocalVariance(
            **local_options,
        )
        self.loading, self.reference_coordinates, factor_diagnostics = self._fit_factor(self.local_model)
        self.point_mean = self.local_model.point_means[0]
        self.point_variance = self.local_model.point_variances[0]
        self.entity_mean = np.add.reduceat(self.point_mean, self.entity_starts)
        self.entity_variance = np.add.reduceat(self.point_variance, self.entity_starts)
        self.entity_loading = np.add.reduceat(self.loading, self.entity_starts)
        self.fold_ids = np.arange(len(self.entity_ids), dtype=int) % self.n_folds
        self.fit_diagnostics = {
            "model": "hbe_global_latent",
            "reference_only_fit": True,
            "target_family_used": False,
            "factor_rank": 1,
            "factor_steps": factor_diagnostics["steps"],
            "factor_converged": factor_diagnostics["converged"],
            "factor_explained_fraction": factor_diagnostics["explained_fraction"],
            "loading_sd": float(self.loading.std()),
            "reference_coordinate_min": float(self.reference_coordinates.min()),
            "reference_coordinate_max": float(self.reference_coordinates.max()),
            "local_variance": self.local_model.fit_diagnostics,
        }
        self.last_target_diagnostics = None

    def _validate_inputs(
        self,
        audit_table,
        shadow_loss_sigs,
        shadow_entity_mask,
        n_folds,
        factor_steps,
        factor_tolerance,
        min_variance,
    ):
        '''Reject incompatible target-independent model and cross-fitting settings.'''
        if shadow_loss_sigs.ndim != 3 or shadow_loss_sigs.shape[2] < 2:
            raise ValueError("HBE_GlobalLatent requires models by images by repeated queries.")
        if shadow_entity_mask.ndim != 2 or shadow_entity_mask.shape[0] != shadow_loss_sigs.shape[0]:
            raise ValueError("HBE_GlobalLatent reference model dimensions do not match.")
        if len(audit_table) < 2:
            raise ValueError("HBE_GlobalLatent requires at least two entities.")
        if int(n_folds) != n_folds or n_folds < 2 or n_folds > len(audit_table):
            raise ValueError("HBE_GlobalLatent requires between two and the number of entities folds.")
        if int(factor_steps) != factor_steps or factor_steps < 1:
            raise ValueError("HBE_GlobalLatent requires a positive integer factor iteration count.")
        if not np.isfinite(factor_tolerance) or factor_tolerance <= 0.0:
            raise ValueError("HBE_GlobalLatent requires a positive finite factor tolerance.")
        if not np.isfinite(min_variance) or min_variance <= 0.0:
            raise ValueError("HBE_GlobalLatent requires a positive finite variance floor.")
        if any(len(indices) < 2 for indices in audit_table.values()):
            raise ValueError("HBE_GlobalLatent requires at least two probes per entity for contrast inference.")

    def _initial_coordinates(self, residual, variance):
        '''Initialize model coordinates with the leading right singular vector of standardized O residuals.'''
        standardized = residual / np.sqrt(variance[:, None])
        standardized = np.where(self.reference_mask, standardized, 0.0)
        column_counts = self.reference_mask.sum(axis=0)
        column_scale = np.sqrt(np.maximum(column_counts, 1))
        standardized /= column_scale[None, :]
        _, _, vectors = np.linalg.svd(standardized, full_matrices=False)
        coordinates = vectors[0].copy()
        coordinates -= coordinates.mean()
        scale = coordinates.std()
        if scale <= self.min_variance:
            return np.linspace(-1.0, 1.0, residual.shape[1])
        return coordinates / scale

    def _normalize_factor(self, loading, coordinates):
        '''Center and standardize coordinates while preserving their rank-one fitted contribution.'''
        coordinate_mean = float(coordinates.mean())
        coordinates = coordinates - coordinate_mean
        coordinate_scale = float(coordinates.std())
        if coordinate_scale <= self.min_variance:
            return np.zeros_like(loading), np.zeros_like(coordinates)
        loading = loading * coordinate_scale
        coordinates = coordinates / coordinate_scale
        nonzero = np.flatnonzero(np.abs(loading) > self.min_variance)
        if len(nonzero) and loading[nonzero[0]] < 0.0:
            loading = -loading
            coordinates = -coordinates
        return loading, coordinates

    def _fit_factor(self, local_model):
        '''Fit a regularized rank-one factor to O-reference model means by alternating least squares.'''
        center = local_model.point_means[0]
        variance = np.maximum(local_model.point_variances[0], self.min_variance)
        residual = self.reference_means - center[:, None]
        coordinates = self._initial_coordinates(residual, variance)
        loading = np.zeros(len(center), dtype=float)
        previous = np.inf
        converged = False
        for step in range(self.factor_steps):
            coordinate_square = coordinates[None, :] ** 2 * self.reference_mask
            loading_denominator = coordinate_square.sum(axis=1) + 1.0
            loading_numerator = np.sum(residual * coordinates[None, :] * self.reference_mask, axis=1)
            loading = loading_numerator / loading_denominator
            weights = self.reference_mask / variance[:, None]
            coordinate_denominator = np.sum(weights * loading[:, None] ** 2, axis=0) + 1.0
            coordinate_numerator = np.sum(weights * loading[:, None] * residual, axis=0)
            coordinates = coordinate_numerator / coordinate_denominator
            loading, coordinates = self._normalize_factor(loading, coordinates)
            fitted = loading[:, None] * coordinates[None, :]
            error = np.sum(weights * (residual - fitted) ** 2)
            change = np.inf
            if np.isfinite(previous):
                change = abs(previous - error) / max(abs(previous), 1.0)
            if change < self.factor_tolerance:
                converged = True
                break
            previous = error
        weights = self.reference_mask / variance[:, None]
        baseline_error = np.sum(weights * residual ** 2)
        fitted_error = np.sum(weights * (residual - loading[:, None] * coordinates[None, :]) ** 2)
        explained = 1.0 - fitted_error / max(baseline_error, self.min_variance)
        diagnostics = {
            "steps": step + 1,
            "converged": converged,
            "explained_fraction": float(explained),
        }
        return loading, coordinates, diagnostics

    def _target_values(self, target_loss_sigs):
        '''Return aligned target evidence means after validating dimensions and finiteness.'''
        if tuple(target_loss_sigs.shape) != self.target_shape:
            raise ValueError("HBE_GlobalLatent target dimensions do not match the reference signals.")
        selected = target_loss_sigs[self.selected_indices]
        values = -selected.detach().to(device="cpu", dtype=torch.float64).numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("HBE_GlobalLatent requires finite target signals.")
        return values.mean(axis=1)

    def _contrast_statistics(self, target_values):
        '''Profile entity-wide offsets and return Gaussian information for the global latent coordinate.'''
        residual = target_values - self.point_mean
        information = np.empty(len(self.entity_ids), dtype=float)
        score = np.empty_like(information)
        residual_square = np.empty_like(information)
        for entity_index, (start, size) in enumerate(zip(self.entity_starts, self.entity_sizes)):
            stop = start + size
            variance = self.point_variance[start:stop]
            weight = 1.0 / variance
            loading = self.loading[start:stop]
            values = residual[start:stop]
            weight_sum = weight.sum()
            centered_loading = loading - np.sum(weight * loading) / weight_sum
            centered_values = values - np.sum(weight * values) / weight_sum
            information[entity_index] = np.sum(weight * centered_loading ** 2)
            score[entity_index] = np.sum(weight * centered_loading * centered_values)
            residual_square[entity_index] = np.sum(weight * centered_values ** 2)
        return information, score, residual_square

    def infer_target_latent(self, target_loss_sigs):
        '''Infer fold-specific Gaussian posteriors for one target-wide coordinate from other entities.'''
        target_values = self._target_values(target_loss_sigs)
        information, score, residual_square = self._contrast_statistics(target_values)
        means = np.empty(self.n_folds, dtype=float)
        variances = np.empty(self.n_folds, dtype=float)
        for fold in range(self.n_folds):
            training = self.fold_ids != fold
            precision = 1.0 + information[training].sum()
            means[fold] = score[training].sum() / precision
            variances[fold] = 1.0 / precision
        diagnostics = self._target_diagnostics(target_values, means, variances, residual_square)
        return means, variances, diagnostics

    def _contrast_log_likelihood(self, indices, target_values, mean, variance):
        '''Integrate latent-coordinate uncertainty and an unrestricted entity offset for one contrast vector.'''
        covariance = np.diag(self.point_variance[indices])
        loading = self.loading[indices]
        covariance += variance * np.outer(loading, loading)
        residual = target_values[indices] - self.point_mean[indices] - loading * mean
        precision = np.linalg.inv(covariance)
        ones = np.ones(len(indices))
        offset_precision = float(ones @ precision @ ones)
        offset = float(ones @ precision @ residual) / offset_precision
        centered = residual - offset
        quadratic = float(centered @ precision @ centered)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0.0:
            raise ValueError("HBE_GlobalLatent contrast covariance is not positive definite.")
        log_density = -0.5 * ((len(indices) - 1) * np.log(2.0 * np.pi))
        log_density -= 0.5 * (log_determinant + np.log(offset_precision) + quadratic)
        return log_density, quadratic

    def _target_diagnostics(self, target_values, means, variances, residual_square):
        '''Summarize cross-fitted target coordinate stability and held-fold contrast prediction.'''
        log_density = np.empty(len(self.entity_ids), dtype=float)
        quadratic = np.empty_like(log_density)
        degrees = self.entity_sizes - 1
        for entity_index, (start, size) in enumerate(zip(self.entity_starts, self.entity_sizes)):
            indices = np.arange(start, start + size)
            fold = self.fold_ids[entity_index]
            log_density[entity_index], quadratic[entity_index] = self._contrast_log_likelihood(
                indices,
                target_values,
                means[fold],
                variances[fold],
            )
        pit = chi2.cdf(quadratic, degrees)
        return {
            "fold_mean": means,
            "fold_sd": np.sqrt(variances),
            "fold_mean_range": float(means.max() - means.min()),
            "mean_posterior_sd": float(np.sqrt(variances).mean()),
            "mean_contrast_log_density": float(log_density.mean()),
            "mean_chi_square_ratio": float(np.mean(quadratic / degrees)),
            "contrast_pit_mean": float(pit.mean()),
            "contrast_upper_0p01": float(np.mean(pit > 0.99)),
            "contrast_upper_0p001": float(np.mean(pit > 0.999)),
            "raw_profile_residual": float(residual_square.mean()),
        }

    def predictive_log_tails(self, target_loss_sigs):
        '''Return cross-fitted entity log CDFs and survival probabilities conditional on target contrasts.'''
        target_values = self._target_values(target_loss_sigs)
        target_total = np.add.reduceat(target_values, self.entity_starts)
        means, variances, diagnostics = self.infer_target_latent(target_loss_sigs)
        entity_fold_mean = means[self.fold_ids]
        entity_fold_variance = variances[self.fold_ids]
        predictive_mean = self.entity_mean + self.entity_loading * entity_fold_mean
        predictive_variance = self.entity_variance + self.entity_loading ** 2 * entity_fold_variance
        scale = np.sqrt(np.maximum(predictive_variance, self.min_variance))
        self.last_target_diagnostics = diagnostics
        log_cdf = norm.logcdf(target_total, loc=predictive_mean, scale=scale)
        log_tail = norm.logsf(target_total, loc=predictive_mean, scale=scale)
        return log_cdf, log_tail

    def predictive_log_density(self, target_loss_sigs):
        '''Return cross-fitted entity-total log densities conditional on target contrast inference.'''
        target_values = self._target_values(target_loss_sigs)
        target_total = np.add.reduceat(target_values, self.entity_starts)
        means, variances, diagnostics = self.infer_target_latent(target_loss_sigs)
        predictive_mean = self.entity_mean + self.entity_loading * means[self.fold_ids]
        predictive_variance = self.entity_variance + self.entity_loading ** 2 * variances[self.fold_ids]
        self.last_target_diagnostics = diagnostics
        return norm.logpdf(target_total, loc=predictive_mean, scale=np.sqrt(predictive_variance))

    def run_attack(self, target_loss_sigs):
        '''Return negative log upper-tail entity scores using cross-fitted target-wide latent inference.'''
        _, log_tail = self.predictive_log_tails(target_loss_sigs)
        if not np.all(np.isfinite(log_tail)):
            raise ValueError("HBE_GlobalLatent predictive tails must be finite.")
        scores = {}
        for entity_id, value in zip(self.entity_ids, -log_tail):
            scores[entity_id] = torch.tensor(value, dtype=torch.float64)
        return scores
