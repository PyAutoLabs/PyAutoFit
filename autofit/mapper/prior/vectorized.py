import numpy as np

from autofit.mapper.prior.gaussian import GaussianPrior
from autofit.mapper.prior.truncated_gaussian import TruncatedGaussianPrior
from autofit.mapper.prior.uniform import UniformPrior
from autofit.mapper.prior.log_uniform import LogUniformPrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel


class PriorVectorized:

    def __init__(self, model: AbstractPriorModel):
        """
        Vectorized transformer for a model's priors that batches together priors by type
        and applies inverse transformations from unit cube to physical parameter space.

        This performs the same unit transformation as the individual prior classes
        `value_for` functions, it simply groups and performs them all in one go
        in order to make the mapping from unit cube to physical parameter space
        more efficient.

        Supports Uniform, Gaussian, Truncated Gaussian, and LogUniform priors.

        Raises exceptions if the model contains unsupported prior types.


        Parameters
        ----------
        model : Model
            A model object that contains priors ordered by ID, accessible via
            `model.priors_ordered_by_id`.
        """

        self.model = model
        self.prior_list = model.priors_ordered_by_id

        supported_prior_list = [
            UniformPrior,
            GaussianPrior,
            TruncatedGaussianPrior,
            LogUniformPrior,
        ]

        # 1) Group UniformPriors with key info
        self.uniform_idx = [
            i for i, p in enumerate(self.prior_list) if isinstance(p, UniformPrior)
        ]

        if self.uniform_idx:

            self.uniform_lowers = np.array(
                [self.prior_list[i].lower_limit for i in self.uniform_idx]
            )  # (n_uniforms,)
            self.uniform_uppers = np.array(
                [self.prior_list[i].upper_limit for i in self.uniform_idx]
            )  # (n_uniforms,)

        self.gaussian_idx = [
            i for i, p in enumerate(self.prior_list) if isinstance(p, GaussianPrior)
        ]

        if self.gaussian_idx:

            self.gaussian_means = np.array(
                [self.prior_list[i].mean for i in self.gaussian_idx]
            )
            self.gaussian_sigmas = np.array(
                [self.prior_list[i].sigma for i in self.gaussian_idx]
            )

        self.truncated_gaussian_idx = [
            i
            for i, p in enumerate(self.prior_list)
            if isinstance(p, TruncatedGaussianPrior)
        ]

        if self.truncated_gaussian_idx:

            self.truncated_gaussian_means = np.array(
                [self.prior_list[i].mean for i in self.truncated_gaussian_idx]
            )
            self.truncated_gaussian_sigmas = np.array(
                [self.prior_list[i].sigma for i in self.truncated_gaussian_idx]
            )
            lowers = np.array(
                [self.prior_list[i].lower_limit for i in self.truncated_gaussian_idx]
            )
            uppers = np.array(
                [self.prior_list[i].upper_limit for i in self.truncated_gaussian_idx]
            )

            a = (
                        lowers - self.truncated_gaussian_means
                ) / self.truncated_gaussian_sigmas
            b = (
                        uppers - self.truncated_gaussian_means
                ) / self.truncated_gaussian_sigmas

            from scipy.stats import norm

            self.truncated_gaussian_cdf_a = norm.cdf(a)
            self.truncated_gaussian_cdf_b = norm.cdf(b)

        self.loguniform_idx = [
            i
            for i, p in enumerate(self.prior_list)
            if isinstance(p, LogUniformPrior)
        ]

        if self.loguniform_idx:

            self.loguniform_lowers = np.array(
                [self.prior_list[i].lower_limit for i in self.loguniform_idx]
            )  # (n_loguniforms,)
            self.loguniform_uppers = np.array(
                [self.prior_list[i].upper_limit for i in self.loguniform_idx]
            )  # (n_loguniforms,)
            # Map unit interval to log scale:
            # x = exp(log(lower) + unit * (log(upper) - log(lower)))
            self.loguniform_log_lowers = np.log(self.loguniform_lowers)
            self.loguniform_log_uppers = np.log(self.loguniform_uppers)

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        """
        Apply vectorized prior transformation from unit cube [0, 1] to physical space.

        Parameters
        ----------
        cube
            Array of shape (n_samples, n_priors) with values in [0, 1] which are mapped
            to physical parameter space via the priors defined in the model.

        Returns
        -------
        out
            Transformed parameters of shape (n_samples, n_priors).
        """

        cube_reshaped = False

        if len(cube.shape) == 1:
            cube = cube[None, :]
            cube_reshaped = True

        out = np.empty_like(cube)

        # 2) Batch‐process all UniformPriors
        if self.uniform_idx:
            subcube = cube[:, self.uniform_idx]  # shape (n_samples, n_uniforms)

            out[:, self.uniform_idx] = self.uniform_lowers + subcube * (
                    self.uniform_uppers - self.uniform_lowers
            )

        # 3) Batch‐process all GaussianPriors
        if self.gaussian_idx:
            from scipy.stats import norm

            subcube = cube[:, self.gaussian_idx]  # (n_samples, n_gaussians)

            inv = norm.ppf(subcube)  # inverse CDF of standard normal
            out[:, self.gaussian_idx] = self.gaussian_means + self.gaussian_sigmas * inv

        # 4) Batch‐process all TruncatedGaussianPriors
        if self.truncated_gaussian_idx:

            subcube = cube[:, self.truncated_gaussian_idx]  # (n_samples, n_truncs)

            from scipy.stats import norm

            truncated_cdf = self.truncated_gaussian_cdf_a + subcube * (
                    self.truncated_gaussian_cdf_b - self.truncated_gaussian_cdf_a
            )
            x_std = norm.ppf(truncated_cdf)

            out[:, self.truncated_gaussian_idx] = (
                    self.truncated_gaussian_means + self.truncated_gaussian_sigmas * x_std
            )

        # 5) Batch‐process all LogUniformPriors
        if self.loguniform_idx:

            subcube = cube[:, self.loguniform_idx]  # (n_samples, n_loguniforms)

            out[:, self.loguniform_idx] = np.exp(
                self.loguniform_log_lowers
                + subcube * (self.loguniform_log_uppers - self.loguniform_log_lowers)
            )

        if cube_reshaped:

            return out[0]

        return out