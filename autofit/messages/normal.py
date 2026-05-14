from collections.abc import Hashable

from typing import Optional, Tuple, Union

import numpy as np

from autoconf import cached_property
from autofit.mapper.operator import LinearOperator
from autofit.messages.abstract import AbstractMessage
from .composed_transform import TransformedMessage
from .transform import (
    phi_transform,
    log_transform,
    multinomial_logit_transform,
    log_10_transform,
)
from .. import exc


def is_nan(value):
    is_nan_ = np.isnan(value)
    if isinstance(is_nan_, np.ndarray):
        is_nan_ = is_nan_.all()
    return is_nan_

def assert_sigma_non_negative(sigma, xp=np):

    is_negative = sigma < 0

    if xp.__name__.startswith("jax"):
        import jax
        # JAX path: cannot convert to Python bool
        # Raise using JAX control flow:
        return jax.lax.cond(
            is_negative,
            lambda _: (_ for _ in ()).throw(
                ValueError("Sigma cannot be negative")
            ),
            lambda _: None,
            operand=None,
        )
    else:
        # NumPy path: normal boolean works
        if bool(is_negative):
            raise ValueError("Sigma cannot be negative")

class NormalMessage(AbstractMessage):

    def log_partition(self, xp=np):
        """
        Compute the log-partition function (also called log-normalizer or cumulant function)
        for the normal distribution in its natural (canonical) parameterization.
        #
        Let the natural parameters be:
          η₁ = μ / σ²
          η₂ = -1 / (2σ²)

        Then the log-partition function A(η) for the Gaussian is:
           A(η) = η₁² / (-4η₂) - 0.5 * log(-2η₂)
        This ensures normalization of the exponential-family distribution.
        """
        eta1, eta2 = self.natural_parameters(xp=xp)
        return -(eta1**2) / 4 / eta2 - xp.log(-2 * eta2) / 2

    log_base_measure = -0.5 * np.log(2 * np.pi)
    _support = ((-np.inf, np.inf),)
    _parameter_support = ((-np.inf, np.inf), (0, np.inf))

    def __init__(
        self,
        mean : Union[float, np.ndarray],
        sigma : Union[float, np.ndarray],
        log_norm : Optional[float] = 0.0,
        id_ : Optional[Hashable] = None,
    ):
        """
        A Gaussian (Normal) message representing a probability distribution over a continuous variable.

        This message defines a Normal distribution parameterized by its mean and standard deviation (sigma).

        Parameters
        ----------
        mean
            The mean (μ) of the normal distribution.

        sigma
            The standard deviation (σ) of the distribution. Must be non-negative.

        log_norm
            An additive constant to the log probability of the message. Used internally for message-passing normalization.
            Default is 0.0.

        id_
            An optional unique identifier used to track the message in larger probabilistic graphs or models.
        """

        if isinstance(mean, (np.ndarray, float, int, list)):
            xp = np
        else:
            import jax.numpy as jnp
            xp = jnp

        # assert_sigma_non_negative(sigma, xp=xp)

        super().__init__(
            mean,
            sigma,
            log_norm=log_norm,
            id_=id_,
            _xp=xp
        )
        self.mean, self.sigma = self.parameters

    def cdf(self, x : Union[float, np.ndarray], xp=np) -> Union[float, np.ndarray]:
        """
        Compute the cumulative distribution function (CDF) of the Gaussian distribution
        at a given value or array of values `x`.

        Parameters
        ----------
        x
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        The cumulative probability P(X ≤ x).
        """
        from scipy.stats import norm

        return norm.cdf(x, loc=self.mean, scale=self.sigma)

    def ppf(self, x : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the percent-point function (inverse CDF) of the Gaussian distribution.

        This function maps a probability value `x` in [0, 1] to the corresponding value
        of the distribution with that cumulative probability.

        Parameters
        ----------
        x
            The cumulative probability or array of probabilities.

        Returns
        -------
        The value(s) corresponding to the input quantiles.
        """
        from scipy.stats import norm

        return norm.ppf(x, loc=self.mean, scale=self.sigma)

    def natural_parameters(self, xp=np) -> np.ndarray:
        """
        The natural (canonical) parameters of the Gaussian distribution in exponential-family form.

        For a Normal distribution with mean μ and standard deviation σ, the natural parameters η are:

            η₁ = μ / σ²
            η₂ = -1 / (2σ²)

        Returns
        -------
        A NumPy array containing the two natural parameters [η₁, η₂].
        """
        return self.calc_natural_parameters(self.mean, self.sigma, xp=xp)

    @staticmethod
    def calc_natural_parameters(mu : Union[float, np.ndarray], sigma : Union[float, np.ndarray], xp=np) -> np.ndarray:
        """
        Convert standard parameters of a Gaussian distribution (mean and standard deviation)
        into natural parameters used in its exponential family representation.

        Parameters
        ----------
        mu
            Mean of the Gaussian distribution.
        sigma
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        Natural parameters [η₁, η₂], where:
            η₁ = μ / σ²
            η₂ = -1 / (2σ²)
        """
        precision = 1 / sigma**2
        return xp.array([mu * precision, -precision / 2])

    @staticmethod
    def invert_natural_parameters(natural_parameters : np.ndarray) -> Tuple[float, float]:
        """
        Convert natural parameters [η₁, η₂] back into standard parameters (mean and sigma)
        of a Gaussian distribution.

        Parameters
        ----------
        natural_parameters
            The natural parameters [η₁, η₂] from the exponential family form.

        Returns
        -------
        The corresponding (mean, sigma) of the Gaussian distribution.
        """
        eta1, eta2 = natural_parameters
        mu = -0.5 * eta1 / eta2
        sigma = np.sqrt(-0.5 / eta2)
        return mu, sigma

    @staticmethod
    def to_canonical_form(x : Union[float, np.ndarray], xp=np) -> np.ndarray:
        """
        Convert a scalar input `x` to its sufficient statistics for the Gaussian exponential family.

        The sufficient statistics for a normal distribution are [x, x²], which correspond to the
        inner product with the natural parameters in the exponential-family log-likelihood.

        Parameters
        ----------
        x
            Input data point or array of points.

        Returns
        -------
        The sufficient statistics [x, x²].
        """
        return xp.array([x, x**2])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats: Tuple[float, float]) -> np.ndarray:
        """
        Convert sufficient statistics [E[x], E[x²]] into natural parameters [η₁, η₂].

        Parameters
        ----------
        suff_stats
            First and second moments of the distribution.

        Returns
        -------
        Natural parameters of the Gaussian.
        """
        m1, m2 = suff_stats
        sigma = np.sqrt(m2 - m1**2)
        return cls.calc_natural_parameters(m1, sigma)

    @cached_property
    def variance(self) -> np.ndarray:
        """
        Return the variance σ² of the Gaussian distribution.
        """
        return self.sigma**2

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Draw samples from the Gaussian distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw. If None, returns a single sample.

        Returns
        -------
        Sample(s) from the distribution.
        """
        if n_samples:
            x = np.random.randn(n_samples, *self.shape)
            if self.shape:
                return x * self.sigma[None, ...] + self.mean[None, ...]
        else:
            x = np.random.randn(*self.shape)

        return x * self.sigma + self.mean

    def kl(self, dist : "NormalMessage") -> float:
        """
        Compute the Kullback-Leibler (KL) divergence to another Gaussian distribution.

        Parameters
        ----------
        dist : Gaussian
            The target distribution for the KL divergence.

        Returns
        -------
        float
            The KL divergence KL(self || dist).
        """
        return (
            np.log(dist.sigma / self.sigma)
            + (self.sigma**2 + (self.mean - dist.mean) ** 2) / 2 / dist.sigma**2
            - 1 / 2
        )

    @classmethod
    def from_mode(
        cls,
        mode: np.ndarray,
        covariance: Union[float, LinearOperator] = 1.0,
        **kwargs
    ) -> "NormalMessage":
        """
        Construct a Gaussian from its mode and covariance.

        Parameters
        ----------
        mode
            The mode (same as mean for Gaussian).
        covariance
            The covariance or a linear operator with `.diagonal()` method.

        Returns
        -------
        An instance of the NormalMessage class.
        """
        if isinstance(covariance, LinearOperator):
            variance = covariance.diagonal()
        else:
            mode, variance = cls._get_mean_variance(mode, covariance)

        if kwargs.get("upper_limit") is not None:
            kwargs.pop("upper_limit")

        if kwargs.get("lower_limit") is not None:
            kwargs.pop("lower_limit")

        return cls(mode, np.abs(variance) ** 0.5, **kwargs)

    def _normal_gradient_hessian(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the log-pdf, gradient, and Hessian of a Gaussian with respect to x.

        Parameters
        ----------
        x
            Points at which to evaluate.

        Returns
        -------
        Log-pdf values, gradients, and Hessians.
        """
        # raise Exception
        shape = np.shape(x)
        if shape:
            x = np.asanyarray(x)
            deltax = x - self.mean
            hess_logl = -self.sigma**-2
            grad_logl = deltax * hess_logl
            eta_t = 0.5 * grad_logl * deltax
            logl = self.log_base_measure + eta_t - np.log(self.sigma)

            if shape[1:] == self.shape:
                hess_logl = np.repeat(
                    np.reshape(hess_logl, (1,) + np.shape(hess_logl)), shape[0], 0
                )

        else:
            deltax = x - self.mean
            hess_logl = -self.sigma**-2
            grad_logl = deltax * hess_logl
            eta_t = 0.5 * grad_logl * deltax
            logl = self.log_base_measure + eta_t - np.log(self.sigma)

        return logl, grad_logl, hess_logl

    def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the gradient of the log-pdf of the Gaussian evaluated at `x`.

        Parameters
        ----------
        x
            Evaluation points.

        Returns
        -------
        Log-pdf values and gradients.
        """
        return self._normal_gradient_hessian(x)[:2]

    def logpdf_gradient_hessian(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the gradient and Hessian of the log-pdf of the Gaussian at `x`.

        Parameters
        ----------
        x
            Evaluation points.

        Returns
        -------
        Gradient and Hessian of the log-pdf.
        """
        return self._normal_gradient_hessian(x)

    __name__ = "gaussian_prior"

    __default_fields__ = ("log_norm", "id_")

    def value_for(self, unit, xp=np):
        """
        Map a unit value in [0, 1] to a physical value drawn from this Gaussian prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1 representing a uniform draw.
        xp
            Array-module to dispatch on (``numpy`` or ``jax.numpy``). Default ``numpy``.

        Returns
        -------
        A physical value sampled from the Gaussian prior corresponding to the given unit.

        Examples
        --------
        >>> prior = af.GaussianPrior(mean=1.0, sigma=2.0)
        >>> physical_value = prior.value_for(unit=0.5)
        """
        if xp is np:
            from scipy.special import erfinv
        else:
            from jax.scipy.special import erfinv

        inv = erfinv(2.0 * unit - 1.0)
        return self.mean + self.sigma * xp.sqrt(2.0) * inv

    def log_prior_from_value(self, value: float, xp=np) -> float:
        """
        Compute the log prior probability of a given physical value under this Gaussian prior.

        Used to convert a likelihood to a posterior in non-linear searches (e.g., Emcee).

        Parameters
        ----------
        value
            A physical parameter value for which the log prior is evaluated.

        Returns
        -------
        The log prior probability of the given value.
        """
        return (value - self.mean) ** 2.0 / (2 * self.sigma**2.0)

    def __str__(self):
        """
        Generate a short string summary describing the prior for use in model summaries.
        """
        return f"NormalMessage, mean = {self.mean}, sigma = {self.sigma}"

    def __repr__(self):
        """
        Return the official string representation of this Gaussian prior including
        the ID, mean, sigma, and optional bounds.
        """
        return (
            "<NormalMessage id={} mean={} sigma={}>".format(
                self.id, self.mean, self.sigma,
            )
        )

    @property
    def natural(self)-> "NaturalNormal":
        """
        Return a 'zeroed' natural parameterization of this Gaussian prior.

        Returns
        -------
        A natural form Gaussian with zeroed parameters but same configuration.
        """
        return NaturalNormal.from_natural_parameters(
            self.natural_parameters() * 0.0, **self._init_kwargs
        )

    def zeros_like(self) -> "AbstractMessage":
        """
        Return a new instance of this prior with the same structure but zeroed natural parameters.

        Useful for initializing messages in variational inference frameworks.

        Returns
        -------
        A new prior object with zeroed natural parameters.
        """
        return self.natural.zeros_like()


class NaturalNormal(NormalMessage):
    """
    Identical to the NormalMessage but allows non-normalised values,
    e.g negative or infinite variances
    """

    _parameter_support = ((-np.inf, np.inf), (-np.inf, 0))

    def __init__(
        self,
        eta1 : float,
        eta2 : float,
        log_norm : Optional[float] = 0.0,
        id_ : Optional[Hashable] = None,
    ):
        """
        A natural parameterization of a Gaussian distribution.

        This class behaves like `NormalMessage`, but allows non-normalized or degenerate distributions,
        including those with negative or infinite variance. This flexibility is useful in advanced
        inference settings like message passing or variational approximations, where intermediate
        natural parameter values may fall outside standard constraints.

        In natural form, the parameters `eta1` and `eta2` correspond to:
            - eta1 = mu / sigma^2
            - eta2 = -1 / (2 * sigma^2)

        Parameters
        ----------
        eta1
            First natural parameter, related to the mean.
        eta2
            Second natural parameter, related to the variance (must be < 0).
        log_norm
            Optional additive normalization term for use in message passing.
        id_
            Optional identifier for the distribution instance.
        """
        AbstractMessage.__init__(
            self,
            eta1,
            eta2,
            log_norm=log_norm,
            id_=id_,
        )

    @cached_property
    def sigma(self)-> float:
        """
        Return the standard deviation corresponding to the natural parameters.

        Returns
        -------
        The standard deviation σ, derived from eta2 via σ² = -1/(2η₂).
        """
        precision = -2 * self.parameters[1]
        return precision**-0.5

    @cached_property
    def mean(self) -> float:
        """
        Return the mean corresponding to the natural parameters.

        Returns
        -------
        The mean μ = -η₁ / (2η₂), with NaNs replaced by 0 for numerical stability.
        """
        return np.nan_to_num(-self.parameters[0] / self.parameters[1] / 2)

    @staticmethod
    def calc_natural_parameters(eta1: float, eta2: float, xp=np) -> np.ndarray:
        """
        Return the natural parameters in array form (identity function for this class).

        Parameters
        ----------
        eta1
            The first natural parameter.
        eta2
            The second natural parameter.
        """
        return xp.array([eta1, eta2])

    def natural_parameters(self, xp=np) -> np.ndarray:
        """
        Return the natural parameters of this distribution.
        """
        return self.calc_natural_parameters(*self.parameters, xp=xp)

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats: Tuple[float, float]) -> np.ndarray:
        """
        Convert sufficient statistics back to natural parameters.

        Parameters
        ----------
        suff_stats
            Tuple of first and second moments: (mean, second_moment).

        Returns
        -------
        Natural parameters [eta1, eta2] recovered from the sufficient statistics.
        """
        m1, m2 = suff_stats
        precision = 1 / (m2 - m1**2)
        return cls.calc_natural_parameters(m1 * precision, -precision / 2)

    @staticmethod
    def invert_natural_parameters(natural_parameters: np.ndarray) -> np.ndarray:
        """
        Identity function for natural parameters (no inversion needed).

        Parameters
        ----------
        natural_parameters : np.ndarray
            Natural parameters [eta1, eta2].

        Returns
        -------
        np.ndarray
            The same input array.
        """
        return natural_parameters

    @classmethod
    def from_mode(
            cls,
            mode: np.ndarray,
            covariance: Union[float, LinearOperator] = 1.0,
            **kwargs
    ) -> "NaturalNormal":
        """
        Construct a `NaturalNormal` distribution from mode and covariance.

        Parameters
        ----------
        mode
            The mode (mean) of the distribution.
        covariance
            Covariance of the distribution. If a `LinearOperator`, its inverse is used for precision.
        kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        An instance of `NaturalNormal` with the corresponding natural parameters.
        """
        if isinstance(covariance, LinearOperator):
            precision = covariance.inv().diagonal()
        else:
            mode, variance = cls._get_mean_variance(mode, covariance)
            precision = 1 / variance

        return cls(mode * precision, -precision / 2, **kwargs)

    zeros_like = AbstractMessage.zeros_like

    @property
    def natural(self) -> "NaturalNormal":
        """
        Return self — already in natural form -- for clean API.
        """
        return self


UniformNormalMessage = TransformedMessage(NormalMessage(0, 1), phi_transform)

Log10UniformNormalMessage = TransformedMessage(UniformNormalMessage, log_10_transform)

LogNormalMessage = TransformedMessage(NormalMessage(0, 1), log_transform)
Log10NormalMessage = TransformedMessage(NormalMessage(0, 1), log_10_transform)

# Support is the simplex
MultiLogitNormalMessage = TransformedMessage(
    NormalMessage(0, 1), multinomial_logit_transform
)
