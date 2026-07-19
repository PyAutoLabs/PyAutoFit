import math
import warnings
from typing import Tuple, Union

import numpy as np

from autonerves import cached_property
from ..messages.abstract import AbstractMessage


def grad_betaln(ab: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the log Beta function with respect to parameters a and b.

    Parameters
    ----------
    ab
        Array of shape (N, 2) where each row contains parameters [a, b] of Beta distributions.

    Returns
    -------
    Gradient array of shape (N, 2) with derivatives of log Beta function w.r.t a and b.
    """
    from scipy.special import psi

    psiab = psi(ab.sum(axis=1, keepdims=True))
    return psi(ab) - psiab


def jac_grad_betaln(ab: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian matrix of the gradient of the log Beta function.

    Parameters
    ----------
    ab
        Array of shape (N, 2) with Beta parameters [a, b].

    Returns
    -------
    Array of shape (N, 2, 2), the Jacobian matrices for each parameter pair.
    """
    from scipy.special import polygamma

    psi1ab = polygamma(1, ab.sum(axis=1, keepdims=True))
    fii = polygamma(1, ab) - psi1ab
    fij = -psi1ab[:, 0]
    return np.array([[fii[:, 0], fij], [fij, fii[:, 1]]]).T


def inv_beta_suffstats(
    lnX: Union[np.ndarray, float],
    ln1X: Union[np.ndarray, float],
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Solve for Beta distribution parameters (a, b) given log sufficient statistics.

    The system solves:

        psi(a) - psi(a + b) = lnX
        psi(b) - psi(a + b) = ln1X

    Parameters
    ----------
    lnX
        Logarithm of the expected value of X.
    ln1X
        Logarithm of the expected value of 1 - X.

    Returns
    -------
    a
        Estimated alpha parameter(s) of the Beta distribution.
    b
        Estimated beta parameter(s) of the Beta distribution.

    Raises
    ------
    ValueError
        If the Newton-Raphson projection produces a negative alpha or beta, which
        is not a valid Beta distribution. Previously this was silently warned and
        clamped into a local that was immediately overwritten (a no-op).
    """

    _lnX, _ln1X = np.ravel(lnX), np.ravel(ln1X)
    lnXs = np.c_[_lnX, _ln1X]

    # Find initial starting location
    Gs = np.exp(lnXs)
    dG = 1 - Gs.sum(axis=1, keepdims=True)
    ab = np.maximum(1, (1 + Gs / dG) / 2)

    # 5 Newton Raphson itertions is generally enough
    for i in range(5):
        f = grad_betaln(ab) - lnXs
        jac = jac_grad_betaln(ab)
        ab += np.linalg.solve(jac, - f)

    if np.any(ab < 0):
        raise ValueError(
            "inv_beta_suffstats produced negative Beta parameters "
            f"(alpha, beta):\n\n{ab}\n\n"
            "A negative alpha or beta is not a valid Beta distribution, so the "
            "moment-matching projection has failed. The previous behaviour "
            "silently warned and clamped into a local that was immediately "
            "overwritten (a no-op), letting the invalid parameters escape."
        )

    shape = np.shape(lnX)
    if shape:
        a = ab[:, 0].reshape(shape)
        b = ab[:, 1].reshape(shape)
    else:
        a, b = ab[0, :]

    return a, b


class BetaMessage(AbstractMessage):

    log_base_measure = 0
    _support = ((0, 1),)
    _min = 0
    _max = 1
    _range = 1
    _parameter_support = ((0, np.inf), (0, np.inf))

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        log_norm: float = 0,
        id_: Union[str, None] = None,
    ):
        """
        Represents a Beta distribution message in natural parameter form.

        Parameters
        ----------
        alpha
            Alpha (shape) parameter of the Beta distribution. Default is 0.5.
        beta
            Beta (shape) parameter of the Beta distribution. Default is 0.5.
        log_norm
            Logarithm of normalization constant for message passing. Default is 0.
        id_
            Identifier for the message. Default is None.
        """
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            alpha,
            beta,
            log_norm=log_norm,
            id_=id_
        )

    def value_for(self, unit: float) -> float:
        """
        Map a unit interval value (0 to 1) to a value consistent with the Beta distribution.

        Parameters
        ----------
        unit
            Input value in the unit interval [0, 1].

        Returns
        -------
        float
            Corresponding Beta-distributed value.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplemented()

    def log_partition(self, xp=np) -> np.ndarray:
        """
        Compute the log partition function (log normalization constant) of the Beta distribution.

        Returns
        -------
        The value of the log Beta function, i.e. betaln(alpha, beta).
        """
        from scipy.special import betaln

        return betaln(*self.parameters)

    def natural_parameters(self, xp=np) -> np.ndarray:
        """
        Compute the natural parameters of the Beta distribution.

        Returns
        -------
        np.ndarray
            Natural parameters array [alpha - 1, beta - 1].
        """
        return self.calc_natural_parameters(
            self.alpha,
            self.beta,
            xp=xp
        )

    @staticmethod
    def calc_natural_parameters(
            alpha: Union[float, np.ndarray],
            beta: Union[float, np.ndarray],
            xp=np
    ) -> np.ndarray:
        """
        Calculate the natural parameters of a Beta distribution from alpha and beta.

        Parameters
        ----------
        alpha
            Alpha (shape) parameter(s) of the Beta distribution.
        beta
            Beta (shape) parameter(s) of the Beta distribution.

        Returns
        -------
        Natural parameters [alpha - 1, beta - 1].
        """
        return xp.array([alpha - 1, beta - 1])

    @staticmethod
    def invert_natural_parameters(
            natural_parameters: np.ndarray
    ) -> np.ndarray:
        """
        Convert natural parameters back to standard Beta distribution parameters.

        Parameters
        ----------
        natural_parameters
            Array of natural parameters [alpha - 1, beta - 1].

        Returns
        -------
        Standard Beta parameters [alpha, beta].
        """
        return natural_parameters + 1

    @classmethod
    def invert_sufficient_statistics(
            cls, sufficient_statistics: np.ndarray
    ) -> np.ndarray:
        """
        Estimate natural parameters from sufficient statistics using inverse operations.

        Parameters
        ----------
        sufficient_statistics
            Sufficient statistics (e.g. expectations of log X and log(1 - X)).

        Returns
        -------
        Natural parameters computed from sufficient statistics.
        """
        a, b = inv_beta_suffstats(*sufficient_statistics)
        return cls.calc_natural_parameters(a, b)

    @classmethod
    def to_canonical_form(cls, x: np.ndarray, xp=np) -> np.ndarray:
        """
        Convert a value x in (0,1) to the canonical sufficient statistics for Beta.

        Parameters
        ----------
        x
            Values in the support of the Beta distribution (0 < x < 1).

        Returns
        -------
        Canonical sufficient statistics [log(x), log(1 - x)].
        """
        return xp.array([xp.log(x), xp.log1p(-x)])

    @cached_property
    def mean(self) -> Union[np.ndarray, float]:
        """
        Compute the mean of the Beta distribution.

        Returns
        -------
        Mean value alpha / (alpha + beta).
        """
        return self.alpha / (self.alpha + self.beta)

    @cached_property
    def variance(self) -> Union[np.ndarray, float]:
        """
        Compute the variance of the Beta distribution.

        Returns
        -------
        Variance value of the Beta distribution.
        """
        return (
                self.alpha * self.beta
                / (self.alpha + self.beta) ** 2
                / (self.alpha + self.beta + 1)
        )

    def sample(self, n_samples: int = None) -> np.ndarray:
        """
        Draw samples from the Beta distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw. If None, returns a single sample.

        Returns
        -------
        Samples drawn from Beta(alpha, beta).
        """
        a, b = self.parameters
        shape = (n_samples,) + self.shape if n_samples else self.shape
        return np.random.beta(a, b, size=shape)

    def kl(self, dist: "BetaMessage") -> float:
        """
        Calculate the Kullback-Leibler divergence KL(self || dist).

        Parameters
        ----------
        dist
            The Beta distribution to compare against.

        Returns
        -------
        float
            The KL divergence value.

        Raises
        ------
        TypeError
            If the support of the two distributions does not match.
        """
        from scipy.special import betaln
        from scipy.special import psi

        # Family-wide contract (#1332 F2): kl(self, dist) = KL(self || dist).
        # Previously P was taken from `dist` and Q from `self`, computing the
        # reverse direction KL(dist || self) despite this docstring.
        # Reference: https://arxiv.org/pdf/0911.4863.pdf
        if self._support != dist._support:
            raise TypeError('Support does not match')

        aP, bP = self.parameters
        aQ, bQ = dist.parameters
        return (
                betaln(aQ, bQ) - betaln(aP, bP)
                - (aQ - aP) * psi(aP)
                - (bQ - bP) * psi(bP)
                + (aQ - aP + bQ - bP) * psi(aP + bP)
        )

    def logpdf_gradient(
            self,
            x: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Compute the log probability density function and its gradient at x.

        Parameters
        ----------
        x
            Point(s) in (0, 1) where to evaluate the logpdf and gradient.

        Returns
        -------
        logl
            Log of the PDF evaluated at x.
        gradl
            Gradient of the log PDF at x.
        """
        logl = self.logpdf(x)
        a, b = self.parameters
        gradl = (a - 1) / x + (b - 1) / (x - 1)
        return logl, gradl

    def logpdf_gradient_hessian(
            self,
            x: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Compute the logpdf, its gradient, and Hessian at x.

        Parameters
        ----------
        x
            Point(s) in (0, 1) where to evaluate the logpdf, gradient, and Hessian.

        Returns
        -------
        logl
            Log of the PDF evaluated at x.
        gradl
            Gradient of the log PDF at x.
        hessl
            Hessian (second derivative) of the log PDF at x.
        """
        logl = self.logpdf(x)
        a, b = self.parameters
        ax, bx = (a - 1) / x, (b - 1) / (x - 1)
        gradl = ax + bx
        hessl = -ax / x - bx / (x - 1)
        return logl, gradl, hessl
