import numpy as np

from autoconf import cached_property
from autofit.messages.abstract import AbstractMessage
from autofit.messages.utils import invpsilog


class GammaMessage(AbstractMessage):

    def log_partition(self, xp=np):
        from scipy import special

        alpha, beta = GammaMessage.invert_natural_parameters(self.natural_parameters(xp=xp))
        return special.gammaln(alpha) - alpha * np.log(beta)

    log_base_measure = 0.0
    _support = ((0, np.inf),)
    _parameter_support = ((0, np.inf), (0, np.inf))

    def __init__(
            self,
            alpha=1.0,
            beta=1.0,
            log_norm=0.0,
            id_=None
    ):
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            alpha,
            beta,
            log_norm=log_norm,
            id_=id_
        )

    def value_for(self, unit: float) -> float:
        raise NotImplemented()

    def natural_parameters(self, xp=np) -> np.ndarray:
        return self.calc_natural_parameters(self.alpha, self.beta, xp=xp)

    @staticmethod
    def calc_natural_parameters(alpha, beta, xp=np):
        return xp.array([alpha - 1, -beta])

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        eta1, eta2 = natural_parameters
        return eta1 + 1, -eta2

    @staticmethod
    def to_canonical_form(x, xp=np):
        return xp.array([np.log(x), x])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        logX, X = suff_stats
        alpha = invpsilog(logX - np.log(X))
        beta = alpha / X
        return cls.calc_natural_parameters(alpha, beta)

    @cached_property
    def mean(self):
        return self.alpha / self.beta

    @cached_property
    def variance(self):
        return self.alpha / self.beta ** 2

    def sample(self, n_samples=None):
        a1, b1 = self.parameters
        shape = (n_samples,) + self.shape if n_samples else self.shape
        return np.random.gamma(a1, scale=1 / b1, size=shape)

    @classmethod
    def from_mode(cls, mode, covariance, **kwargs):
        m, V = cls._get_mean_variance(mode, covariance)

        # Match the requested mean m and variance V, consistent with the Normal
        # family. For a Gamma, mean = alpha / beta and variance = alpha / beta**2,
        # which inverts to alpha = m**2 / V, beta = m / V. The previous
        # ``alpha = 1 + m**2 * V`` had the variance the wrong way up (requesting
        # variance 0.25 produced 2.0, and 4.0 produced 0.235).
        alpha = m ** 2 / V
        beta = m / V
        return cls(alpha, beta, **kwargs)

    def kl(self, dist):
        from scipy import special

        P, Q = dist, self
        logP = np.log(P.alpha)
        # TODO check this is correct
        # https://arxiv.org/pdf/0911.4863.pdf
        return (
                (P.alpha - Q.alpha) * special.psi(P.alpha)
                - special.gammaln(P.alpha)
                + special.gammaln(Q.alpha)
                + Q.alpha * (np.log(P.beta / Q.beta))
                + P.alpha * (Q.beta / P.beta - 1)
        )

    def logpdf_gradient(self, x):
        logl = self.logpdf(x)
        eta1 = self.natural_parameters()[0]
        gradl = eta1 / x - self.beta
        return logl, gradl

    def logpdf_gradient_hessian(self, x):
        logl = self.logpdf(x)
        eta1 = self.natural_parameters()[0]
        gradl = eta1 / x
        hessl = -gradl / x
        gradl -= self.beta
        return logl, gradl, hessl
