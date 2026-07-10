from typing import Optional

import numpy as np

from autofit.messages.normal import NormalMessage
from .abstract import Prior
from ...messages.composed_transform import TransformedMessage
from ...messages.transform import log_transform


class LogGaussianPrior(Prior):
    __identifier_fields__ = ("mean", "sigma")
    __database_args__ = ("mean", "sigma", "id_")

    def __init__(
        self,
        mean: float,
        sigma: float,
        id_: Optional[int] = None,
    ):
        r"""
        A prior for a variable whose logarithm is gaussian distributed. Work in natural log.

        The conversion of an input unit value, ``u``, to a physical value, ``p``, via the prior is as follows:

        .. math::

            p = \mu + (\sigma * sqrt(2) * erfcinv(2.0 * (1.0 - u))

        For example for ``prior = LogGaussianPrior(mean=1.0, sigma=2.0)``, an
        input ``prior.value_for(unit=0.5)`` is equal to 1.0.

        [Rich describe how this is done via message]

        Parameters
        ----------
        mean
            The *natural log* of the distribution's mean.
        sigma
            The spread of this distribution in *natural log* space, e.g. sigma=1.0 means P(ln x) has a
            standard deviation of 1.

        Examples
        --------

        prior = af.LogGaussianPrior(mean=1.0, sigma=2.0)

        physical_value = prior.value_for(unit=0.5)
        """

        self.mean = mean
        self.sigma = sigma

        message = TransformedMessage(
            NormalMessage(mean, sigma),
            log_transform,
        )

        super().__init__(
            message=message,
            id_=id_,
        )

    def tree_flatten(self):
        return (
            self.mean,
            self.sigma,
            self.id,
        ), ()

    @classmethod
    def with_limits(cls, lower_limit: float, upper_limit: float) -> "LogGaussianPrior":
        """
        Create a new gaussian prior centred between two limits
        with sigma distance between this limits.

        Note that these limits are not strict so exceptions will not
        be raised for values outside of the limits.

        This function is typically used in prior passing, where the
        result of a model-fit are used to create new Gaussian priors
        centred on the previously estimated median PDF model.

        Parameters
        ----------
        lower_limit
            The lower limit of the new Gaussian prior.
        upper_limit
            The upper limit of the new Gaussian Prior.

        Returns
        -------
        A new GaussianPrior
        """
        return cls(
            mean=(lower_limit + upper_limit) / 2,
            sigma=upper_limit - lower_limit,
        )

    def _new_for_base_message(self, message):
        """
        Create a new instance of this wrapper but change the parameters used
        to instantiate the underlying message. This is useful for retaining
        the same transform stack after recreating the underlying message during
        projection.
        """
        return LogGaussianPrior(
            *message.parameters,
            id_=self.id,
        )

    def value_for(self, unit, xp=np):
        """
        Return a physical value for a value between 0 and 1 with the transformation
        described by this prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.
        xp
            Array-module to dispatch on (``numpy`` or ``jax.numpy``). Default ``numpy``.
            The NumPy path delegates to the message stack; the JAX path uses the
            closed-form ``exp(mean + sigma * sqrt(2) * erfinv(2*unit - 1))``.

        Returns
        -------
        A physical value, mapped from the unit value accoridng to the prior.
        """
        if xp is np:
            return super().value_for(unit)
        from jax.scipy.special import erfinv
        log_value = self.mean + self.sigma * xp.sqrt(2.0) * erfinv(2.0 * unit - 1.0)
        return xp.exp(log_value)

    @property
    def parameter_string(self) -> str:
        return f"mean = {self.mean}, sigma = {self.sigma}"

    def log_normalisation(self, xp=np) -> float:
        """The constant ``-log(sigma) - 0.5*log(2*pi)`` dropped from the Gaussian-in-log
        density in ``log_prior_from_value`` (the value-dependent ``-log(value)``
        change-of-variables Jacobian is kept). See ``Prior.log_normalisation``."""
        return -xp.log(self.sigma) - 0.5 * xp.log(2.0 * np.pi)

    def log_prior_from_value(self, value, xp=np):
        """
        Compute the log prior density of a given physical value under this log-Gaussian prior.

        The change-of-variables Jacobian for the log transform contributes
        ``-log(value)``; the underlying Gaussian-in-log-space contributes the
        density-form quadratic via ``NormalMessage.log_prior_from_value``.
        Out-of-support (``value <= 0``) returns ``-inf``.

        See ``NormalMessage.log_prior_from_value`` for the constant-dropping
        convention.
        """
        if xp is np:
            if value <= 0:
                return float("-inf")
            return self.message.base_message.log_prior_from_value(
                np.log(value)
            ) - np.log(value)

        log_value = xp.log(value)
        base_log_prior = -((log_value - self.mean) ** 2) / (2 * self.sigma ** 2)
        return xp.where(value > 0, base_log_prior - log_value, -xp.inf)
