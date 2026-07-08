from typing import Optional, Tuple

import numpy as np

from autofit.messages.normal import UniformNormalMessage
from autofit.messages.transform import log_10_transform, LinearShiftTransform
from .abstract import Prior
from ...messages.composed_transform import TransformedMessage

from autofit import exc

class LogUniformPrior(Prior):
    __identifier_fields__ = ("lower_limit", "upper_limit")
    __database_args__ = ("lower_limit", "upper_limit", "id_")

    def __init__(
        self,
        lower_limit: float = 1e-6,
        upper_limit: float = 1.0,
        id_: Optional[int] = None,
    ):
        """
        A prior with a log base 10 uniform distribution, defined between a lower limit and upper limit.

        The conversion of an input unit value, ``u``, to a physical value, ``p``, via the prior is as follows:

        .. math::

        For example for ``prior = LogUniformPrior(lower_limit=10.0, upper_limit=1000.0)``, an
        input ``prior.value_for(unit=0.5)`` is equal to 100.0.

        [Rich describe how this is done via message]

        Parameters
        ----------
        lower_limit
            The lower limit of the log10 uniform distribution defining the prior.
        upper_limit
            The upper limit of the log10 uniform distribution defining the prior.

        Examples
        --------

        prior = af.LogUniformPrior(lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.2)
        """

        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)

        if self.lower_limit <= 0.0:
            raise exc.PriorException(
                "The lower limit of a LogUniformPrior cannot be zero or negative."
            )
        if self.lower_limit >= self.upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )

        message = TransformedMessage(
            UniformNormalMessage,
            LinearShiftTransform(
                shift=np.log10(self.lower_limit),
                scale=np.log10(self.upper_limit / self.lower_limit),
            ),
            log_10_transform,
        )

        super().__init__(
            message=message,
            id_=id_,
        )

    def tree_flatten(self):
        return (
            self.lower_limit,
            self.upper_limit,
            self.id,
        ), ()

    @classmethod
    def with_limits(cls, lower_limit: float, upper_limit: float) -> "LogUniformPrior":
        """
        Create a new log 10 uniform prior centred between two limits
        with sigma distance between this limits.

        Note that these limits are not strict so exceptions will not
        be raised for values outside of the limits.

        This function is typically used in prior passing, where the
        result of a model-fit are used to create new Gaussian priors
        centred on the previously estimated median PDF model.

        Parameters
        ----------
        lower_limit
            The lower limit of the new LogUniform prior.
        upper_limit
            The upper limit of the new LogUniform Prior.

        Returns
        -------
        A new LogUniform.
        """
        return cls(
            lower_limit=max(0.000001, lower_limit),
            upper_limit=upper_limit,
        )

    __identifier_fields__ = ("lower_limit", "upper_limit")

    def log_prior_from_value(self, value, xp=np):
        """
        Returns the log prior density at a physical value, used by Emcee / Zeus
        / MLE searches to form a log-posterior via ``log_likelihood + sum(log_priors)``.

        For a log-uniform prior on ``[lower_limit, upper_limit]`` the density is
        ``p(x) = 1 / (x * log(upper_limit / lower_limit))``, giving
        ``log p(x) = -log(x) - log(log(upper_limit / lower_limit))``. The
        normalisation constant ``-log(log(upper_limit / lower_limit))`` is
        dropped (it is irrelevant to posterior shape), matching the convention
        used by ``UniformPrior.log_prior_from_value`` which drops ``-log(b - a)``
        to return ``0.0``.

        Non-positive ``value`` (``value <= 0``) returns ``-inf``. Emcee's stretch
        move proposes physical values that can leave the support and go
        non-positive; ``-log`` of a non-positive value is ``NaN``, which propagates
        into the summed figure-of-merit and crashes the search with
        ``ValueError: Probability function returned NaN``. Returning ``-inf``
        (zero density -> rejected move) keeps the figure-of-merit finite. The
        "double where" pattern (a safe surrogate inside the ``log``) ensures no
        ``log`` of a non-positive value is evaluated, avoiding NumPy
        ``RuntimeWarning``s.

        The NumPy path is otherwise unnormalised and unbounded: for any positive
        ``value`` it returns ``-log(value)`` regardless of ``[lower_limit,
        upper_limit]`` (dropping the normalisation constant, matching
        ``UniformPrior``'s convention of returning ``0.0``). The JAX path
        additionally returns ``-inf`` outside ``[lower_limit, upper_limit]``.

        Parameters
        ----------
        value
            The physical value of this prior's corresponding parameter in a
            ``NonLinearSearch`` sample.
        xp
            Array-module to dispatch on (``numpy`` or ``jax.numpy``). Default ``numpy``.
        """
        if xp is np:
            positive = value > 0.0
            return xp.where(positive, -xp.log(xp.where(positive, value, 1.0)), -xp.inf)
        in_bounds = (value >= self.lower_limit) & (value <= self.upper_limit)
        return xp.where(in_bounds, -xp.log(value), -xp.inf)

    def value_for(self, unit, xp=np):
        """
        Returns a physical value from an input unit value according to the limits of the log10 uniform prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.
        xp
            Array-module to dispatch on (``numpy`` or ``jax.numpy``). Default ``numpy``.
            The NumPy path delegates to the message stack (scipy-backed); the JAX
            path uses the closed-form ``lower * (upper / lower) ** unit``.

        Returns
        -------
        value
            The unit value mapped to a physical value according to the prior.

        Examples
        --------

        prior = af.LogUniformPrior(lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.2)
        """
        if xp is np:
            return super().value_for(unit)
        return self.lower_limit * (self.upper_limit / self.lower_limit) ** unit

    def dict(self) -> dict:
        """
        Return a dictionary representation of this GaussianPrior instance,
        including mean and sigma.

        Returns
        -------
        Dictionary containing prior parameters.
        """
        prior_dict = super().dict()
        return {**prior_dict, "lower_limit": self.lower_limit, "upper_limit": self.upper_limit}

    @property
    def limits(self) -> Tuple[float, float]:
        return self.lower_limit, self.upper_limit

    @property
    def parameter_string(self) -> str:
        return f"lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"
