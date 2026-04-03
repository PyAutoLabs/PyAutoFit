import numpy as np
from typing import Optional, Tuple

from autofit.messages.normal import UniformNormalMessage
from .abstract import Prior
from .abstract import epsilon
from ...messages.composed_transform import TransformedMessage
from ...messages.transform import LinearShiftTransform

from autofit import exc

class UniformPrior(Prior):
    __identifier_fields__ = ("lower_limit", "upper_limit")
    __database_args__ = ("lower_limit", "upper_limit", "id_")

    def __init__(
        self,
        lower_limit: float = 0.0,
        upper_limit: float = 1.0,
        id_: Optional[int] = None,
    ):
        """
        A prior with a uniform distribution, defined between a lower limit and upper limit.

        The conversion of an input unit value, ``u``, to a physical value, ``p``, via the prior is as follows:

        .. math::

        For example for ``prior = UniformPrior(lower_limit=0.0, upper_limit=2.0)``, an
        input ``prior.value_for(unit=0.5)`` is equal to 1.0.

        [Rich describe how this is done via message]

        Parameters
        ----------
        lower_limit
            The lower limit of the uniform distribution defining the prior.
        upper_limit
            The upper limit of the uniform distribution defining the prior.

        Examples
        --------

        prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.2)
        """

        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)

        if self.lower_limit >= self.upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )

        message = TransformedMessage(
            UniformNormalMessage,
            LinearShiftTransform(shift=self.lower_limit, scale=self.upper_limit - self.lower_limit),
        )
        super().__init__(
            message,
            id_=id_,
        )

    def tree_flatten(self):
        """Flatten this prior into a JAX-compatible PyTree representation.

        Returns
        -------
        tuple
            A (children, aux_data) pair where children are (lower_limit, upper_limit, id).
        """
        return (self.lower_limit, self.upper_limit, self.id), ()

    @property
    def width(self):
        """The width of the uniform distribution (upper_limit - lower_limit)."""
        return self.upper_limit - self.lower_limit

    def with_limits(
        self,
        lower_limit: float,
        upper_limit: float,
    ) -> "Prior":
        """Create a new UniformPrior with different bounds.

        Parameters
        ----------
        lower_limit
            The new lower bound.
        upper_limit
            The new upper bound.
        """
        return UniformPrior(
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

    def logpdf(self, x):
        """Compute the log probability density at x.

        Adjusts boundary values by epsilon to avoid evaluating exactly at
        the distribution edges where the PDF is undefined.

        Parameters
        ----------
        x
            The value at which to evaluate the log PDF.
        """
        # TODO: handle x as a numpy array
        if x == self.lower_limit:
            x += epsilon
        elif x == self.upper_limit:
            x -= epsilon
        return self.message.logpdf(x)

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
    def parameter_string(self) -> str:
        """A human-readable string summarizing the prior's lower and upper limits."""
        return f"lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"

    def value_for(self, unit: float) -> float:
        """
        Returns a physical value from an input unit value according to the limits of the uniform prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.

        Returns
        -------
        value
            The unit value mapped to a physical value according to the prior.

        Examples
        --------

        prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.2)
        """
        return float(
            round(super().value_for(unit), 14)
        )

    def log_prior_from_value(self, value, xp=np):
        """
        Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.

        This is used by certain non-linear searches (e.g. Emcee) in the log likelihood function evaluation.

        For a UniformPrior this is always zero, provided the value is between the lower and upper limit.
        """
        return 0.0

    @property
    def limits(self) -> Tuple[float, float]:
        """The (lower_limit, upper_limit) bounds of this uniform prior."""
        return self.lower_limit, self.upper_limit