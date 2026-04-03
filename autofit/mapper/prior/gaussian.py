import numpy as np
from typing import Optional

from autofit.messages.normal import NormalMessage
from .abstract import Prior


class GaussianPrior(Prior):
    __identifier_fields__ = ("mean", "sigma")
    __database_args__ = ("mean", "sigma", "id_")

    def __init__(
        self,
        mean: float,
        sigma: float,
        id_: Optional[int] = None,
    ):
        """
        A Gaussian prior defined by a normal distribution.

        The prior transforms a unit interval input `u` in [0, 1] into a physical parameter `p` via
        the inverse error function (erfcinv) based on the Gaussian CDF:

        .. math::
            p = \mu + \sigma \sqrt{2} \, \mathrm{erfcinv}(2 \times (1 - u))

        where :math:`\mu` is the mean and :math:`\sigma` the standard deviation.

        For example, with `mean=1.0` and `sigma=2.0`, the value at `u=0.5` corresponds to the mean, 1.0.

        This mapping is implemented using a NormalMessage instance, encapsulating
        the Gaussian distribution and any specified truncation limits.

        Parameters
        ----------
        mean
            The mean (center) of the Gaussian prior distribution.
        sigma
            The standard deviation (spread) of the Gaussian prior.
        id_ : Optional[int], optional
            Optional identifier for the prior instance.

        Examples
        --------
        Create a GaussianPrior with mean 1.0, sigma 2.0, truncated between 0.0 and 2.0:

        >>> prior = GaussianPrior(mean=1.0, sigma=2.0)
        >>> physical_value = prior.value_for(unit=0.5)  # Returns ~1.0 (mean)
        """

        super().__init__(
            message=NormalMessage(
                mean=mean,
                sigma=sigma,
            ),
            id_=id_,
        )

    def tree_flatten(self):
        """Flatten this prior into a JAX-compatible PyTree representation.

        Returns
        -------
        tuple
            A (children, aux_data) pair where children are (mean, sigma, id).
        """
        return (self.mean, self.sigma, self.id), ()

    @classmethod
    def with_limits(cls, lower_limit: float, upper_limit: float) -> "GaussianPrior":
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

    def dict(self) -> dict:
        """
        Return a dictionary representation of this GaussianPrior instance,
        including mean and sigma.

        Returns
        -------
        Dictionary containing prior parameters.
        """
        prior_dict = super().dict()
        return {**prior_dict, "mean": self.mean, "sigma": self.sigma}

    @property
    def parameter_string(self) -> str:
        """
        Return a human-readable string summarizing the GaussianPrior parameters.
        """
        return f"mean = {self.mean}, sigma = {self.sigma}"
