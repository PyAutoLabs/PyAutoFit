import math
import numpy as np
from typing import Tuple

"""
The `Gaussian` class in this module is the model components that is fitted to data using a non-linear search. The
inputs of its __init__ constructor are the parameters which can be fitted for.

The log_likelihood_function in the Analysis class receives an instance of this classes where the values of its
parameters have been set up according to the non-linear search. Because instances of the classes are used, this means
their methods (e.g. model_data_from) can be used in the log likelihood function.
"""


class Gaussian:
    def __init__(
        self,
        centre: float = 0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization: float = 0.1,  # <- are the Gaussian`s model parameters.
        sigma: float = 0.01,
    ):
        """
        Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    @property
    def fwhm(self) -> float:
        """
        The full-width half-maximum of the Gaussian profile.

        This is used to illustrate latent variables in **PyAutoFit**, which are values that can be inferred from
        the free parameters of the model which we are interested and may want to store the full samples information
        on (e.g. to create posteriors).
        """
        return 2 * np.sqrt(2 * np.log(2)) * self.sigma

    def _tree_flatten(self):
        return (self.centre, self.normalization, self.sigma), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return Gaussian(*children)

    def __eq__(self, other):
        return (
            isinstance(other, Gaussian)
            and self.centre == other.centre
            and self.normalization == other.normalization
            and self.sigma == other.sigma
        )

    def model_data_from(self, xvalues: np.ndarray, xp=np) -> np.ndarray:
        """
        Calculate the normalization of the profile on a 1D grid of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = xvalues - self.centre

        return xp.multiply(
            xp.divide(self.normalization, self.sigma * xp.sqrt(2.0 * xp.pi)),
            xp.exp(-0.5 * xp.square(xp.divide(transformed_xvalues, self.sigma))),
        )

    def f(self, x: float, xp=np):
        return (
            self.normalization
            / (self.sigma * xp.sqrt(2 * math.pi))
            * xp.exp(-0.5 * ((x - self.centre) / self.sigma) ** 2)
        )

    def __call__(self, xvalues: np.ndarray) -> np.ndarray:
        """
        For certain graphical models, the `__call__` function is overwritten for producing the model-fit.
        We include this here so these examples work, but it should not be important for most PyAutoFit users.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the grid.
        """
        return self.model_data_from(xvalues=xvalues)

    def inverse(self, y):
        """
        For graphical models, the inverse of the Gaussian is used to test certain aspects of the calculation.
        """

        a = self.normalization / (y * self.sigma * math.sqrt(2 * math.pi))

        b = 2 * math.log(a)

        return self.centre + self.sigma * math.sqrt(b)


class Exponential:
    def __init__(
        self,
        centre: float = 0.0,  # <- PyAutoFit recognises these constructor arguments are the model
        normalization: float = 0.1,  # <- parameters of the Gaussian.
        rate: float = 0.01,
    ):
        """
        Represents a 1D Exponential profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a `NonLinearSearch`.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        rate
            The decay rate controlling has fast the Exponential declines.
        """
        self.centre = centre
        self.normalization = normalization
        self.rate = rate

    def model_data_from(self, xvalues: np.ndarray, xp=np) -> np.ndarray:
        """
        Calculate the 1D Gaussian profile on a 1D grid of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = xp.subtract(xvalues, self.centre)
        return self.normalization * xp.multiply(
            self.rate, xp.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )

    def __call__(self, xvalues: np.ndarray) -> np.ndarray:
        """
        Calculate the 1D Gaussian profile on a 1D grid of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values
            The x coordinates in the original reference frame of the grid.
        """
        return self.model_data_from(xvalues=xvalues)


class PhysicalNFW:
    def __init__(
        self,
        centre: Tuple[float, float],
        ell_comps: Tuple[float, float],
        log10m: float,
        concentration: float,
    ):
        self.centre = centre
        self.ell_comps = ell_comps
        self.log10m = log10m
        self.concentration = concentration
