import numpy as np

from scipy import stats

import autofit as af


def _gaussian(x, centre, normalization, sigma):
    return Gaussian(centre=centre, normalization=normalization, sigma=sigma)(x)


_norm = stats.norm(loc=0, scale=1.0)


# TODO: use autofit likelihood
def _likelihood(z, y):
    return np.multiply(-0.5, np.square(np.subtract(z, y)))


class Profile:
    def __init__(self, centre=0.0, normalization=0.01):
        """Represents an Abstract 1D profile.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the profile.
        """
        self.centre = centre
        self.normalization = normalization


class Gaussian(Profile):
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian's model parameters.
        sigma=0.01,
    ):
        """Represents a 1D Gaussian profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the Gaussian profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        super().__init__(centre=centre, normalization=normalization)
        self.sigma = sigma  # We still need to set sigma for the Gaussian, of course.

    def __call__(self, xvalues):
        """
        Calculate the normalization of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


def make_data(gaussian, x):
    model_line = gaussian(xvalues=x)
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, len(x))
    y = model_line + noise
    return y


class Analysis(af.Analysis):
    def __init__(self, x, y, sigma=0.04):
        self.x = x
        self.y = y
        self.sigma = sigma

        super().__init__()

    def log_likelihood_function(self, instance: Gaussian) -> np.array:
        """
        This function takes an instance created by the Model and computes the
        likelihood that it fits the data.
        """
        y_model = instance(self.x)
        return np.sum(_likelihood(y_model, self.y) / self.sigma**2)
