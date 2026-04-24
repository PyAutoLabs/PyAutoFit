import math
import numpy as np
from typing import Dict, List, Optional
import warnings

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.non_linear.samples.samples import Sample

from autofit.non_linear.samples.samples import to_instance

from autofit import exc


class SamplesMCMC(SamplesPDF):
    def __init__(
        self,
        model: AbstractPriorModel,
        sample_list: List[Sample],
        samples_info: Optional[Dict] = None,
        auto_correlation_settings: Optional[AutoCorrelationsSettings] = None,
        auto_correlations: Optional[AutoCorrelations] = None,
    ):
        """
        Contains the samples of the non-linear search, including parameter values, log likelihoods,
        weights and other quantites.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        Attributes
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        auto_correlation_settings
            Customizes and performs auto correlation calculations performed during and after the search.
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        """

        self.auto_correlations = auto_correlations
        self.auto_correlation_settings = auto_correlation_settings

        super().__init__(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )

    def __add__(self, other: "SamplesMCMC") -> "SamplesMCMC":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        Parameters
        ----------
        other
            Another Samples class

        Returns
        -------
        A class that combined the samples of the two Samples objects.
        """

        self._check_addition(other=other)

        warnings.warn(
            f"Addition of {self.__class__.__name__} cannot retain results in native format. "
            "Visualization of summed samples diabled.",
            exc.SamplesWarning,
        )

        return self.__class__(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
            samples_info=self.samples_info,
            auto_correlation_settings=self.auto_correlation_settings,
        )

    @classmethod
    def from_list_info_and_model(
        cls,
        sample_list,
        samples_info,
        model: AbstractPriorModel,
    ):
        try:
            auto_correlation_settings = AutoCorrelationsSettings(
                check_for_convergence=True,
                check_size=samples_info["check_size"],
                required_length=samples_info["required_length"],
                change_threshold=samples_info["change_threshold"],
            )
        except (KeyError, NameError):
            auto_correlation_settings = None

        return cls(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
            auto_correlation_settings=auto_correlation_settings,
        )

    @property
    def pdf_converged(self):
        """
        To analyse and visualize samples, the analysis must be sufficiently converged to produce smooth enough PDF
        for analysis.

        This property checks whether the non-linear search's samples are sufficiently converged for analysis and
        visualization.

        Emcee samples can be analysed irrespective of how long the sampler has run, albeit low run times will likely
        produce inaccurate results.
        """
        try:
            if len(self.parameter_lists) == 0:
                return False
            return True
        except ValueError:
            return False

    @property
    def converged(self) -> bool:
        """
        Whether the emcee samples have converged on a solution or if they are still in a burn-in period, based on the
        auto correlation times of parameters.
        """
        return self.auto_correlations.check_if_converged(
            total_samples=self.total_samples
        )

    @to_instance
    def median_pdf(self, as_instance: bool = True) -> [float]:
        """
        The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        This is computed by binning all sampls after burn-in into a histogram and take its median (e.g. 50%) value.
        """

        if self.pdf_converged:
            return [
                float(np.percentile(self.parameters_extract[i, :], 50))
                for i in range(self.model.prior_count)
            ]

        return self.max_log_likelihood(as_instance=False)

    @to_instance
    def values_at_sigma(self, sigma: float) -> [float]:
        """
        The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For Emcee, if the samples have converged this is estimated by binning the samples after burn-in into a
        histogram and taking the parameter values at the input PDF %.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            samples = self.parameters_extract

            return [
                tuple(
                    np.percentile(samples[i, :], [100.0 * (1.0 - limit), 100.0 * limit])
                )
                for i in range(self.model.prior_count)
            ]

        parameters_min = list(
            np.min(self.parameter_lists[-self.unconverged_sample_size :], axis=0)
        )
        parameters_max = list(
            np.max(self.parameter_lists[-self.unconverged_sample_size :], axis=0)
        )

        return [
            (parameters_min[index], parameters_max[index])
            for index in range(len(parameters_min))
        ]

    @property
    def total_steps(self) -> int:
        return self.samples_info["total_steps"]

    @property
    def total_walkers(self) -> int:
        return self.samples_info["total_walkers"]

    @property
    def log_evidence(self):
        return None
