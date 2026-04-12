import numpy as np
import os

from typing import Optional

from autofit.non_linear.test_mode import is_test_mode

class AutoCorrelationsSettings:

    def __init__(
            self,
            check_for_convergence: bool = True,
            check_size: int = 100,
            required_length: int = 50,
            change_threshold: float = 0.01,
    ):
        """
        Class for performing and customizing AutoCorrelation calculations, which are used:

         - By the `Samples` object during a model-fit to determine is an ensemble MCMC sampler should terminate.
         - After a model-fit is finished to investigate whether the resutls converged.

        Parameters
        ----------
        check_for_convergence
            Whether the auto-correlation lengths of the Emcee samples are checked to determine the stopping criteria.
            If `True`, this option may terminate the Emcee run before the input number of steps, nsteps, has
            been performed. If `False` nstep samples will be taken.
        check_size
            The length of the samples used to check the auto-correlation lengths (from the latest sample backwards).
            For convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
            thus requires more samples meet the auto-correlation threshold, taking longer to terminate sampling.
            However, shorter chains risk stopping sampling early due to noise.
        required_length
            The length an auto_correlation chain must be for it to be used to evaluate whether its change threshold is
            sufficiently small to terminate sampling early.
        change_threshold
            The threshold value by which if the change in auto_correlations is below sampling will be terminated early.
        """
        self.check_for_convergence = check_for_convergence
        self.check_size = check_size
        self.required_length = required_length
        self.change_threshold = change_threshold

        if is_test_mode():
            self.check_size = 1


class AutoCorrelations(AutoCorrelationsSettings):

    def __init__(
            self,
            check_size,
            required_length,
            change_threshold,
            times,
            previous_times,

    ):
        """
        Class for performing and customizing AutoCorrelation calculations, which are used:

         - By the `Samples` object during a model-fit to determine is an ensemble MCMC sampler should terminate.
         - After a model-fit is finished to investigate whether the resutls converged.

        Parameters
        ----------
        check_for_convergence
            Whether the auto-correlation lengths of the Emcee samples are checked to determine the stopping criteria.
            If `True`, this option may terminate the Emcee run before the input number of steps, nsteps, has
            been performed. If `False` nstep samples will be taken.
        check_size
            The length of the samples used to check the auto-correlation lengths (from the latest sample backwards).
            For convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
            thus requires more samples meet the auto-correlation threshold, taking longer to terminate sampling.
            However, shorter chains risk stopping sampling early due to noise.
        required_length
            The length an auto_correlation chain must be for it to be used to evaluate whether its change threshold is
            sufficiently small to terminate sampling early.
        change_threshold
            The threshold value by which if the change in auto_correlations is below sampling will be terminated early.
        """

        super().__init__(
            check_size=check_size,
            required_length=required_length,
            change_threshold=change_threshold
        )

        self.times = times
        self.previous_times = previous_times

    @property
    def relative_times(self) -> [float]:
        return np.abs(self.previous_times - self.times) / self.times

    def check_if_converged(self, total_samples):
        """
        Whether the emcee samples have converged on a solution or if they are still in a burn-in period, based on the
        auto correlation times of parameters.
        """

        if self.times is None or self.previous_times is None:
            return False

        converged = np.all(
            self.times * self.required_length
            < total_samples
        )
        if converged:
            try:
                converged &= np.all(self.relative_times < self.change_threshold)
            except IndexError:
                return False
        return converged
