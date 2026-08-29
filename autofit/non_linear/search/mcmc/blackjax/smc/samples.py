from typing import Dict, List, Optional
import warnings

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.non_linear.samples.sample import Sample

from autofit import exc


class SamplesSMC(SamplesPDF):
    def __init__(
        self,
        model: AbstractPriorModel,
        sample_list: List[Sample],
        samples_info: Optional[Dict] = None,
    ):
        """
        The samples of a Sequential Monte Carlo (SMC) search -- the final tempered particle cloud, each particle
        carrying its normalised importance weight.

        SMC is a *weighted* sampler, so this subclasses `SamplesPDF` (which computes medians, errors and PDFs from
        the weights) rather than `SamplesMCMC` (which assumes unweighted draws from a chain and derives its
        diagnostics from auto-correlation times). The properties added below expose the tempering diagnostics that
        an SMC run must be judged by -- see `log_evidence` and `lambda_schedule`.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Sample`s, one per particle, containing parameters, log likelihood, log prior and weight.
        samples_info
            Information on the samples (the tempering schedule, per-step acceptance, ESS, log evidence, run time).
        """

        super().__init__(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )

    def __add__(self, other: "SamplesSMC") -> "SamplesSMC":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        Parameters
        ----------
        other
            Another Samples class
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
        )

    @property
    def log_evidence(self) -> Optional[float]:
        """
        The log evidence `log Z = log integral(prior(x) L(x) dx)`, accumulated for free from the SMC tempering
        bridge as the sum of every step's `log_likelihood_increment` (plus the whitening Jacobian `log|det L|`
        when the run was warm-started; see `SMC._fit`).

        This is the true evidence on the same scale a nested sampler reports, and is directly comparable to a
        `Nautilus` or `Dynesty` `log_evidence` for the same model and data.

        **Do not read it without reading `acceptance_rate_list` first.** Adaptive tempering can force a single
        huge jump straight to `lambda = 1` when acceptance has collapsed to ~0, which terminates the run and
        reports a meaningless `log_evidence` while `converged` is still `True`.
        """
        return self.samples_info.get("log_evidence")

    @property
    def lambda_list(self) -> Optional[List[float]]:
        """
        The adaptive tempering schedule: the inverse temperature `lambda` reached after each SMC step, from the
        first step through to `1.0`. A schedule of only one or two steps on a sharply peaked likelihood means the
        run jumped rather than annealed, and its `log_evidence` cannot be trusted.
        """
        return self.samples_info.get("lambda_list")

    @property
    def acceptance_rate_list(self) -> Optional[List[float]]:
        """
        The mean inner-kernel (MALA / HMC) acceptance rate across particles at each SMC step. This, together with
        the max-log-likelihood progression, is the criterion an SMC run is judged by -- a trace that sits at ~0 is
        a failed run whatever `converged` says.
        """
        return self.samples_info.get("acceptance_rate_list")

    @property
    def ess_list(self) -> Optional[List[float]]:
        """
        The realised effective sample size of the particle weights after each SMC step. Adaptive tempering picks
        each `lambda` increment to hit `target_ess * num_particles`, so this should track that target closely.
        """
        return self.samples_info.get("ess_list")

    @property
    def converged(self) -> bool:
        """
        Whether the tempering path reached `lambda = 1.0` (rather than stopping at `max_smc_steps`).

        This is a **weak** criterion on its own: it only means `lambda` reached 1.0, which adaptive tempering can
        do in a single forced jump when acceptance has collapsed. Judge the run by `acceptance_rate_list` and the
        max-log-likelihood progression.
        """
        return bool(self.samples_info.get("converged", False))
