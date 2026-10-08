from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from .analysis import Analysis
from ... import SamplesSummary, AbstractPaths, SamplesPDF


class ModelAnalysis(Analysis):
    def __init__(
        self,
        analysis: Analysis,
        model: AbstractPriorModel,
        use_jax: Optional[bool] = None,
    ):
        """
        Comprises a model and an analysis that can be applied to instances of that model.

        Parameters
        ----------
        analysis
            The wrapped analysis.
        model
            The model whose instances the wrapped analysis is evaluated on.
        use_jax
            ``None`` (the default) forwards the wrapped analysis's ``is_jax``, so the
            wrapper and the analysis it wraps can never disagree about the backend. An
            explicit bool is honoured as before.
        """
        if use_jax is None:
            use_jax = analysis.is_jax

        super().__init__(use_jax=use_jax)

        self.analysis = analysis
        self.model = model

    @property
    def gradient_mode(self) -> str:
        """
        The wrapped analysis's declared gradient mode (``Analysis.gradient_mode``), so a
        gradient search fitting the wrapper differentiates the way the analysis asked.
        """
        return self.analysis.gradient_mode

    def __getattr__(self, item):
        if item in ("__getstate__", "__setstate__"):
            raise AttributeError(item)
        return getattr(self.analysis, item)

    def log_likelihood_function(self, instance):
        return self.analysis.log_likelihood_function(instance)

    def make_result(
        self,
        samples_summary: SamplesSummary,
        paths: AbstractPaths,
        samples: Optional[SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ):
        """
        Return the correct type of result by calling the underlying analysis.
        """
        try:
            return self.analysis.make_result(
                samples_summary=samples_summary,
                paths=paths,
                samples=samples,
                search_internal=search_internal,
            )
        except TypeError:
            raise
