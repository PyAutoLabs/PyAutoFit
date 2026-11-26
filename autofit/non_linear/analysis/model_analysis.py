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

    def modify_before_fit(self, paths: AbstractPaths, model: AbstractPriorModel):
        """
        Forward ``modify_before_fit`` to the wrapped analysis, rebind this wrapper to the
        analysis it returns and refresh the wrapper's backend from it.

        The base ``Analysis.modify_before_fit`` (which returns ``self``) would otherwise
        shadow ``__getattr__`` forwarding, so the wrapped analysis's own
        ``modify_before_fit`` (e.g. a PyAutoGalaxy / PyAutoLens ``setup="auto"``
        resolution) would never run.

        The wrapped analysis receives this wrapper's own ``self.model``, not the
        search's ``model``: the wrapper evaluates the wrapped likelihood on instances of
        ``self.model`` (see ``Analysis.with_model``), so that is the model whose
        structure any model-dependent setup must be resolved from. The rebinding mirrors
        ``AnalysisFactor.modify_before_fit`` / ``rebind_analysis``.

        Returns
        -------
        This wrapper, its wrapped analysis now the modified one.
        """
        self._rebind(self.analysis.modify_before_fit(paths=paths, model=self.model))
        return self

    def modify_after_fit(self, paths: AbstractPaths, model: AbstractPriorModel, result):
        """
        Forward ``modify_after_fit`` to the wrapped analysis with this wrapper's own
        ``self.model`` (as ``modify_before_fit`` does), rebinding to the analysis it
        returns.

        Returns
        -------
        This wrapper.
        """
        self._rebind(
            self.analysis.modify_after_fit(paths=paths, model=self.model, result=result)
        )
        return self

    def prepare_for_model(self, model: AbstractPriorModel):
        """
        Forward ``Analysis.prepare_for_model`` to the wrapped analysis with this
        wrapper's own ``self.model`` (the model its likelihood is evaluated on), then
        refresh the wrapper's backend, which the wrapped analysis may only now know.
        """
        self.analysis.prepare_for_model(self.model)
        self._use_jax = self.analysis.is_jax

    @property
    def setup_is_default(self) -> bool:
        """
        The wrapped analysis's ``setup_is_default`` (the base ``Analysis`` property would
        otherwise shadow ``__getattr__`` forwarding).
        """
        return self.analysis.setup_is_default

    def adopt_backend(self, use_jax: bool):
        """
        Forward ``adopt_backend`` to the wrapped analysis, then refresh this wrapper's
        backend from it.
        """
        self.analysis.adopt_backend(use_jax)
        self._use_jax = self.analysis.is_jax

    def _rebind(self, analysis: Optional[Analysis]):
        """
        Point this wrapper at ``analysis`` (when not ``None``) and refresh its
        ``is_jax`` from it, so the wrapper and the analysis it wraps never disagree
        about the backend once the wrapped analysis has resolved its setup.
        """
        if analysis is not None:
            self.analysis = analysis
        self._use_jax = self.analysis.is_jax

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
