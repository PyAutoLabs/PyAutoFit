"""
``ModelAnalysis`` (``analysis.with_model(model)``) forwards the pre-fit hooks to the
analysis it wraps: ``modify_before_fit`` (rebinding to the analysis returned and
refreshing ``is_jax``), ``modify_after_fit`` and ``prepare_for_model``, each with the
wrapper's own model. Also inside an ``AnalysisFactor``.
"""

import pytest

import autofit as af
import autofit.graphical as g
from autofit.non_linear.analysis.model_analysis import ModelAnalysis
from autofit.non_linear.paths.null import NullPaths

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class ResolvingAnalysis(af.Analysis):
    """
    Stands in for a ``setup="auto"`` analysis: its backend is only known once
    ``modify_before_fit`` / ``prepare_for_model`` has seen a model.
    """

    def __init__(self, return_new=False):
        super().__init__(use_jax=False)
        self.return_new = return_new
        self.before_models = []
        self.prepared_models = []
        self.after_models = []

    def log_likelihood_function(self, instance):
        return 0.0

    def _resolve(self):
        # Flip the backend flag directly so the test needs no JAX install.
        self._use_jax = True

    def modify_before_fit(self, paths, model):
        self.before_models.append(model)
        if self.return_new:
            new = ResolvingAnalysis()
            new._resolve()
            new.before_models = self.before_models
            return new
        self._resolve()
        return self

    def prepare_for_model(self, model):
        self.prepared_models.append(model)
        self._resolve()

    def modify_after_fit(self, paths, model, result):
        self.after_models.append(model)
        return self


@pytest.fixture(name="explicit_model")
def make_explicit_model():
    return af.Model(af.ex.Gaussian)


@pytest.fixture(name="search_model")
def make_search_model():
    return af.Collection(gaussian=af.Model(af.ex.Gaussian))


def test_modify_before_fit_is_forwarded_and_returns_the_wrapper(
    explicit_model, search_model
):
    inner = ResolvingAnalysis()
    wrapper = inner.with_model(explicit_model)
    assert wrapper.is_jax is False

    returned = wrapper.modify_before_fit(paths=NullPaths(), model=search_model)

    assert returned is wrapper
    assert isinstance(returned, ModelAnalysis)
    # The wrapped analysis sees the model its likelihood is evaluated on.
    assert inner.before_models == [explicit_model]
    assert wrapper.analysis is inner
    assert wrapper.is_jax is True


def test_modify_before_fit_rebinds_to_a_new_inner_analysis(explicit_model):
    inner = ResolvingAnalysis(return_new=True)
    wrapper = inner.with_model(explicit_model)

    returned = wrapper.modify_before_fit(paths=NullPaths(), model=explicit_model)

    assert returned is wrapper
    assert wrapper.analysis is not inner
    assert isinstance(wrapper.analysis, ResolvingAnalysis)
    assert wrapper.is_jax is True


def test_prepare_for_model_is_forwarded_with_the_wrapper_model(
    explicit_model, search_model
):
    inner = ResolvingAnalysis()
    wrapper = inner.with_model(explicit_model)

    wrapper.prepare_for_model(search_model)

    assert inner.prepared_models == [explicit_model]
    assert wrapper.is_jax is True


def test_modify_after_fit_is_forwarded(explicit_model):
    inner = ResolvingAnalysis()
    wrapper = inner.with_model(explicit_model)

    returned = wrapper.modify_after_fit(
        paths=NullPaths(), model=explicit_model, result=None
    )

    assert returned is wrapper
    assert inner.after_models == [explicit_model]


def test_base_prepare_for_model_is_a_no_op(explicit_model):
    analysis = af.Analysis()
    assert analysis.prepare_for_model(explicit_model) is None
    assert analysis.is_jax is False


def test_analysis_factor_over_a_model_analysis_resolves_the_inner_analysis(
    explicit_model,
):
    inner = ResolvingAnalysis(return_new=True)
    wrapper = inner.with_model(explicit_model)
    factor = g.AnalysisFactor(explicit_model, analysis=wrapper, name="f0")

    returned = factor.modify_before_fit(paths=NullPaths(), model=explicit_model)

    assert returned is factor
    # The factor keeps the (same) wrapper, whose inner analysis is the new one.
    assert factor.analysis is wrapper
    assert wrapper.analysis is not inner
    assert inner.before_models == [explicit_model]
    assert factor.is_jax is True
