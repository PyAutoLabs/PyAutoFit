"""
``af.NSS`` on the shared ``Fitness`` (search-extensibility A3b; survey 02 §3.3, §5.5).

NSS used to hand blackjax an inline closure of its own, so it missed what ``Fitness``
gives every other search:

- (b) a model **assertion** raised ``TracerBoolConversionError`` under ``jax.jit``,
  because the closure built instances with the raising check; ``Fitness.call`` applies
  the assertions as a traced ``xp.where`` penalty instead;
- (c) its ``Fitness`` was built after sampling, so the resume likelihood sanity check
  ran at the end of a resumed run rather than the start.

NSS is ``jax_use='required'``, so the file skips whole without JAX / blackjax.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import autofit as af
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.search.nest.nss import search as nss_search_module
from autofit.non_linear.search.nest.nss.search import NSS_INVALID_LOG_LIKELIHOOD

pytestmark = [
    pytest.mark.skipif(
        not nss_search_module._HAS_NSS, reason="requires blackjax >= 1.6"
    ),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]


class _GaussianJaxAnalysis(af.Analysis):
    def __init__(self):
        super().__init__(use_jax=True)

    def log_likelihood_function(self, instance):
        import jax.numpy as jnp

        return -0.5 * (
            (instance.centre - 50.0) ** 2
            + (instance.normalization - 25.0) ** 2
            + (instance.sigma - 10.0) ** 2
        ) * jnp.ones(())


def _model_with_assertion():
    model = af.Model(af.ex.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
    model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=20.0)
    model.add_assertion(model.centre > model.normalization)
    return model


def test_nss_objective_applies_assertions_under_jit():
    """
    The scalar objective NSS samples through, jitted the way blackjax's step jits it:
    an assertion-violating vector maps to NSS's declared invalid value, without a tracer
    error, and a satisfying one keeps its log likelihood.
    """
    model = _model_with_assertion()
    fitness = af.NSS().make_fitness(
        model=model, analysis=_GaussianJaxAnalysis(), paths=None
    )

    objective = jax.jit(fitness.objective("scalar", compile=False))

    violating = jax.numpy.array([10.0, 25.0, 10.0])  # centre < normalization
    satisfying = jax.numpy.array([50.0, 25.0, 10.0])

    assert float(objective(violating)) == NSS_INVALID_LOG_LIKELIHOOD
    assert float(objective(satisfying)) == pytest.approx(0.0)


def test_nss_fit_with_assertions_runs_under_jit(monkeypatch):
    """
    The whole NSS loop (``jax.jit`` of blackjax's step) on a model with an assertion:
    before A3b this raised ``TracerBoolConversionError`` (survey 02 §3.3(b)). The
    maximum-likelihood sample satisfies the assertion.
    """
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    model = _model_with_assertion()

    search = af.NSS(n_live=50, num_delete=10, num_mcmc_steps=3)

    result = search.fit(model=model, analysis=_GaussianJaxAnalysis())

    instance = result.samples.max_log_likelihood()
    assert instance.centre > instance.normalization

    log_likelihoods = np.asarray(result.samples.log_likelihood_list)
    assert np.max(log_likelihoods) > NSS_INVALID_LOG_LIKELIHOOD


def test_nss_fitness_is_built_before_the_backend_runs(monkeypatch):
    """
    The resume likelihood sanity check (``Fitness.check_log_likelihood``, run when a
    ``Fitness`` with paths is built) happens before sampling, not after it (survey 02
    §3.3(c)).
    """
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    calls = []

    original = Fitness.check_log_likelihood

    def recording_check(self, fitness):
        calls.append("check")
        return original(self, fitness)

    class _Stop(Exception):
        pass

    def recording_run(self, ctx):
        calls.append("run")
        raise _Stop

    monkeypatch.setattr(Fitness, "check_log_likelihood", recording_check)
    monkeypatch.setattr(af.NSS, "run", recording_run)

    with pytest.raises(_Stop):
        af.NSS().fit(model=_model_with_assertion(), analysis=_GaussianJaxAnalysis())

    # The preflight's throwaway Fitness has no paths, so it never runs the check.
    assert calls == ["check", "run"]


def test_nss_switches_the_fitness_quick_update_cadence_off():
    """
    NSS fires its own quick update between outer iterations, so the ``Fitness`` it
    samples through carries no cadence (no second background worker).
    """
    search = af.NSS(iterations_per_quick_update=5)

    fitness = search.make_fitness(
        model=_model_with_assertion(),
        analysis=_GaussianJaxAnalysis(),
        paths=None,
        **search.fitness_overrides(_GaussianJaxAnalysis()),
    )

    assert fitness.iterations_per_quick_update is None
    assert search.iterations_per_quick_update == 5
