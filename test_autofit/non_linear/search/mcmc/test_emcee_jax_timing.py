"""Emcee reuses the compiled scalar objective without per-walker retracing.

CI checks dispatch and numerical equivalence. The sub-millisecond timing ratio
is a local profiling diagnostic, opt-in with PYAUTO_PROFILE_EMCEE_JAX=1.
"""
import os
from unittest.mock import Mock

import time

import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _per_call(func, vector, n=200, repeats=5, sync=False):
    func(vector)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(n):
            result = func(vector)
            if sync:
                float(result)
        best = min(best, (time.perf_counter() - start) / n)
    return best


def _fitness_and_vector(monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    x = np.arange(100.0)
    data = 3.0 * np.exp(-0.5 * ((x - 50.0) / 8.0) ** 2)
    analysis = af.ex.Analysis(data=data, noise_map=np.full(100, 0.1), use_jax=True)
    model = af.Model(af.ex.Gaussian)

    fitness = af.Emcee().make_fitness(analysis=analysis, model=model)
    vector = np.array(model.physical_values_from_prior_medians)

    return fitness, vector


def test_emcee_call_path_reuses_compiled_objective(monkeypatch):
    traces = []
    likelihood = af.ex.Analysis.log_likelihood_function

    def counting_likelihood(self, *args, **kwargs):
        traces.append(1)
        return likelihood(self, *args, **kwargs)

    monkeypatch.setattr(af.ex.Analysis, "log_likelihood_function", counting_likelihood)
    fitness, vector = _fitness_and_vector(monkeypatch)
    jitted = fitness.objective("scalar")

    # Emcee's log_prob_fn is `fitness.call_wrap`, which dispatches to the jitted objective.
    assert fitness._call is jitted
    assert hasattr(jitted, "__wrapped__")

    # Warm one shape, then use distinct values and containers without re-tracing.
    vectors = [vector, vector + 0.01, (vector + 0.02).tolist()]
    expected = [float(jitted(np.asarray(v, dtype=float))) for v in vectors]
    assert len(traces) == 1
    compiled_call = Mock(wraps=jitted)
    monkeypatch.setattr(fitness, "_call", compiled_call)
    for i, (v, value) in enumerate(zip(vectors, expected), start=1):
        assert fitness.call_wrap(v) == pytest.approx(value)
        assert compiled_call.call_count == i
        np.testing.assert_array_equal(compiled_call.call_args.args[0], v)
        assert fitness.objective("scalar") is jitted
        assert len(traces) == 1


@pytest.mark.skipif(
    os.environ.get("PYAUTO_PROFILE_EMCEE_JAX") != "1",
    reason="wall-clock comparison is opt-in local profiling, not a CI gate",
)
def test_emcee_call_path_runs_at_jitted_speed(monkeypatch):
    fitness, vector = _fitness_and_vector(monkeypatch)
    jitted = fitness.objective("scalar")
    jit_per_call = _per_call(jitted, vector, sync=True)
    emcee_per_call = _per_call(fitness.call_wrap, vector)

    assert emcee_per_call <= 2.0 * jit_per_call, (
        f"call_wrap {emcee_per_call * 1e3:.3f} ms vs jit {jit_per_call * 1e3:.3f} ms"
    )
