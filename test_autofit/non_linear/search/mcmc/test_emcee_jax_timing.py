"""
Eager-versus-jit timing guard for Emcee on a JAX analysis (search-extensibility A2).

Emcee declares `jax_use='none'` and evaluates one walker at a time through
`Fitness.call_wrap`. Before A2 that ran the JAX likelihood eagerly, op by op (7.80 ms
eager vs 0.29 ms jitted per call on `af.ex.Gaussian`, survey 02 §3.1). Now `call_wrap`
dispatches to the lazily jitted scalar objective, so the per-call cost Emcee pays must
stay within 2x the jitted call itself.
"""
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


def test_emcee_call_path_runs_at_jitted_speed(monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    x = np.arange(100.0)
    data = 3.0 * np.exp(-0.5 * ((x - 50.0) / 8.0) ** 2)
    analysis = af.ex.Analysis(data=data, noise_map=np.full(100, 0.1), use_jax=True)
    model = af.Model(af.ex.Gaussian)

    fitness = af.Emcee().make_fitness(analysis=analysis, model=model)
    vector = np.array(model.physical_values_from_prior_medians)

    jitted = fitness.objective("scalar")

    # Emcee's log_prob_fn is `fitness.call_wrap`, which dispatches to the jitted objective.
    assert fitness._call is jitted
    assert hasattr(jitted, "__wrapped__")

    jit_per_call = _per_call(jitted, vector, sync=True)
    emcee_per_call = _per_call(fitness.call_wrap, vector)

    assert emcee_per_call <= 2.0 * jit_per_call, (
        f"call_wrap {emcee_per_call * 1e3:.3f} ms vs jit {jit_per_call * 1e3:.3f} ms"
    )
