"""
BFGS / LBFGS on a JAX analysis minimise with the exact gradient of the "value_and_grad"
objective (`jac=True`), not scipy's finite differences (search-extensibility phase A2).

With `jac=True` every scipy function evaluation is one call of the jitted value-and-gradient
objective, which returns the gradient too: `nfev` equals the number of those calls and no
scalar (finite-difference) call happens inside `minimize`. Finite differences cost
`n_params` extra function evaluations per gradient, so on the numpy path
`nfev >= (n_params + 1) * njev`.
"""
import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _data():
    x = np.arange(50)
    signal = 3.0 * np.exp(-0.5 * ((x - 25.0) / 5.0) ** 2)
    return signal, np.full(50, 0.1)


def _model():
    model = af.Model(af.ex.Gaussian)
    model.centre = af.UniformPrior(lower_limit=10.0, upper_limit=40.0)
    model.normalization = af.UniformPrior(lower_limit=0.5, upper_limit=6.0)
    model.sigma = af.UniformPrior(lower_limit=1.0, upper_limit=10.0)
    return model


@pytest.mark.parametrize("search_cls", [af.BFGS, af.LBFGS])
def test_jax_analysis_uses_the_exact_gradient(search_cls, monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    data, noise_map = _data()
    analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

    from autofit.non_linear.fitness import Fitness

    calls = {"value_and_grad": 0}
    original = Fitness.call_wrap_value_and_grad

    def counted(self, parameters):
        calls["value_and_grad"] += 1
        return original(self, parameters)

    monkeypatch.setattr(Fitness, "call_wrap_value_and_grad", counted)

    search = search_cls(maxiter=50)
    result = search.fit(model=_model(), analysis=analysis)

    internal = result.search_internal
    n_params = 3

    assert internal.njev > 0
    assert calls["value_and_grad"] == internal.nfev
    assert internal.nfev < (n_params + 1) * internal.njev


def test_numpy_analysis_still_differences_finitely(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    data, noise_map = _data()
    analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=False)

    search = af.BFGS(maxiter=20)
    result = search.fit(model=_model(), analysis=analysis)

    internal = result.search_internal

    assert internal.nfev >= 4 * internal.njev
