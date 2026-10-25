"""Short chains keep aligned draws and never request a zero thinning step."""
from types import SimpleNamespace

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations


@pytest.mark.parametrize("search_cls", [af.Emcee, af.Zeus])
@pytest.mark.parametrize("tau, thin", [(0.5, 1), (1.9, 1), (4.0, 2)])
def test_thinning_preserves_matching_draws(search_cls, tau, thin, monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    chain = np.arange(60.0).reshape(30, 2, 1)
    log_prob = -chain[..., 0] ** 2

    def select(values, discard=0, thin=1, flat=False):
        selected = values[discard + thin - 1::thin]
        return selected.reshape((-1,) + values.shape[2:]) if flat else selected

    sampler = SimpleNamespace(
        get_chain=lambda **kw: select(chain, **kw),
        get_log_prob=lambda **kw: select(log_prob, **kw),
        ncall_total=60,
    )
    search = search_cls()
    monkeypatch.setattr(search, "auto_correlations_from", lambda **_: AutoCorrelations(
        times=np.array([tau]), previous_times=None,
        check_size=100, required_length=50, change_threshold=0.01,
    ))
    raw = search.raw_samples_from(af.Collection(x=af.GaussianPrior(mean=0, sigma=1)), sampler)
    expected = chain[int(3 * tau) + thin - 1::thin].reshape(-1, 1)
    np.testing.assert_array_equal(raw.parameters, expected)
    np.testing.assert_array_equal(raw.log_posterior, -expected[:, 0] ** 2)


@pytest.mark.parametrize("length", [10, 100, 101])
def test_previous_autocorrelation_requires_a_nonempty_chain(length, monkeypatch):
    import emcee

    calls = []

    def integrated_time(x, tol):
        calls.append(x)
        assert len(x) > 0
        return np.array([2.0])

    monkeypatch.setattr(emcee.autocorr, "integrated_time", integrated_time)
    sampler = SimpleNamespace(
        get_chain=lambda: np.ones((length, 4, 1)),
        get_autocorr_time=lambda **_: np.array([3.0]),
    )
    result = af.Emcee().auto_correlations_from(sampler)
    if length <= 100:
        assert result.previous_times is None
        assert calls == []
        assert not result.check_if_converged(total_samples=length)
    else:
        np.testing.assert_array_equal(result.previous_times, [2.0])
        assert calls[0].shape == (1, 4, 1)
