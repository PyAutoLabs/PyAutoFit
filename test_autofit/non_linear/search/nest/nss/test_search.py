"""
Unit tests for ``af.NSS`` — initialisation, config, optional-import guard,
and samples round-trip via a synthetic ``_NSSInternal`` holder.

No JAX in unit tests (library policy — cross-xp checks live in
``autofit_workspace_test``). The numerical-parity smoke against a real
``run_nested_sampling`` call lives in
``autolens_workspace_developer/searches_minimal/nss_first_class.py``.
"""

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search.nest.nss import search as nss_search_module
from autofit.non_linear.search.nest.nss.samples import NSSamples


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__explicit_params():
    search = af.NSS(
        n_live=500,
        num_mcmc_steps=10,
        num_delete=20,
        chunk_size=4,
        termination=-2.0,
        seed=7,
    )

    assert search.n_live == 500
    assert search.num_mcmc_steps == 10
    assert search.num_delete == 20
    assert search.chunk_size == 4
    assert search.termination == -2.0
    assert search.seed == 7

    default = af.NSS()
    assert default.n_live == 200
    assert default.num_mcmc_steps == 5
    assert default.num_delete == 50
    assert default.chunk_size is None
    assert default.termination == -3.0
    assert default.seed == 42


def test__chunked_update_strategy_factory():
    """``make_chunked_update_strategy`` returns a callable with the same
    signature as blackjax's ``update_with_mcmc_take_last`` regardless of
    whether ``chunk_size`` is set. This lets ``af.NSS._fit`` drop it into
    ``blackjax.nss(update_strategy=...)`` without further branching.
    """
    from autofit.non_linear.search.nest.nss._chunked_update import (
        make_chunked_update_strategy,
    )

    strategy_none = make_chunked_update_strategy(None)
    strategy_chunked = make_chunked_update_strategy(4)
    # Both are callables with the upstream three-arg signature
    # (constrained_mcmc_step_fn, num_mcmc_steps, num_delete).
    import inspect

    for strategy in (strategy_none, strategy_chunked):
        params = list(inspect.signature(strategy).parameters)
        assert params == [
            "constrained_mcmc_step_fn",
            "num_mcmc_steps",
            "num_delete",
        ]


def test__identifier_fields():
    search = af.NSS()
    for field in ("n_live", "num_mcmc_steps", "num_delete", "termination", "seed"):
        assert field in search.__identifier_fields__


def test__test_mode_loosens_termination():
    search = af.NSS(termination=-3.0)
    search.apply_test_mode()
    assert search.termination == -1.0


def test__init_raises_when_nss_unavailable(monkeypatch):
    monkeypatch.setattr(nss_search_module, "_HAS_NSS", False)
    with pytest.raises(ImportError, match="af.NSS requires the optional `nss` package"):
        af.NSS()


def test__init_warns_when_number_of_cores_gt_one(caplog):
    with caplog.at_level("WARNING"):
        af.NSS(number_of_cores=4)
    assert any(
        "number_of_cores=4" in record.message and "ignored" in record.message
        for record in caplog.records
    )


def _make_synthetic_internal(n_live=20, ndim=2, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.normal(loc=0.0, scale=1.0, size=(n_live, ndim))
    loglikelihoods = rng.normal(loc=0.0, scale=0.5, size=(n_live,))
    log_weights = rng.normal(loc=-5.0, scale=0.1, size=(n_live,))
    logZs = rng.normal(loc=-10.0, scale=0.1, size=(100,))

    return nss_search_module._NSSInternal(
        positions=positions,
        loglikelihoods=loglikelihoods,
        log_weights=log_weights,
        logZs=logZs,
        wall_time=12.5,
        sampling_time=10.0,
        evals=5000,
        ess=150,
        n_live=n_live,
        num_mcmc_steps=5,
        num_delete=10,
        termination=-3.0,
        seed=42,
    )


def test__samples_info_from_synthetic_internal():
    search = af.NSS()
    internal = _make_synthetic_internal()

    info = search.samples_info_from(search_internal=internal)

    assert info["log_evidence"] == pytest.approx(float(internal.logZs.mean()), abs=1e-12)
    assert info["log_evidence_error"] == pytest.approx(
        float(internal.logZs.std()), abs=1e-12
    )
    assert info["total_samples"] == 5000
    assert info["total_accepted_samples"] == 20
    assert info["number_live_points"] == 20
    assert info["num_mcmc_steps"] == 5
    assert info["num_delete"] == 10
    assert info["termination"] == -3.0
    assert info["ess"] == 150
    assert info["time"] == pytest.approx(12.5, abs=1e-12)
    assert info["sampling_time"] == pytest.approx(10.0, abs=1e-12)


def test__samples_via_internal_from_returns_nssamples():
    model = af.Model(af.m.MockClassx2)
    model.one = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)
    model.two = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)

    search = af.NSS()
    internal = _make_synthetic_internal(n_live=20, ndim=model.prior_count)

    samples = search.samples_via_internal_from(model=model, search_internal=internal)

    assert isinstance(samples, NSSamples)
    assert len(samples.sample_list) == 20
    assert samples.model is model

    weights = np.array([s.weight for s in samples.sample_list])
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert (weights >= 0).all()

    assert samples.samples_info["log_evidence"] == pytest.approx(
        float(internal.logZs.mean()), abs=1e-12
    )


def test__samples_weight_normalisation_handles_zero_total():
    """When every log_weight is -inf (e.g. catastrophic underflow) we should
    not divide by zero. The implementation clamps via max-subtraction so this
    case should still produce zero-weights without crashing."""
    model = af.Model(af.m.MockClassx2)
    model.one = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)
    model.two = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)

    search = af.NSS()
    internal = _make_synthetic_internal(n_live=10, ndim=model.prior_count)
    # Sentinel: all log-weights identical → exp(log_w - max) = 1 for all,
    # weights normalise uniformly to 1/n.
    internal.log_weights = np.full_like(internal.log_weights, -3.14)

    samples = search.samples_via_internal_from(model=model, search_internal=internal)

    weights = np.array([s.weight for s in samples.sample_list])
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(weights, 1.0 / 10, atol=1e-12)


def test__nssamples_log_evidence_error_property():
    model = af.Model(af.m.MockClassx2)
    model.one = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)
    model.two = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)

    search = af.NSS()
    internal = _make_synthetic_internal(n_live=10, ndim=model.prior_count)

    samples = search.samples_via_internal_from(model=model, search_internal=internal)

    assert samples.log_evidence_error == pytest.approx(
        float(internal.logZs.std()), abs=1e-12
    )
