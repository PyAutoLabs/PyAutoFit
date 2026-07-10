"""Regression tests for the priors/messages correctness batch (PyAutoFit #1344,
decision hub #1331; per-bug detail #1330). Numpy-only — no JAX in library unit
tests.
"""
import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

import autofit as af
from autofit.messages.fixed import FixedMessage
from autofit.messages.gamma import GammaMessage
from autofit.messages.truncated_normal import TruncatedNormalMessage
from autofit.messages.beta import inv_beta_suffstats


# --- #01: LogGaussianPrior.with_limits / _new_for_base_message no longer crash ---

def test__log_gaussian_with_limits_no_crash():
    # Previously raised TypeError: the ctor does not accept lower/upper_limit.
    prior = af.LogGaussianPrior.with_limits(0.5, 1.5)
    assert prior.mean == pytest.approx(1.0)   # (0.5 + 1.5) / 2
    assert prior.sigma == pytest.approx(1.0)  # 1.5 - 0.5
    # _new_for_base_message (called during projection with the base NormalMessage)
    # was also broken — the invalid kwargs plus a stale self.instance().id call.
    remade = prior._new_for_base_message(prior.message.base_message)
    assert isinstance(remade, af.LogGaussianPrior)
    assert remade.id == prior.id


# --- #02: UniformPrior.logpdf handles array input; scalar path bit-identical ---

def test__uniform_logpdf_array_matches_scalar():
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    arr = np.asarray(prior.logpdf(xs))
    assert arr.shape == xs.shape
    for i, x in enumerate(xs):
        # scalar path must equal the corresponding array element exactly
        assert float(prior.logpdf(float(x))) == float(arr[i])


# --- #04: TruncatedNormalMessage.log_partition includes the Gaussian term, so the
# generic exponential-family pdf integrates to 1.0 (was sigma*exp(mu^2/2sigma^2)) ---

def test__truncated_message_log_partition_normalises_pdf():
    # mu=1, sigma=2 -> old error factor sigma*exp(mu^2/2sigma^2) = 2*exp(1/8) ~= 2.27.
    m = TruncatedNormalMessage(mean=1.0, sigma=2.0, lower_limit=-20.0, upper_limit=20.0)
    integral, _ = integrate.quad(lambda x: float(np.exp(m.logpdf(np.array(x)))), -20.0, 20.0)
    assert integral == pytest.approx(1.0, rel=1e-4)

    # log_partition = untruncated Gaussian cumulant + log(truncation mass).
    a = (m.lower_limit - m.mean) / m.sigma
    b = (m.upper_limit - m.mean) / m.sigma
    expected = (m.mean ** 2) / (2 * m.sigma ** 2) + np.log(m.sigma) + np.log(norm.cdf(b) - norm.cdf(a))
    assert float(m.log_partition()) == pytest.approx(expected, rel=1e-9)


# --- #10: FixedMessage.logpdf returns a fresh array (no aliased class-level cache) ---

def test__fixed_message_logpdf_not_aliased():
    assert not hasattr(FixedMessage, "logpdf_cache")
    m = FixedMessage(np.array([1.0, 2.0, 3.0]))
    x = np.zeros(3)
    first = m.logpdf(x)
    first[0] = 999.0            # mutating one result must not corrupt the next
    second = m.logpdf(x)
    assert np.all(second == 0.0)
    assert first is not second


# --- Decision 1: inv_beta_suffstats raises on negative projection (was a no-op clamp) ---

def test__inv_beta_suffstats_raises_on_negative(monkeypatch):
    # Force the Newton-Raphson step negative (per the #05 reproducer) and confirm
    # the projection now raises instead of silently warning + no-op clamping.
    from scipy.special import digamma
    a0, b0 = np.array([2.0, 4.0, 3.0]), np.array([3.0, 1.5, 5.0])
    lnX = digamma(a0) - digamma(a0 + b0)
    ln1X = digamma(b0) - digamma(a0 + b0)
    monkeypatch.setattr(np.linalg, "solve", lambda J, f: np.full(np.shape(f), -1e3))
    with pytest.raises(ValueError):
        inv_beta_suffstats(lnX, ln1X)


def test__inv_beta_suffstats_positive_projection_does_not_raise(monkeypatch):
    # The guard only fires on a negative projection: a positive Newton step must
    # pass through unchanged (this is the in-scope behaviour Decision 1 changes —
    # the underlying solve shape-handling is pre-existing and untouched here).
    from scipy.special import digamma
    a0, b0 = np.array([2.0, 4.0, 3.0]), np.array([3.0, 1.5, 5.0])
    lnX = digamma(a0) - digamma(a0 + b0)
    ln1X = digamma(b0) - digamma(a0 + b0)
    monkeypatch.setattr(
        np.linalg, "solve", lambda J, f: np.zeros(np.shape(f))
    )  # no Newton movement -> ab stays at its positive initial guess
    a, b = inv_beta_suffstats(lnX, ln1X)
    assert np.all(np.asarray(a) > 0) and np.all(np.asarray(b) > 0)


# --- Decision 3: GammaMessage.from_mode matches the requested mean and variance ---

@pytest.mark.parametrize("mean, variance", [(2.0, 0.25), (1.0, 4.0), (5.0, 0.5)])
def test__gamma_from_mode_matches_mean_variance(mean, variance):
    g = GammaMessage.from_mode(mean, variance)
    assert float(g.mean) == pytest.approx(mean, rel=1e-9)
    assert float(g.variance) == pytest.approx(variance, rel=1e-9)


# --- Decision 4: log_normalisation recovers the fully normalised density from the
# constant-dropping log_prior_from_value, for every prior with a known normaliser ---

def test__uniform_log_normalisation():
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    # normalised density of a uniform on [0, 2] is log(1/2)
    recovered = prior.log_prior_from_value(1.0) + prior.log_normalisation()
    assert float(recovered) == pytest.approx(np.log(0.5), rel=1e-9)


def test__log_uniform_log_normalisation():
    prior = af.LogUniformPrior(lower_limit=1.0, upper_limit=100.0)
    value = 10.0
    recovered = prior.log_prior_from_value(value) + prior.log_normalisation()
    expected = -np.log(value) - np.log(np.log(100.0 / 1.0))
    assert float(recovered) == pytest.approx(expected, rel=1e-9)


def test__gaussian_log_normalisation():
    prior = af.GaussianPrior(mean=1.0, sigma=2.0)
    value = 1.5
    recovered = prior.log_prior_from_value(value) + prior.log_normalisation()
    expected = norm.logpdf(value, loc=1.0, scale=2.0)
    assert float(recovered) == pytest.approx(expected, rel=1e-9)
