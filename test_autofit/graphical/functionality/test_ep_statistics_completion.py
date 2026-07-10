"""Regression tests for the EP statistics completion (PyAutoFit #1353;
findings F6 and F7(b) from the #1332 audit). Numpy-only.
"""
import numpy as np
import pytest
from scipy.stats import truncnorm

import autofit as af
from autofit.graphical.mean_field import MeanField
from autofit.messages.truncated_normal import TruncatedNormalMessage


# --- F6: exact same-support truncated KL via truncated moments ---

def _mc_kl_truncnorm(p, q, n=400_000, seed=3):
    a_p = (p.lower_limit - p.mean) / p.sigma
    b_p = (p.upper_limit - p.mean) / p.sigma
    a_q = (q.lower_limit - q.mean) / q.sigma
    b_q = (q.upper_limit - q.mean) / q.sigma
    rng = np.random.default_rng(seed)
    x = truncnorm.rvs(
        a_p, b_p, loc=p.mean, scale=p.sigma, size=n, random_state=rng
    )
    return float(
        np.mean(
            truncnorm.logpdf(x, a_p, b_p, loc=p.mean, scale=p.sigma)
            - truncnorm.logpdf(x, a_q, b_q, loc=q.mean, scale=q.sigma)
        )
    )


def _old_untruncated_kl(p, q):
    return (
        np.log(q.sigma / p.sigma)
        + (p.sigma**2 + (p.mean - q.mean) ** 2) / 2 / q.sigma**2
        - 1 / 2
    )


@pytest.mark.parametrize(
    "p, q",
    [
        # comfortable interior — old approximation was nearly right here
        (
            TruncatedNormalMessage(0.0, 1.0, -10.0, 10.0),
            TruncatedNormalMessage(0.5, 2.0, -10.0, 10.0),
        ),
        # mass pressed against the bounds — old approximation degrades badly
        (
            TruncatedNormalMessage(0.9, 1.5, -1.0, 1.0),
            TruncatedNormalMessage(-0.5, 0.7, -1.0, 1.0),
        ),
        # half-bounded, mean below the bound (prior-passing shape)
        (
            TruncatedNormalMessage(-0.5, 1.0, 0.0, np.inf),
            TruncatedNormalMessage(1.0, 2.0, 0.0, np.inf),
        ),
    ],
    ids=["interior", "near-bounds", "half-bounded"],
)
def test__truncated_kl_matches_monte_carlo(p, q):
    analytic = float(p.kl(q))
    mc = _mc_kl_truncnorm(p, q)
    assert analytic == pytest.approx(mc, abs=0.02)


def test__truncated_kl_reduces_to_gaussian_for_wide_bounds():
    p = TruncatedNormalMessage(0.0, 1.0, -50.0, 50.0)
    q = TruncatedNormalMessage(1.0, 2.0, -50.0, 50.0)
    assert float(p.kl(q)) == pytest.approx(float(_old_untruncated_kl(p, q)), rel=1e-9)


def test__truncated_kl_near_bounds_differs_from_old_formula():
    # The case the audit quantified (errors 1.5% -> 140% as mass reaches the
    # bounds): the exact value must both match MC and differ measurably from
    # the old untruncated formula.
    p = TruncatedNormalMessage(0.9, 1.5, -1.0, 1.0)
    q = TruncatedNormalMessage(-0.5, 0.7, -1.0, 1.0)
    exact = float(p.kl(q))
    old = float(_old_untruncated_kl(p, q))
    assert abs(exact - old) / abs(exact) > 0.05


def test__truncated_kl_different_support_still_raises():
    p = TruncatedNormalMessage(0.0, 1.0, -1.0, 1.0)
    q = TruncatedNormalMessage(0.0, 1.0, -2.0, 2.0)
    with pytest.raises(ValueError):
        p.kl(q)


# --- F7(b): from_priors records the per-factor evidence on log_norm ---

def test__from_priors_log_norm_default_and_explicit():
    priors = [af.GaussianPrior(0.0, 1.0), af.GaussianPrior(1.0, 2.0)]
    assert MeanField.from_priors(priors).log_norm == 0.0
    mf = MeanField.from_priors(priors, log_norm=-42.5)
    assert mf.log_norm == -42.5
    # and it survives the (now-fixed, #1351) operator algebra
    assert (mf / MeanField.from_priors(priors, log_norm=-2.5)).log_norm == -40.0
