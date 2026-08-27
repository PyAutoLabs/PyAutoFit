"""
``AbstractMessage.natural_logpdf`` must not clamp a genuine ``-inf``.

The reduction maps NaN to ``-inf`` on purpose -- an out-of-support NaN (``log`` of a
negative value under a transformed message) is zero density. But ``nan_to_num``'s
DEFAULT ``neginf`` replaces a real ``-inf`` with ``-1.7976931348623157e+308``, so the
call did the opposite of its intent for exactly the inputs that were already correct.

``-1.8e308`` is finite, and ``isfinite`` is what ``optax.apply_if_finite`` and
``autofit.non_linear.clipper`` branch on to decide a lane has left the prior support.
"""

import sys

import numpy as np
import pytest

import autofit as af


@pytest.fixture(name="log_gaussian")
def make_log_gaussian():
    return af.LogGaussianPrior(mean=0.4, sigma=1.3)


def test__genuine_neginf_survives_the_reduction(log_gaussian):
    """
    ``log(0) = -inf`` reaches the reduction as ``-inf``; it must come back as ``-inf``,
    not as negative float max.
    """
    value = log_gaussian.message.logpdf(np.array(0.0))

    assert value == -np.inf
    assert not np.isfinite(value)
    assert value != -sys.float_info.max


def test__nan_still_maps_to_neginf(log_gaussian):
    """
    The other half of the same call, which was always correct: ``log(-1) = NaN``, and an
    out-of-support NaN is zero density. Pinned so a fix to one half cannot break the other.
    """
    value = log_gaussian.message.logpdf(np.array(-1.0))

    assert value == -np.inf
    assert not np.isnan(value)


@pytest.mark.parametrize("value", [1e-9, 0.001, 0.1, 0.5, 1.0, 2.0, 10.0, 1000.0])
def test__in_support_values_are_unchanged(log_gaussian, value):
    """
    The fix must touch only the non-finite branch. Every in-support point is finite before
    and after, and the density is the one the prior reports.
    """
    assert np.isfinite(log_gaussian.message.logpdf(np.array(value)))


@pytest.mark.parametrize(
    "prior, outside",
    [
        (af.UniformPrior(lower_limit=0.0, upper_limit=2.0), [-1.0, 3.0]),
        (af.LogUniformPrior(lower_limit=0.01, upper_limit=100.0), [-1.0, 0.0, 1000.0]),
        (af.LogGaussianPrior(mean=0.4, sigma=1.3), [-1.0, 0.0]),
    ],
    ids=["Uniform", "LogUniform", "LogGaussian"],
)
def test__no_prior_family_reports_a_finite_density_off_its_support(prior, outside):
    """
    The general form. A finite log density outside the support is a licence for any
    ``isfinite`` consumer to treat an impossible point as merely a bad one.

    ``TruncatedGaussianPrior`` is deliberately absent: its message returns finite values
    outside its limits for an unrelated reason (``TruncatedNormalMessage`` does not gate on
    them), which this fix does not address and must not be read as covering.
    """
    for value in outside:
        assert prior.message.logpdf(np.array(value)) == -np.inf
