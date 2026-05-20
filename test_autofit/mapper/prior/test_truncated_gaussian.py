import math

import numpy as np
import pytest
from scipy.stats import norm, truncnorm

import autofit as af
from autofit.messages.truncated_normal import TruncatedNormalMessage


@pytest.fixture(name="truncated_gaussian")
def make_truncated_gaussian():
    return af.TruncatedGaussianPrior(mean=1.0, sigma=2.0, lower_limit=0.95, upper_limit=1.05)


@pytest.mark.parametrize(
    "unit, value",
    [
        (0.001, 0.95),
        (0.5, 1.0),
        (0.999, 1.05),
    ],
)
def test__values(truncated_gaussian, unit, value):

    assert truncated_gaussian.value_for(unit) == pytest.approx(value, rel=0.1)

@pytest.mark.parametrize(
    "unit, value",
    [
        (0.01, -np.inf),
        (1.0, 2.3026892553),
        (2.0, -np.inf),
    ],
)
def test__log_prior_from_value(truncated_gaussian, unit, value):

    assert truncated_gaussian.log_prior_from_value(unit) == pytest.approx(value, rel=0.1)


# --- Numerical equivalence: new direct-ndtr path vs the OLD scipy.stats.norm
# CDF/PPF composition that this PR replaces. They must be bit-exact equal —
# that's the "numerics don't change" gate.

PARAMS = [
    # (mean, sigma, lower_limit, upper_limit)
    (0.0, 1.0, -3.0, 3.0),       # symmetric, moderate
    (0.0, 1.0, -10.0, 10.0),     # very wide
    (5.0, 2.0, 0.0, math.inf),   # half-bounded (matches toy normalization)
    (5.0, 5.0, 0.0, math.inf),   # half-bounded (matches toy sigma)
    (1.0, 2.0, 0.95, 1.05),      # narrow bracket
    (0.0, 1.0, -0.001, 0.001),   # very narrow
]

UNITS = [1e-6, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-3, 1 - 1e-6]


def _old_value_for(unit, mean, sigma, lower, upper):
    """Reproduces the pre-refactor scipy.stats.norm.cdf/ppf composition.
    This is the algorithm whose results must be preserved."""
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    lower_cdf = norm.cdf(a)
    upper_cdf = norm.cdf(b)
    truncated_cdf = lower_cdf + unit * (upper_cdf - lower_cdf)
    x_standard = norm.ppf(truncated_cdf)
    return mean + sigma * x_standard


@pytest.mark.parametrize("mean,sigma,lower,upper", PARAMS)
@pytest.mark.parametrize("unit", UNITS)
def test__prior_value_for_bit_exact_to_old_path(unit, mean, sigma, lower, upper):
    """`TruncatedGaussianPrior.value_for` must produce results bit-exact to
    the pre-refactor scipy.stats.norm.cdf/ppf composition that this PR
    replaces. This is the "numerics don't change" gate at the algorithmic
    level — both paths share the same ndtr/ndtri Cephes routines, only the
    Python-side wrapper differs."""
    prior = af.TruncatedGaussianPrior(
        mean=mean, sigma=sigma, lower_limit=lower, upper_limit=upper,
    )
    expected = float(_old_value_for(unit, mean, sigma, lower, upper))
    actual = float(prior.value_for(unit))

    if expected == 0.0:
        assert actual == 0.0
    else:
        # Same Cephes routines under the hood — must be bit-exact.
        assert actual == expected, f"new={actual!r} old={expected!r}"


@pytest.mark.parametrize("mean,sigma,lower,upper", PARAMS)
@pytest.mark.parametrize("unit", [0.1, 0.3, 0.5, 0.7, 0.9])
def test__prior_value_for_close_to_scipy_truncnorm(unit, mean, sigma, lower, upper):
    """`TruncatedGaussianPrior.value_for` matches scipy.stats.truncnorm.ppf
    away from the deep tails. scipy.stats.truncnorm uses its own tail-safe
    branching that the simple ndtr/ndtri composition does not — so this
    test deliberately covers only ``unit in [0.1, 0.9]`` where both paths
    are stable. Documents the precision regime; not a regression gate."""
    prior = af.TruncatedGaussianPrior(
        mean=mean, sigma=sigma, lower_limit=lower, upper_limit=upper,
    )
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    expected = float(truncnorm.ppf(unit, a=a, b=b, loc=mean, scale=sigma))
    actual = float(prior.value_for(unit))

    if expected == 0.0:
        assert actual == pytest.approx(0.0, abs=1e-12)
    else:
        assert actual == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("mean,sigma,lower,upper", PARAMS)
@pytest.mark.parametrize("unit", UNITS)
def test__message_value_for_matches_prior(unit, mean, sigma, lower, upper):
    """`TruncatedNormalMessage.value_for` must produce the same output as
    `TruncatedGaussianPrior.value_for` for matching parameters — both now
    route through the shared helper, so the equality is bit-exact."""
    prior = af.TruncatedGaussianPrior(
        mean=mean, sigma=sigma, lower_limit=lower, upper_limit=upper,
    )
    message = TruncatedNormalMessage(
        mean=mean, sigma=sigma, lower_limit=lower, upper_limit=upper,
    )
    assert float(message.value_for(unit)) == float(prior.value_for(unit))


def test__jax_value_for_parity():
    """JAX path must match the numpy path to within float64 rounding noise.

    Uses moderate (half-bounded) parameters representative of the toy model.
    Skipped if jax is not installed; CI / dev installs both.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    prior = af.TruncatedGaussianPrior(
        mean=5.0, sigma=2.0, lower_limit=0.0, upper_limit=math.inf,
    )
    for unit in [0.1, 0.5, 0.9]:
        numpy_val = float(prior.value_for(unit, xp=np))
        jax_val = float(prior.value_for(jnp.asarray(unit), xp=jnp))
        assert jax_val == pytest.approx(numpy_val, rel=1e-9)
