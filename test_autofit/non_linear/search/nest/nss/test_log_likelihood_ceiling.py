"""
The log-likelihood guards on the closure ``af.NSS`` samples through.

``af.NSS`` does not route sampling through `Fitness.call` — it hands blackjax an inline JAX
closure — so the guards there are a second implementation and need their own test. The closure is
built by `nss_log_likelihood_from`, and these tests call that factory rather than restating its
body, so a change to the sampled path cannot pass a test that only resembles it.

The factory is JAX-only, so the file skips whole when JAX is absent (library policy: unit tests are
the numpy always-green layer).
"""

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search.nest.nss.search import (
    NSS_INVALID_LOG_LIKELIHOOD,
    nss_log_likelihood_from,
)
from test_autofit.non_linear.constant_analysis import ConstantAnalysis

jnp = pytest.importorskip("jax.numpy")


PARAMETERS = [1.0, 1.0, 1.0]

#: The ceiling `autolens_profiling` opts into. PyAutoFit ships the guard off, so every
#: enabled-path test below passes it explicitly.
CEILING = 1.0e20

#: Above the 1e20 ceiling, inside float32 range — so the value is rejected by the magnitude guard
#: and not by the isfinite guard that precedes it.
OVER_CEILING = 1.0e30

UNDER_CEILING = 1.0e19


def _log_likelihood_from(log_likelihood, **kwargs):
    return nss_log_likelihood_from(
        model=af.Model(af.ex.Gaussian),
        analysis=ConstantAnalysis(log_likelihood=log_likelihood),
        **kwargs,
    )


def test_nss_closure_rejects_a_log_likelihood_above_the_ceiling():
    """
    The failure this exists for: a finite `3e+303` out of an fp64 Cholesky becomes the
    highest-likelihood live point, the shell log evidence explodes and the run never terminates.
    """
    log_likelihood = _log_likelihood_from(OVER_CEILING, log_likelihood_ceiling=CEILING)

    assert float(log_likelihood(jnp.array(PARAMETERS))) == NSS_INVALID_LOG_LIKELIHOOD


def test_nss_closure_passes_a_log_likelihood_below_the_ceiling():
    log_likelihood = _log_likelihood_from(UNDER_CEILING, log_likelihood_ceiling=CEILING)

    assert float(log_likelihood(jnp.array(PARAMETERS))) == pytest.approx(
        UNDER_CEILING, rel=1.0e-5
    )


@pytest.mark.parametrize("log_likelihood", [np.nan, np.inf, -np.inf])
def test_nss_closure_still_rejects_non_finite_log_likelihoods(log_likelihood):
    """
    The magnitude guard is added *after* the isfinite guard; the pre-existing behaviour must be
    unchanged.
    """
    closure = _log_likelihood_from(log_likelihood, log_likelihood_ceiling=CEILING)

    assert float(closure(jnp.array(PARAMETERS))) == NSS_INVALID_LOG_LIKELIHOOD


def test_nss_closure_sentinel_maps_to_itself():
    """
    `NSS_INVALID_LOG_LIKELIHOOD` is itself above the ceiling in magnitude, so the guard has to be
    idempotent against the value it substitutes.
    """
    closure = _log_likelihood_from(
        NSS_INVALID_LOG_LIKELIHOOD, log_likelihood_ceiling=CEILING
    )

    assert float(closure(jnp.array(PARAMETERS))) == NSS_INVALID_LOG_LIKELIHOOD


def test_nss_closure_with_the_ceiling_disabled_passes_any_finite_value():
    closure = _log_likelihood_from(OVER_CEILING, log_likelihood_ceiling=float("inf"))

    assert float(closure(jnp.array(PARAMETERS))) == pytest.approx(
        OVER_CEILING, rel=1.0e-5
    )


def test_nss_closure_defaults_to_the_configured_ceiling_which_is_disabled():
    """
    With no explicit ceiling the closure reads the config — and the packaged config ships the guard
    **off**, because the threshold is a bare magnitude and a log likelihood scales with the
    noise-map units. So `af.NSS` picks up "disabled" by default and an enormous finite value passes
    through, exactly as it did before the guard existed. A config that opts in (as
    `autolens_profiling` does) is what turns the tests above into the live behaviour.
    """
    closure = _log_likelihood_from(OVER_CEILING)

    assert float(closure(jnp.array(PARAMETERS))) == pytest.approx(
        OVER_CEILING, rel=1.0e-5
    )


def test_nss_closure_still_rejects_non_finite_values_with_the_ceiling_disabled():
    """
    Turning the magnitude guard off must not turn the isfinite guard off with it — blackjax's
    nested-sampler arithmetic still cannot see a `NaN`.
    """
    closure = _log_likelihood_from(np.nan)

    assert float(closure(jnp.array(PARAMETERS))) == NSS_INVALID_LOG_LIKELIHOOD
