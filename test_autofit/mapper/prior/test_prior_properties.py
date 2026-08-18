"""
Property-based correctness sweep over every concrete ``Prior`` subclass (#1497).

Phase 3 of the priors/messages census (hub #1331): the nine confirmed bugs from
the audit are fixed and locked pointwise by ``test_priors_messages_fixes_1331.py``
and ``test_prior_width_safety.py``; the properties here are the *general* form of
those checks, parametrised over every prior family, so the next latent math bug
of the same shape fails CI instead of surviving for years:

- P1 ``value_for`` is the inverse CDF (would catch inverse-CDF errors).
- P2 the normalised log prior integrates to 1, locking the #1331 Option A
  contract: ``log_prior_from_value`` drops constants, ``log_normalisation()``
  recovers them (would catch the TruncatedNormal log-partition bug, #1331-04).
- P3 the ``log_prior_from_value`` gradient matches the message-density gradient
  (constants cancel — would catch the LogUniform sign-convention bug, #1266).
- P4 ``with_limits`` constructs for every family with its documented semantics
  (would catch the LogGaussian ``with_limits`` crash, #1331-01).
- P5 ``from_mode(mode, variance)`` reproduces mean and variance at variance != 1
  discriminating points (would catch the Gamma ``from_mode`` inversion, #1331-D3).

House rule: library-level property tests stay NumPy-only; EP-level integration
coverage lives with the EP framework review and complements this file.
"""
import numpy as np
import pytest
from scipy.integrate import quad

import autofit as af
from autofit.messages.beta import BetaMessage
from autofit.messages.composed_transform import TransformedMessage
from autofit.messages.gamma import GammaMessage
from autofit.messages.normal import NormalMessage
from autofit.messages.truncated_normal import TruncatedNormalMessage


def all_priors():
    """
    Every concrete Prior family, two parameterisations each where the family
    has enough freedom to make that meaningful.

    Skipped deliberately: TuplePrior (container, not a density), Constant
    (point mass), DeferredArgument (not a density).
    """
    return [
        af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        af.UniformPrior(lower_limit=-3.0, upper_limit=7.5),
        af.GaussianPrior(mean=0.0, sigma=1.0),
        af.GaussianPrior(mean=2.5, sigma=0.3),
        af.LogUniformPrior(lower_limit=0.01, upper_limit=100.0),
        af.LogGaussianPrior(mean=0.0, sigma=1.0),
        af.LogGaussianPrior(mean=1.0, sigma=0.5),
        af.TruncatedGaussianPrior(
            mean=0.0, sigma=1.0, lower_limit=-2.0, upper_limit=2.0
        ),
        af.TruncatedGaussianPrior(
            mean=1.0, sigma=0.5, lower_limit=0.0, upper_limit=3.0
        ),
    ]


def prior_id(prior):
    return f"{type(prior).__name__}({prior.message})"


def support_window(prior, tail=1e-11):
    """
    A finite integration window covering all but ``tail`` of the prior mass,
    for families whose support is infinite.
    """
    lo, hi = prior.limits
    if not np.isfinite(lo):
        lo = prior.value_for(tail)
    if not np.isfinite(hi):
        hi = prior.value_for(1.0 - tail)
    return float(lo), float(hi)


def physical_log_density(prior, x):
    """
    log p(x) in physical coordinates via the wrapped message.

    For a ``TransformedMessage`` the generic ``logpdf`` path omits the
    transform Jacobian (#1498), so the physical density lives on ``factor``;
    a direct message's ``logpdf`` is already the physical density. Once #1498
    is resolved this helper can collapse to ``message.logpdf``.
    """
    message = prior.message
    if isinstance(message, TransformedMessage):
        return float(message.factor(np.asarray(x)))
    return float(message.logpdf(np.asarray(x)))


# === P1: value_for is the inverse CDF ===


@pytest.mark.parametrize("prior", all_priors(), ids=prior_id)
@pytest.mark.parametrize("unit", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test__value_for_is_inverse_cdf(prior, unit):
    x = prior.value_for(unit)
    assert float(prior.cdf(x)) == pytest.approx(unit, abs=1e-6)


# === P2: the normalised prior density integrates to 1 ===


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("prior", all_priors(), ids=prior_id)
def test__normalised_log_prior_integrates_to_one(prior):
    """
    Locks the #1331 Option A contract: ``log_prior_from_value`` returns the
    log density up to an additive constant and ``log_normalisation()`` is
    exactly that constant, so together they are the normalised density.
    """
    lo, hi = support_window(prior)
    log_norm = float(prior.log_normalisation())

    def pdf(x):
        return float(np.exp(prior.log_prior_from_value(x) + log_norm))

    integral, err = quad(pdf, lo, hi, limit=200)
    assert integral == pytest.approx(1.0, abs=max(err, 1e-3))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("prior", all_priors(), ids=prior_id)
def test__message_physical_density_integrates_to_one(prior):
    """
    The message's physical-density path must agree that the prior is a
    normalised distribution — the general form of the #1331-04 regression
    (TruncatedNormalMessage pdf integrated to 2.27 through the generic path).
    """
    lo, hi = support_window(prior)

    def pdf(x):
        return float(np.exp(physical_log_density(prior, float(x))))

    integral, err = quad(pdf, lo, hi, limit=200)
    assert integral == pytest.approx(1.0, abs=max(err, 1e-3))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "message",
    [
        NormalMessage(0.5, 1.3),
        TruncatedNormalMessage(0.3, 1.1, -1.0, 2.5),
        GammaMessage(2.0, 1.5),
        BetaMessage(2.5, 3.5),
    ],
    ids=lambda m: type(m).__name__,
)
def test__generic_interface_pdf_integrates_to_one(message):
    """
    The generic exponential-family ``logpdf`` path (natural parameters +
    log_partition) must produce a normalised pdf for every direct message.
    Extends the ``check_dist_norm`` sweep to ``TruncatedNormalMessage`` —
    the exact exclusion that let #1331-04 survive.
    """
    # Some direct messages (e.g. GammaMessage) implement no inverse CDF, so an
    # infinite support is truncated at mean +/- 40 sigma — far beyond any mass
    # the 1e-3 integral tolerance could feel.
    lo, hi = message._support[0]
    mean, sigma = float(message.mean), float(np.sqrt(message.variance))
    if not np.isfinite(lo):
        lo = mean - 40.0 * sigma
    if not np.isfinite(hi):
        hi = mean + 40.0 * sigma

    integral, err = quad(
        lambda x: float(message.pdf(np.asarray(float(x)))), lo, hi, limit=200
    )
    assert integral == pytest.approx(1.0, abs=max(err, 1e-3))


# === P3: log_prior gradient matches the density gradient ===


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("prior", all_priors(), ids=prior_id)
@pytest.mark.parametrize("unit", [0.3, 0.5, 0.7])
def test__log_prior_gradient_matches_density_gradient(prior, unit):
    """
    ``log_prior_from_value`` and the physical density differ by a constant,
    so their gradients must agree everywhere. A sign or Jacobian slip in
    either path breaks this — exactly the #1266 LogUniform failure mode.
    """
    x = prior.value_for(unit)
    eps = 1e-6 * max(1.0, abs(x))

    grad_log_prior = (
        prior.log_prior_from_value(x + eps) - prior.log_prior_from_value(x - eps)
    ) / (2 * eps)
    grad_density = (
        physical_log_density(prior, x + eps) - physical_log_density(prior, x - eps)
    ) / (2 * eps)

    assert np.isclose(grad_log_prior, grad_density, rtol=1e-2, atol=1e-3)


# === P4: with_limits constructs with its documented semantics ===


@pytest.mark.parametrize("prior", all_priors(), ids=prior_id)
def test__with_limits_constructs_for_every_family(prior):
    """
    The general form of the #1331-01 regression: ``with_limits`` is the
    standard prior-passing entry point and must never crash.
    """
    new = prior.with_limits(0.5, 2.0)
    assert new is not None
    assert type(new) is type(prior)


@pytest.mark.parametrize(
    "prior, limits",
    [
        (af.UniformPrior(0.0, 1.0), (0.2, 0.8)),
        (af.UniformPrior(-3.0, 7.5), (-1.0, 5.0)),
        (af.LogUniformPrior(0.01, 100.0), (0.1, 10.0)),
        (
            af.TruncatedGaussianPrior(
                mean=0.0, sigma=1.0, lower_limit=-2.0, upper_limit=2.0
            ),
            (-1.0, 1.0),
        ),
        (
            af.TruncatedGaussianPrior(
                mean=1.0, sigma=0.5, lower_limit=0.0, upper_limit=3.0
            ),
            (0.5, 2.0),
        ),
    ],
)
def test__with_limits_hard_limit_families_hit_the_limits(prior, limits):
    lower, upper = limits
    new = prior.with_limits(lower, upper)
    assert new.value_for(1e-6) == pytest.approx(lower, abs=1e-3)
    assert new.value_for(1.0 - 1e-6) == pytest.approx(upper, abs=1e-3)


@pytest.mark.parametrize(
    "cls", [af.GaussianPrior, af.LogGaussianPrior], ids=lambda c: c.__name__
)
def test__with_limits_gaussian_families_centre_between_limits(cls):
    """
    Gaussian-family ``with_limits`` documents soft limits: a new prior centred
    between the limits with sigma equal to their separation. For LogGaussian
    the shipped #1331-01 fix drops the (unsupported) hard-limit kwargs, so the
    support stays (0, inf).
    """
    new = cls.with_limits(0.5, 2.5)
    assert new.mean == pytest.approx(1.5)
    assert new.sigma == pytest.approx(2.0)
    if cls is af.LogGaussianPrior:
        assert new.value_for(1e-6) > 0.0


# === P5: from_mode reproduces mean and variance ===


@pytest.mark.parametrize("cls", [NormalMessage, GammaMessage], ids=lambda c: c.__name__)
@pytest.mark.parametrize("mean, variance", [(2.0, 0.25), (2.0, 4.0), (0.7, 0.1)])
def test__from_mode_matches_mean_and_variance(cls, mean, variance):
    """
    Locks the #1331-D3 invariant: ``from_mode(m, V)`` matches mean and
    variance, consistently across the Normal and Gamma families. The
    variance != 1 points discriminate the corrected Gamma formula
    (alpha = m^2/V, beta = m/V) from the historical inverted one
    (alpha = 1 + m^2*V), which these points expose immediately.
    """
    message = cls.from_mode(np.asarray(mean), np.asarray(variance))
    assert float(message.mean) == pytest.approx(mean, rel=1e-6)
    assert float(message.variance) == pytest.approx(variance, rel=1e-6)
