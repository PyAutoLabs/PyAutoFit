import importlib.util
import math

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search.mcmc.blackjax.smc.search import (
    _build_search_internal,
    _log_evidence_from,
    auto_step_size_from,
    ess_from_weights,
    log_prior_normalisation_from,
    prior_limits_from,
    prior_scales_from,
    whitening_from,
)

# The end-to-end tests below run blackjax (which pulls jax); both ship via the `[optional]` extras. The
# Python-version matrix installs autofit without those extras, so skip there rather than fail with
# `No module named 'blackjax'`.
requires_blackjax = pytest.mark.skipif(
    importlib.util.find_spec("blackjax") is None,
    reason="requires blackjax (installed via the [optional] extras; absent on the matrix env)",
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class _FakeSamples:
    """A minimal `Samples`-like object -- just enough for `inverse_mass_matrix_kind_from`
    (construction-time classification only; no resolution) to recognise it as a "result" kind."""

    def __init__(self, covariance_matrix):
        self.covariance_matrix = covariance_matrix


# --------------------------------------------------------------------------
# The analytic target used by the end-to-end evidence tests.
#
# A zero-centred isotropic Gaussian likelihood under uniform priors on [-LIMIT, LIMIT] has a closed-form
# evidence, because the Gaussian's mass sits ~5 sigma inside the box:
#
#   Z = integral prior(x) L(x) dx = (2 pi)^(d/2) sigma^d / (2 * LIMIT)^d
#
# so log Z = 0.5*d*log(2 pi) + d*log(sigma) - d*log(2*LIMIT). This is the *whole point* of SMC — the tempering
# increments must reproduce it — so it is asserted rather than merely smoke-tested.
# --------------------------------------------------------------------------

TARGET_SIGMA = 1.0
TARGET_LIMIT = 5.0
TARGET_N_DIM = 2


def analytic_log_evidence(
    n_dim: int = TARGET_N_DIM, sigma: float = TARGET_SIGMA, limit: float = TARGET_LIMIT
) -> float:
    return (
        0.5 * n_dim * math.log(2.0 * math.pi)
        + n_dim * math.log(sigma)
        - n_dim * math.log(2.0 * limit)
    )


class GaussianTargetAnalysis(af.Analysis):
    """A zero-centred isotropic Gaussian log-likelihood, written through `self._xp` so it is JAX-traceable
    (and therefore differentiable) when the analysis is built with `use_jax=True`."""

    def log_likelihood_function(self, instance):
        xp = self._xp
        values = xp.asarray([instance.one, instance.two])
        return -0.5 * xp.sum((values / TARGET_SIGMA) ** 2)


def gaussian_target_model():
    model = af.Model(af.m.MockClassx2)
    model.one = af.UniformPrior(lower_limit=-TARGET_LIMIT, upper_limit=TARGET_LIMIT)
    model.two = af.UniformPrior(lower_limit=-TARGET_LIMIT, upper_limit=TARGET_LIMIT)
    return model


# --------------------------------------------------------------------------
# Construction / configuration (no jax needed).
# --------------------------------------------------------------------------


def test__defaults():
    search = af.SMC()

    assert search.num_particles == 256
    assert search.kernel == "mala"
    assert search.num_mcmc_steps == 5
    assert search.num_integration_steps == 8
    assert search.target_ess == 0.5
    assert search.step_size is None
    assert search.whiten_inflate == 2.0
    assert search.max_smc_steps == 200
    assert search.batch_size == 0
    assert search.seed == 42
    assert search.inverse_mass_matrix == "none"
    assert search.is_warm_start is False
    # Cold SMC's evidence is only valid if the initial particles are prior draws, so — unlike every other
    # `AbstractMCMC` search — the default initializer is `InitializerPrior`, not `InitializerBall`.
    assert isinstance(search.initializer, af.InitializerPrior)
    assert search.auto_correlation_settings.check_for_convergence is False


def test__explicit_params():
    search = af.SMC(
        num_particles=512,
        kernel="hmc",
        num_mcmc_steps=3,
        num_integration_steps=16,
        target_ess=0.9,
        step_size=0.05,
        whiten_inflate=3.0,
        max_smc_steps=42,
        batch_size=64,
        seed=2024,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        number_of_cores=1,
    )

    assert search.num_particles == 512
    assert search.kernel == "hmc"
    assert search.num_mcmc_steps == 3
    assert search.num_integration_steps == 16
    assert search.target_ess == 0.9
    assert search.step_size == 0.05
    assert search.whiten_inflate == 3.0
    assert search.max_smc_steps == 42
    assert search.batch_size == 64
    assert search.seed == 2024
    assert isinstance(search.initializer, af.InitializerBall)


def test__bad_kernel_raises():
    with pytest.raises(ValueError):
        af.SMC(kernel="nuts")


class TestInverseMassMatrix:
    def test__none_is_cold(self):
        search = af.SMC(inverse_mass_matrix=None)
        assert search.inverse_mass_matrix == "none"
        assert search.is_warm_start is False

    def test__1d_array(self):
        search = af.SMC(inverse_mass_matrix=np.array([1.0, 2.0, 3.0]))
        assert search.inverse_mass_matrix == "array"
        assert search.is_warm_start is True

    def test__2d_array(self):
        search = af.SMC(inverse_mass_matrix=np.eye(3))
        assert search.inverse_mass_matrix == "array"
        assert search.is_warm_start is True

    def test__result_like(self):
        search = af.SMC(inverse_mass_matrix=_FakeSamples(covariance_matrix=np.eye(3)))
        assert search.inverse_mass_matrix == "result"
        assert search.is_warm_start is True

    def test__string_kinds_rejected(self):
        # `BlackJAXNUTS` accepts these — they name an *adaptation* strategy. SMC does not adapt its metric, so
        # accepting them would silently give a cold (prior-whitened) run.
        for kind in ("diagonal", "dense"):
            with pytest.raises(ValueError):
                af.SMC(inverse_mass_matrix=kind)

    def test__bad_type_raises(self):
        with pytest.raises(ValueError):
            af.SMC(inverse_mass_matrix=object())


def test__test_mode_reduces_iterations(monkeypatch):
    # `autofit.non_linear.test_mode` reads PYAUTO_TEST_MODE; `apply_test_mode` is only triggered inside
    # `__init__` when that's set, so the env var must be patched (not the instance mutated after the fact).
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.SMC(num_particles=4096, num_mcmc_steps=25, max_smc_steps=500)

    assert search.num_particles == 16
    assert search.num_mcmc_steps == 2
    assert search.max_smc_steps == 5


def test__test_mode_leaves_small_particle_count_alone(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.SMC(num_particles=8)

    assert search.num_particles == 8


def test__identifier_fields_distinguish_run_shape():
    a = af.SMC(num_particles=256, num_mcmc_steps=5, target_ess=0.5)
    b = af.SMC(num_particles=512, num_mcmc_steps=5, target_ess=0.5)
    c = af.SMC(num_particles=256, num_mcmc_steps=5, target_ess=0.5, kernel="hmc")

    assert af.SMC.__identifier_fields__ == (
        "num_particles",
        "kernel",
        "num_mcmc_steps",
        "num_integration_steps",
        "target_ess",
        "inverse_mass_matrix",
    )

    from autofit.mapper.identifier import Identifier

    assert str(Identifier(a)) != str(Identifier(b))
    assert str(Identifier(a)) != str(Identifier(c))


# --------------------------------------------------------------------------
# Whitening, step size and evidence bookkeeping (pure numpy).
# --------------------------------------------------------------------------


def test__prior_scales_and_limits_from_model():
    model = gaussian_target_model()

    scales = prior_scales_from(model=model)
    lower, upper = prior_limits_from(model=model)

    # Half the central 68% interval of Uniform(-5, 5) is 0.68 * 5.
    assert scales == pytest.approx([3.4135, 3.4135], rel=1e-3)
    assert lower == pytest.approx([-TARGET_LIMIT, -TARGET_LIMIT])
    assert upper == pytest.approx([TARGET_LIMIT, TARGET_LIMIT])


def test__log_prior_normalisation_from_model():
    model = gaussian_target_model()

    # `UniformPrior.log_prior_from_value` drops -log(b - a) per parameter.
    assert log_prior_normalisation_from(model=model) == pytest.approx(
        -2.0 * math.log(2.0 * TARGET_LIMIT)
    )


def test__whitening_cold_is_the_prior_scale_diagonal():
    whitening, kind = whitening_from(
        covariance_seed=None, prior_scales=np.array([2.0, 4.0]), inflate=2.0
    )

    assert whitening == pytest.approx(np.diag([2.0, 4.0]))
    assert "cold" in kind


def test__whitening_diagonal_covariance_takes_the_square_root_and_inflates():
    whitening, kind = whitening_from(
        covariance_seed=np.array([4.0, 9.0]), prior_scales=np.ones(2), inflate=2.0
    )

    assert whitening == pytest.approx(np.diag([4.0, 6.0]))
    assert kind == "diagonal covariance"


def test__whitening_full_covariance_is_the_cholesky_factor():
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])

    whitening, kind = whitening_from(
        covariance_seed=covariance, prior_scales=np.ones(2), inflate=1.0
    )

    # L L^T must reproduce the covariance — this is what decorrelates as well as rescales, which diagonal
    # whitening cannot do (the measured |r| = 0.95 tilted ridge).
    assert whitening @ whitening.T == pytest.approx(covariance)
    assert kind == "full-covariance Cholesky"


def test__auto_step_size_mala_is_a_squared_length():
    # MALA's proposal is x + eps*grad + sqrt(2*eps)*xi, so eps is a SQUARED length: the auto step must be
    # ell^2 / 2 with ell = 2.38 * d^(-1/6) * sigma, not sigma itself.
    n_dim = 15
    sigma = 0.5

    step = auto_step_size_from(target_width=sigma, n_dim=n_dim, kernel="mala")
    length = 2.38 * n_dim ** (-1.0 / 6.0) * sigma

    assert step == pytest.approx(0.5 * length**2)
    # The proposal length it implies is the target width times the optimal constant — NOT eps itself.
    assert math.sqrt(2.0 * step) == pytest.approx(length)


def test__auto_step_size_hmc_is_a_length():
    n_dim = 15
    sigma = 0.5

    step = auto_step_size_from(target_width=sigma, n_dim=n_dim, kernel="hmc")

    assert step == pytest.approx(sigma * n_dim ** (-0.25))


def test__effective_step_size_targets_the_posterior_width_when_warm():
    # Warm, the reference is inflated so it COVERS the posterior: in whitened units the reference has width 1
    # but the posterior has width 1/inflate. Targeting the reference width overshoots by inflate^2.
    warm = af.SMC(inverse_mass_matrix=np.eye(3), whiten_inflate=2.0)
    cold = af.SMC()

    assert warm.effective_step_size(n_dim=3) == pytest.approx(
        auto_step_size_from(target_width=0.5, n_dim=3, kernel="mala")
    )
    assert cold.effective_step_size(n_dim=3) == pytest.approx(
        auto_step_size_from(target_width=0.1, n_dim=3, kernel="mala")
    )


def test__explicit_step_size_overrides_the_auto_scaling():
    search = af.SMC(step_size=0.123, inverse_mass_matrix=np.eye(3))

    assert search.effective_step_size(n_dim=3) == 0.123


def test__ess_from_weights():
    assert ess_from_weights(np.ones(100)) == pytest.approx(100.0)

    # One particle carrying all the weight is one effective sample.
    degenerate = np.zeros(100)
    degenerate[0] = 1.0
    assert ess_from_weights(degenerate) == pytest.approx(1.0)


def test__log_evidence_jacobian_applies_only_to_the_warm_bridge():
    # Cold, pi_0(z) = prior(x(z)) and the whitening Jacobian cancels between the numerator and denominator of
    # the SMC ratio. Warm, pi_0(z) = N(0, I) is already normalised, so log|det L| must be added back.
    assert _log_evidence_from(
        log_evidence_bridge=-10.0, log_jacobian=3.0, is_warm_start=False
    ) == pytest.approx(-10.0)
    assert _log_evidence_from(
        log_evidence_bridge=-10.0, log_jacobian=3.0, is_warm_start=True
    ) == pytest.approx(-7.0)


def test__build_search_internal_shape_contract():
    num_particles, n_dim = 7, 3

    rng = np.random.default_rng(0)

    search_internal = _build_search_internal(
        particles=rng.normal(size=(num_particles, n_dim)),
        weights=np.full(num_particles, 1.0 / num_particles),
        log_likelihood_list=rng.normal(size=num_particles),
        lambda_list=[0.3, 1.0],
        log_likelihood_increment_list=[-1.0, -2.0],
        acceptance_rate_list=[0.8, 0.4],
        ess_list=[5.0, 6.0],
        max_log_likelihood_list=[-3.0, -1.0],
        log_evidence=-3.0,
        log_jacobian=0.5,
        num_particles=num_particles,
        kernel="mala",
        num_mcmc_steps=5,
        num_integration_steps=8,
        target_ess=0.5,
        step_size=0.01,
        whitening_kind="full-covariance Cholesky",
        is_warm_start=True,
        warm_start_source="InitializerParamStartPoints",
        inverse_mass_matrix_kind="result",
        n_smc_steps=2,
        max_smc_steps=200,
    )

    assert search_internal["particles"].shape == (num_particles, n_dim)
    assert search_internal["weights"].shape == (num_particles,)
    assert search_internal["log_likelihood_list"].shape == (num_particles,)
    assert search_internal["converged"] is True
    assert search_internal["warm_start_source"] == "InitializerParamStartPoints"
    assert search_internal["inverse_mass_matrix_kind"] == "result"


def test__build_search_internal_not_converged_when_lambda_short_of_one():
    search_internal = _build_search_internal(
        particles=np.zeros((2, 1)),
        weights=np.ones(2),
        log_likelihood_list=np.zeros(2),
        lambda_list=[0.02],
        log_likelihood_increment_list=[-1.0],
        acceptance_rate_list=[0.0],
        ess_list=[1.0],
        max_log_likelihood_list=[-1.0],
        log_evidence=-1.0,
        log_jacobian=0.0,
        num_particles=2,
        kernel="mala",
        num_mcmc_steps=1,
        num_integration_steps=8,
        target_ess=0.5,
        step_size=0.01,
        whitening_kind="prior-scale diagonal (cold)",
        is_warm_start=False,
        warm_start_source="InitializerPrior",
        inverse_mass_matrix_kind="none",
        n_smc_steps=1,
        max_smc_steps=1,
    )

    assert search_internal["converged"] is False


@requires_blackjax
def test__samples_via_internal_from_uses_the_particle_weights():
    model = gaussian_target_model()

    num_particles = 5
    rng = np.random.default_rng(3)
    particles = rng.uniform(-1.0, 1.0, size=(num_particles, model.prior_count))
    weights = np.array([1.0, 2.0, 3.0, 4.0, 10.0])

    search_internal = _build_search_internal(
        particles=particles,
        weights=weights,
        log_likelihood_list=rng.normal(size=num_particles),
        lambda_list=[0.5, 1.0],
        log_likelihood_increment_list=[-1.0, -1.5],
        acceptance_rate_list=[0.7, 0.5],
        ess_list=[4.0, 3.0],
        max_log_likelihood_list=[-2.0, -1.0],
        log_evidence=-2.5,
        log_jacobian=0.0,
        num_particles=num_particles,
        kernel="mala",
        num_mcmc_steps=5,
        num_integration_steps=8,
        target_ess=0.5,
        step_size=0.01,
        whitening_kind="prior-scale diagonal (cold)",
        is_warm_start=False,
        warm_start_source="InitializerPrior",
        inverse_mass_matrix_kind="none",
        n_smc_steps=2,
        max_smc_steps=200,
    )

    search = af.SMC(num_particles=num_particles)
    samples = search.samples_via_internal_from(
        model=model, search_internal=search_internal
    )

    assert len(samples.sample_list) == num_particles
    # SMC particles are WEIGHTED samples — the normalised importance weights, not a uniform 1.0, are what the
    # PDF, medians and errors are computed from.
    assert samples.weight_list == pytest.approx((weights / weights.sum()).tolist())
    assert samples.log_evidence == pytest.approx(-2.5)
    assert samples.lambda_list == [0.5, 1.0]
    assert samples.acceptance_rate_list == [0.7, 0.5]
    assert samples.ess_list == [4.0, 3.0]
    assert samples.converged is True
    assert samples.samples_info["whitening_kind"] == "prior-scale diagonal (cold)"


# --------------------------------------------------------------------------
# End-to-end: the evidence must match the analytic value.
# --------------------------------------------------------------------------


@requires_blackjax
def test__cold_run_recovers_the_analytic_log_evidence():
    search = af.SMC(
        num_particles=1000,
        num_mcmc_steps=10,
        target_ess=0.8,
        seed=1,
    )

    samples = search.fit(
        model=gaussian_target_model(),
        analysis=GaussianTargetAnalysis(use_jax=True),
    ).samples

    assert samples.converged is True
    assert samples.log_evidence == pytest.approx(analytic_log_evidence(), abs=0.1)

    # `converged` alone is a weak criterion — adaptive tempering can force a single jump to lambda = 1 once
    # acceptance has collapsed. Pin the trace that makes the evidence trustworthy.
    assert len(samples.lambda_list) >= 3
    assert min(samples.acceptance_rate_list) > 0.2

    # The particles are posterior samples of a zero-centred Gaussian.
    assert samples.median_pdf(as_instance=False) == pytest.approx(
        [0.0, 0.0], abs=5.0 * TARGET_SIGMA / math.sqrt(1000)
    )


@requires_blackjax
def test__warm_run_recovers_the_analytic_log_evidence_through_the_bridge():
    # Warm-started, tempering starts from a normalised Gaussian REFERENCE rather than the prior, and lambda
    # still starts at 0. This is the leg that would silently destroy log Z if the particles were simply dropped
    # at the warm-start point, so it is asserted against the same analytic value as the cold run.
    search = af.SMC(
        num_particles=1000,
        num_mcmc_steps=10,
        target_ess=0.8,
        seed=1,
        initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
        inverse_mass_matrix=np.eye(TARGET_N_DIM) * TARGET_SIGMA**2,
    )

    samples = search.fit(
        model=gaussian_target_model(),
        analysis=GaussianTargetAnalysis(use_jax=True),
    ).samples

    assert samples.converged is True
    assert samples.log_evidence == pytest.approx(analytic_log_evidence(), abs=0.1)
    assert samples.samples_info["is_warm_start"] is True
    assert samples.samples_info["whitening_kind"] == "full-covariance Cholesky"
    # The whitening Jacobian is booked into the evidence on the warm path only.
    assert samples.samples_info["log_jacobian"] == pytest.approx(
        TARGET_N_DIM * math.log(2.0 * TARGET_SIGMA)
    )


@requires_blackjax
def test__hmc_kernel_runs_and_holds_acceptance():
    # The HMC inner kernel holds acceptance far higher as lambda -> 1 than MALA does, at ~4x the evaluations.
    search = af.SMC(
        num_particles=500,
        kernel="hmc",
        num_mcmc_steps=3,
        num_integration_steps=8,
        target_ess=0.8,
        seed=1,
        initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
        inverse_mass_matrix=np.eye(TARGET_N_DIM) * TARGET_SIGMA**2,
    )

    samples = search.fit(
        model=gaussian_target_model(),
        analysis=GaussianTargetAnalysis(use_jax=True),
    ).samples

    assert samples.converged is True
    assert samples.log_evidence == pytest.approx(analytic_log_evidence(), abs=0.1)
    assert min(samples.acceptance_rate_list) > 0.5


@requires_blackjax
def test__non_jax_analysis_raises_a_clear_error():
    search = af.SMC(num_particles=8, num_mcmc_steps=1, max_smc_steps=1)

    with pytest.raises(ValueError, match="use_jax=True"):
        search.fit(
            model=gaussian_target_model(),
            analysis=GaussianTargetAnalysis(),
        )


@requires_blackjax
def test__test_mode_runs_end_to_end(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.SMC(num_particles=4096, num_mcmc_steps=25, max_smc_steps=500, seed=1)

    samples = search.fit(
        model=gaussian_target_model(),
        analysis=GaussianTargetAnalysis(use_jax=True),
    ).samples

    # Test mode is about shape and speed, not accuracy: 16 particles over at most 5 temperatures.
    assert len(samples.sample_list) == 16
    assert 1 <= samples.samples_info["n_smc_steps"] <= 5
    assert np.isfinite(samples.log_evidence)
