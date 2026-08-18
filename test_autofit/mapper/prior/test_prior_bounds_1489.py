"""
Regression tests for PyAutoFit#1489: prior bounds must be enforced in the search
objective on the NumPy path.

Between release 2025.10.16.1 (which removed ``assert_within_limits`` /
``PriorLimitException``) and this fix, ``UniformPrior.log_prior_from_value``
returned ``0.0`` unconditionally on the NumPy path and ``LogUniformPrior``
returned a finite ``-log(value)`` outside ``[lower_limit, upper_limit]``, so
nothing in the fitness path penalised an out-of-support parameter for the
searches that form a log posterior (Emcee, Zeus, Drawer, LBFGS). An Emcee walker
on a poorly-constrained parameter then escaped its declared box without bound.

These tests pin the restored contract: ``-inf`` outside the support on the NumPy
path, matching the JAX path; a ``-inf`` figure of merit for an out-of-box
vector; a NaN-free figure-of-merit inversion; and end-to-end Emcee containment.
"""
import warnings

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness


class TestUniformPriorBounds:
    def test__scalar__zero_inside__neg_inf_outside(self):
        prior = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

        assert prior.log_prior_from_value(value=0.0) == 0.0
        assert prior.log_prior_from_value(value=0.05) == 0.0
        assert prior.log_prior_from_value(value=0.11) == -np.inf
        assert prior.log_prior_from_value(value=-0.11) == -np.inf
        assert prior.log_prior_from_value(value=5.0) == -np.inf

    def test__limits_are_inclusive(self):
        prior = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

        assert prior.log_prior_from_value(value=-0.1) == 0.0
        assert prior.log_prior_from_value(value=0.1) == 0.0

    def test__array_input(self):
        prior = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

        log_prior = prior.log_prior_from_value(value=np.array([0.0, 0.1, 0.2, -5.0]))

        assert log_prior == pytest.approx(np.array([0.0, 0.0, -np.inf, -np.inf]))


class TestLogUniformPriorBounds:
    def test__scalar__log_density_inside__neg_inf_outside(self):
        prior = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)

        assert prior.log_prior_from_value(value=10.0) == pytest.approx(
            -np.log(10.0), 1.0e-12
        )
        # Above the upper limit was previously a finite -log(value) — the gap
        # that left the box unenforced from above.
        assert prior.log_prior_from_value(value=1.0e4) == -np.inf
        # Between zero and the lower limit was previously finite too.
        assert prior.log_prior_from_value(value=1.0e-4) == -np.inf

    def test__non_positive_returns_neg_inf_without_warning(self):
        prior = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            assert prior.log_prior_from_value(value=0.0) == -np.inf
            assert prior.log_prior_from_value(value=-1.0) == -np.inf

    def test__array_input(self):
        prior = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            log_prior = prior.log_prior_from_value(
                value=np.array([10.0, 1.0e4, 1.0e-4, -1.0])
            )

        assert log_prior == pytest.approx(
            np.array([-np.log(10.0), -np.inf, -np.inf, -np.inf])
        )


class GaussianWithOffset:
    def __init__(self, centre=50.0, normalization=1.0, sigma=5.0, offset=0.0):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma
        self.offset = offset  # ignored by the likelihood: deliberately unconstrained

    def model_data_from(self, xvalues):
        transformed = xvalues - self.centre
        return (
            self.normalization
            / (self.sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * (transformed / self.sigma) ** 2)
        )


class AnalysisIgnoringOffset(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()
        self.data = data
        self.noise_map = noise_map
        self.xvalues = np.arange(data.shape[0], dtype=float)

    def log_likelihood_function(self, instance):
        model_data = instance.model_data_from(self.xvalues)
        residual_map = self.data - model_data
        chi_squared = float(np.sum((residual_map / self.noise_map) ** 2.0))
        noise_normalization = float(np.sum(np.log(2 * np.pi * self.noise_map**2.0)))
        return -0.5 * (chi_squared + noise_normalization)


def _make_model_and_analysis():
    rng = np.random.default_rng(1)
    xvalues = np.arange(100, dtype=float)
    truth = GaussianWithOffset(centre=50.0, normalization=25.0, sigma=5.0)
    noise = 0.1
    data = truth.model_data_from(xvalues) + rng.normal(0.0, noise, size=xvalues.shape)
    noise_map = np.full(xvalues.shape, noise)

    model = af.Model(GaussianWithOffset)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.UniformPrior(lower_limit=0.1, upper_limit=25.0)
    model.offset = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    return model, AnalysisIgnoringOffset(data=data, noise_map=noise_map)


class TestFitnessObjective:
    def test__out_of_box_vector__log_posterior_is_neg_inf(self):
        model, analysis = _make_model_and_analysis()

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=None,
            fom_is_log_likelihood=False,
        )

        in_bounds = [50.0, 25.0, 5.0, 0.05]
        out_of_box = [50.0, 25.0, 5.0, 5.0]

        assert np.isfinite(fitness(parameters=in_bounds))
        assert fitness(parameters=out_of_box) == -np.inf

    def test__log_likelihood_from__out_of_box_is_not_nan(self):
        # -inf figure of merit minus -inf log prior is NaN unless guarded; the
        # inversion feeds history / quick-update / resume bookkeeping, which
        # compare against it.
        model, analysis = _make_model_and_analysis()

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=None,
            fom_is_log_likelihood=False,
        )

        out_of_box = [50.0, 25.0, 5.0, 5.0]
        figure_of_merit = fitness(parameters=out_of_box)

        log_likelihood = fitness.log_likelihood_from(
            figure_of_merit=figure_of_merit, parameters=out_of_box
        )

        assert not np.isnan(log_likelihood)
        assert log_likelihood == -np.inf


class TestEmceeContainment:
    def test__unconstrained_parameter_stays_inside_declared_box(self):
        """
        End-to-end: with the strict NumPy-path priors every out-of-box proposal
        has a -inf log posterior, so Emcee rejects it and — because walkers are
        initialised inside the box — containment is guaranteed, independent of
        the random seed. Before the fix, the unconstrained ``offset`` escaped
        ``UniformPrior(-0.1, 0.1)`` in 100% of accepted samples (PyAutoFit#1489).
        """
        model, analysis = _make_model_and_analysis()

        search = af.Emcee(
            name="test_uniform_bounds_containment_1489",
            nwalkers=10,
            nsteps=300,
            iterations_per_full_update=1_000_000,
            number_of_cores=1,
        )

        result = search.fit(model=model, analysis=analysis)

        offset_samples = np.asarray(result.samples.parameter_lists)[:, 3]

        assert offset_samples.size > 0
        assert np.all(offset_samples >= -0.1)
        assert np.all(offset_samples <= 0.1)
