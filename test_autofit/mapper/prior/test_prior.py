import math
import warnings

import numpy as np
import pytest

import autofit as af
from autofit import exc


class TestPriorLimits:
    def test_out_of_order_prior_limits(self):
        with pytest.raises(af.exc.PriorException):
            af.UniformPrior(1.0, 0)

    def test_prior_creation(self):
        mapper = af.ModelMapper()
        mapper.component = af.m.MockClassx2

        prior_tuples = mapper.prior_tuples_ordered_by_id

        assert prior_tuples[0].prior.lower_limit == 0
        assert prior_tuples[0].prior.upper_limit == 1

        assert prior_tuples[1].prior.lower_limit == 0
        assert prior_tuples[1].prior.upper_limit == 2

    def test_inf(self):
        mm = af.ModelMapper()
        mm.mock_class_inf = af.m.MockClassInf

        prior_tuples = mm.prior_tuples_ordered_by_id

        assert prior_tuples[0].prior.lower_limit == float("-inf")
        assert prior_tuples[0].prior.upper_limit == 0

        assert prior_tuples[1].prior.lower_limit == 0
        assert prior_tuples[1].prior.upper_limit == float("inf")

        assert mm.instance_from_vector([-10000, 10000]) is not None

    def test_preserve_limits_tuples(self):
        mm = af.ModelMapper()
        mm.mock_class_gaussian = af.m.MockClassx2

        new_mapper = mm.mapper_from_prior_means(
            means=[0.0, 0.0],
        )

        prior_tuples = new_mapper.prior_tuples_ordered_by_id

        assert prior_tuples[0].prior.lower_limit == 0
        assert prior_tuples[0].prior.upper_limit == 1

        assert prior_tuples[1].prior.lower_limit == 0
        assert prior_tuples[1].prior.upper_limit == 2

    def test__only_use_widths_to_pass_priors(self):
        mm = af.ModelMapper()
        mm.mock_class_gaussian = af.m.MockClassx2

        new_mapper = mm.mapper_from_prior_means(
            means=[5.0, 5.0],
        )

        prior_tuples = new_mapper.prior_tuples_ordered_by_id

        assert prior_tuples[0].prior.mean == 5.0
        assert prior_tuples[0].prior.sigma == 1.0

        assert prior_tuples[1].prior.mean == 5.0
        assert prior_tuples[1].prior.sigma == 2.0

    def test_from_gaussian_no_limits(self):
        mm = af.ModelMapper()
        mm.mock_class_gaussian = af.m.MockClassx2

        new_mapper = mm.mapper_from_prior_means(
            [(0.0, 0.5), (0.0, 1)], no_limits=True
        )

        priors = new_mapper.priors
        assert priors[0].lower_limit == float("-inf")
        assert priors[0].upper_limit == float("inf")
        assert priors[1].lower_limit == float("-inf")
        assert priors[1].upper_limit == float("inf")


class TestPriorMean:
    def test_simple(self):
        uniform_prior = af.UniformPrior(0.0, 1.0)
        assert uniform_prior.mean == 0.5

    def test_higher(self):
        uniform_prior = af.UniformPrior(1.0, 2.0)
        assert uniform_prior.mean == 1.5


class TestAddition:
    def test_abstract_plus_abstract(self):
        one = af.AbstractModel()
        two = af.AbstractModel()
        one.a = "a"
        two.b = "b"

        three = one + two

        assert three.a == "a"
        assert three.b == "b"

    def test_list_properties(self):
        one = af.AbstractModel()
        two = af.AbstractModel()
        one.a = ["a"]
        two.a = ["b"]

        three = one + two

        assert three.a == ["a", "b"]

    def test_instance_plus_instance(self):
        one = af.ModelInstance()
        two = af.ModelInstance()
        one.a = "a"
        two.b = "b"

        three = one + two

        assert three.a == "a"
        assert three.b == "b"

    def test_mapper_plus_mapper(self):
        one = af.ModelMapper()
        two = af.ModelMapper()
        one.a = af.Model(af.m.MockClassx2)
        two.b = af.Model(af.m.MockClassx2)

        three = one + two

        assert three.prior_count == 4


class TestUniformPrior:
    def test__simple_assumptions(self):
        uniform_simple = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

        assert uniform_simple.value_for(0.0) == 0.0
        assert uniform_simple.value_for(1.0) == 1.0
        assert uniform_simple.value_for(0.5) == 0.5

    def test__non_zero_lower_limit(self):
        uniform_half = af.UniformPrior(lower_limit=0.5, upper_limit=1.0)

        assert uniform_half.value_for(0.0) == 0.5
        assert uniform_half.value_for(1.0) == 1.0
        assert uniform_half.value_for(0.5) == 0.75

    def test_width(self):
        assert af.UniformPrior(2, 5).width == 3

    def test_negative_range(self):
        prior = af.UniformPrior(-1, 0)
        assert prior.width == 1
        assert prior.value_for(0.0) == -1
        assert prior.value_for(1.0) == 0.0

    def test__log_prior_from_value(self):
        gaussian_simple = af.UniformPrior(lower_limit=-40, upper_limit=70)

        log_prior = gaussian_simple.log_prior_from_value(value=0.0)

        assert log_prior == 0.0

        log_prior = gaussian_simple.log_prior_from_value(value=11.0)

        assert log_prior == 0.0


class TestLogUniformPrior:
    def test__simple_assumptions(self):
        log_uniform_simple = af.LogUniformPrior(lower_limit=1.0e-8, upper_limit=1.0)

        assert log_uniform_simple.value_for(0.0) == 1.0e-8
        assert log_uniform_simple.value_for(1.0) == 1.0
        assert log_uniform_simple.value_for(0.5) == pytest.approx(0.0001, abs=0.000001)

    def test__non_zero_lower_limit(self):
        log_uniform_half = af.LogUniformPrior(lower_limit=0.5, upper_limit=1.0)

        assert log_uniform_half.value_for(0.0) == 0.5
        assert log_uniform_half.value_for(1.0) == 1.0
        assert log_uniform_half.value_for(0.5) == pytest.approx(0.70710678118, 1.0e-4)

    def test__log_prior_from_value(self):
        # LogUniformPrior log-density: -log(value), dropping the normalisation
        # constant -log(log(upper / lower)). Consistent with UniformPrior's
        # convention of returning 0.0 (dropping -log(b - a)).
        log_uniform = af.LogUniformPrior(lower_limit=1e-8, upper_limit=1.0)

        assert log_uniform.log_prior_from_value(value=1.0) == 0.0
        assert log_uniform.log_prior_from_value(value=2.0) == pytest.approx(
            -np.log(2.0), 1.0e-12
        )
        assert log_uniform.log_prior_from_value(value=4.0) == pytest.approx(
            -np.log(4.0), 1.0e-12
        )

        # The normalisation constant being dropped means the returned values
        # do NOT depend on the (lower_limit, upper_limit) pair — only on `value`.
        log_uniform = af.LogUniformPrior(lower_limit=50.0, upper_limit=100.0)

        assert log_uniform.log_prior_from_value(value=1.0) == 0.0
        assert log_uniform.log_prior_from_value(value=2.0) == pytest.approx(
            -np.log(2.0), 1.0e-12
        )
        assert log_uniform.log_prior_from_value(value=4.0) == pytest.approx(
            -np.log(4.0), 1.0e-12
        )

    def test__log_prior_from_value__non_positive_returns_neg_inf(self):
        # Regression (PyAutoHeart #27 / release run 28784914443): Emcee's stretch
        # move proposes physical values that can go non-positive for a LogUniform
        # parameter. `-log(value)` of a non-positive value is NaN, which propagated
        # into the summed figure-of-merit and crashed the search with
        # "ValueError: Probability function returned NaN". Non-positive values must
        # return -inf (zero density -> the move is rejected), and evaluating the
        # log-prior must not emit a NumPy "invalid value in log" RuntimeWarning.
        log_uniform = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            assert log_uniform.log_prior_from_value(value=-1.0) == float("-inf")
            assert log_uniform.log_prior_from_value(value=0.0) == float("-inf")

        # Positive values are unchanged: the NumPy path stays unnormalised and
        # unbounded, returning -log(value) regardless of (lower_limit, upper_limit).
        assert log_uniform.log_prior_from_value(value=10.0) == pytest.approx(
            -np.log(10.0), 1.0e-12
        )
        assert log_uniform.log_prior_from_value(value=1.0e4) == pytest.approx(
            -np.log(1.0e4), 1.0e-12
        )

    def test__lower_limit_zero_or_below_raises_error(self):
        with pytest.raises(exc.PriorException):
            af.LogUniformPrior(lower_limit=-1.0, upper_limit=1.0)

        with pytest.raises(exc.PriorException):
            af.LogUniformPrior(lower_limit=0.0, upper_limit=1.0)


class TestGaussianPrior:
    def test__simple_assumptions(self):
        gaussian_simple = af.GaussianPrior(mean=0.0, sigma=1.0)

        assert gaussian_simple.value_for(0.1) == pytest.approx(-1.281551, 1.0e-4)
        assert gaussian_simple.value_for(0.9) == pytest.approx(1.281551, 1.0e-4)
        assert gaussian_simple.value_for(0.5) == 0.0

    def test__non_zero_mean(self):
        gaussian_half = af.GaussianPrior(mean=0.5, sigma=2.0)

        assert gaussian_half.value_for(0.1) == pytest.approx(-2.0631031, 1.0e-4)
        assert gaussian_half.value_for(0.9) == pytest.approx(3.0631031, 1.0e-4)
        assert gaussian_half.value_for(0.5) == 0.5

    @pytest.mark.parametrize(
        "mean, sigma, value, expected",
        [
            # Density-form log-prior: -(value - mean)**2 / (2 * sigma**2), with
            # the -log(sigma * sqrt(2 * pi)) normalisation constant dropped.
            # Maximum at value == mean (returns 0), negative elsewhere.
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0, -0.5),
            (0.0, 1.0, 2.0, -2.0),
            (1.0, 2.0, 0.0, -0.125),
            (1.0, 2.0, 1.0, 0.0),
            (1.0, 2.0, 2.0, -0.125),
            (30.0, 60.0, 2.0, pytest.approx(-0.108888, 1.0e-4)),
        ],
    )
    def test__log_prior_from_value(self, mean, sigma, value, expected):
        gaussian = af.GaussianPrior(mean=mean, sigma=sigma)
        log_prior = gaussian.log_prior_from_value(value=value)
        assert log_prior == expected


def test_log_gaussian_prior_log_prior_from_value():
    log_gaussian_prior = af.LogGaussianPrior(
        mean=0.0, sigma=1.0,
    )

    assert log_gaussian_prior.log_prior_from_value(value=0.0) == float("-inf")
    # Density form: -(log(value) - mean)**2 / (2 * sigma**2) - log(value),
    # where the second term is the Jacobian of the log-space transform.
    log_half = math.log(0.5)
    expected = -(log_half ** 2) / 2.0 - log_half
    assert log_gaussian_prior.log_prior_from_value(value=0.5) == pytest.approx(
        expected, 1.0e-12
    )
