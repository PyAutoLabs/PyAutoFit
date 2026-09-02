"""
PyAutoFit#1559 (D5): ``TransformedMessage.from_mode`` must push the physical
covariance through the Jacobian of *every* transform in the stack, for 0-d and
n-d covariances alike, whether given as a float, an array or a ``DiagonalMatrix``.
"""
import numpy as np
import pytest

import autofit as af
from autofit.mapper.operator import DiagonalMatrix
from autofit.mapper.variable import Variable
from autofit.mapper.variable_operator import VariableFullOperator


@pytest.fixture(name="log_message")
def make_log_message():
    # Physical variance 25 at mode 10 is log-space variance 0.25 (sigma 0.5)
    return af.LogGaussianPrior(mean=np.log(10.0), sigma=0.5).message


def test__zero_d_diagonal_matrix_applies_the_jacobian(log_message):
    result = log_message.from_mode(np.asarray(10.0), DiagonalMatrix(np.asarray(25.0)))

    assert result.base_message.mean == pytest.approx(np.log(10.0))
    assert result.base_message.sigma == pytest.approx(0.5)


def test__one_d_diagonal_matrix_matches_the_zero_d_path(log_message):
    result = log_message.from_mode(np.asarray([10.0]), DiagonalMatrix(np.asarray([25.0])))

    assert result.base_message.sigma == pytest.approx([0.5])


def test__plain_float_matches_the_operator_path(log_message):
    result = log_message.from_mode(np.asarray(10.0), 25.0)

    assert result.base_message.sigma == pytest.approx(0.5)


def test__variable_full_operator_extraction_matches(log_message):
    # ``MeanField.from_mode_covariance`` hands over ``covar.get(v)``, a 0-d
    # DiagonalMatrix extracted from the Laplace optimiser's full operator.
    v = Variable("v")
    covar = VariableFullOperator.from_diagonal({v: np.asarray(25.0)})

    result = log_message.from_mode(np.asarray(10.0), covar[v])

    assert result.base_message.sigma == pytest.approx(0.5)


def test__physical_limits_never_reach_the_base_message(log_message):
    result = log_message.from_mode(
        np.asarray(10.0), 25.0, lower_limit=0.0, upper_limit=np.inf
    )

    assert result.base_message.sigma == pytest.approx(0.5)
    assert (result.base_message.lower_limit, result.base_message.upper_limit) == (
        -np.inf,
        np.inf,
    )


@pytest.mark.parametrize("mode, variance", [(5.0, 0.04), (2.0, 0.01), (8.0, 0.0025)])
def test__two_transform_stack_round_trips_the_variance(mode, variance):
    # UniformPrior composes phi_transform with a LinearShiftTransform; the
    # loop used to keep only the innermost Jacobian (variance 4.0 for 0.04).
    u = af.UniformPrior(0.0, 10.0).message

    result = u.from_mode(np.asarray(mode), variance)

    assert result.mean == pytest.approx(mode)
    assert result.variance == pytest.approx(variance)


def test__two_transform_stack_one_d_matches_zero_d():
    u = af.UniformPrior(0.0, 10.0).message

    one_d = u.from_mode(np.asarray([5.0]), np.asarray([0.04]))
    zero_d = u.from_mode(np.asarray(5.0), DiagonalMatrix(np.asarray(0.04)))

    assert one_d.variance == pytest.approx([0.04])
    assert zero_d.variance == pytest.approx(0.04)
