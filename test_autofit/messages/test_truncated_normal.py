import pickle

import numpy as np
import pytest
from scipy.stats import norm

from autofit import exc
from autofit.messages.normal import NormalMessage
from autofit.messages.truncated_normal import (
    TruncatedNaturalNormal,
    TruncatedNormalMessage,
)


@pytest.fixture
def truncated_message():
    # Bounds chosen so the inner test points are well inside the support
    # but Z is meaningfully less than 1 (Z ≈ 0.34).
    return TruncatedNormalMessage(mean=0.0, sigma=1.0, lower_limit=-1.0, upper_limit=0.5)


@pytest.fixture
def reference_normal():
    return NormalMessage(mean=0.0, sigma=1.0)


def _expected_log_Z(message):
    a = (message.lower_limit - message.mean) / message.sigma
    b = (message.upper_limit - message.mean) / message.sigma
    return float(np.log(norm.cdf(b) - norm.cdf(a)))


def test_in_support_gradient_and_hessian_match_normal(
    truncated_message, reference_normal
):
    x = np.array([-0.5, 0.0, 0.25])

    _, grad_t, hess_t = truncated_message.logpdf_gradient_hessian(x)
    _, grad_n, hess_n = reference_normal.logpdf_gradient_hessian(x)

    assert np.allclose(grad_t, grad_n)
    # Hessian for a univariate Gaussian is the same scalar at every x.
    assert np.allclose(hess_t, hess_n)


def test_in_support_logl_differs_by_minus_log_Z(
    truncated_message, reference_normal
):
    x = np.array([-0.5, 0.0, 0.25])

    logl_t, _, _ = truncated_message.logpdf_gradient_hessian(x)
    logl_n, _, _ = reference_normal.logpdf_gradient_hessian(x)

    assert np.allclose(logl_t, logl_n - _expected_log_Z(truncated_message))


def test_out_of_support_array(truncated_message):
    x = np.array([-2.0, -0.5, 1.5])  # below, inside, above

    logl, grad, _ = truncated_message.logpdf_gradient_hessian(x)

    assert np.isneginf(logl[0])
    assert np.isfinite(logl[1])
    assert np.isneginf(logl[2])

    assert grad[0] == 0.0
    assert grad[2] == 0.0
    assert grad[1] != 0.0


def test_out_of_support_scalar_below(truncated_message):
    logl, grad, _ = truncated_message.logpdf_gradient_hessian(-2.0)
    assert np.isneginf(logl)
    assert grad == 0.0


def test_out_of_support_scalar_above(truncated_message):
    logl, grad, _ = truncated_message.logpdf_gradient_hessian(1.5)
    assert np.isneginf(logl)
    assert grad == 0.0


def test_scalar_returns_scalar(truncated_message):
    logl, grad, hess = truncated_message.logpdf_gradient_hessian(0.0)

    assert np.ndim(logl) == 0
    assert np.ndim(grad) == 0
    assert np.ndim(hess) == 0


def test_array_returns_array(truncated_message):
    x = np.array([-0.5, 0.0, 0.25])
    logl, grad, _ = truncated_message.logpdf_gradient_hessian(x)

    assert logl.shape == x.shape
    assert grad.shape == x.shape


def test_logpdf_gradient_returns_two(truncated_message):
    result = truncated_message.logpdf_gradient(np.array([-0.5, 0.0]))
    assert len(result) == 2


def test_numerical_gradient_agreement(truncated_message):
    # Stay strictly inside the support so the truncation indicator's
    # discontinuity at the boundary doesn't leak into the finite-difference
    # estimate.
    x = np.array([-0.6, -0.2, 0.3])

    res = truncated_message.logpdf_gradient(x)
    nres = truncated_message.numerical_logpdf_gradient(x)
    for analytic, numerical in zip(res, nres):
        assert np.allclose(analytic, numerical, rtol=1e-2, atol=1e-2)

    res = truncated_message.logpdf_gradient_hessian(x)
    nres = truncated_message.numerical_logpdf_gradient_hessian(x)
    for analytic, numerical in zip(res, nres):
        assert np.allclose(analytic, numerical, rtol=1e-2, atol=1e-2)


def test_no_truncation_matches_normal():
    # With infinite bounds, log Z = 0 and the truncated message should agree
    # with the untruncated one on logl, gradient, and Hessian.
    truncated = TruncatedNormalMessage(mean=0.5, sigma=1.3)
    normal = NormalMessage(mean=0.5, sigma=1.3)

    x = np.array([-1.0, 0.0, 1.0, 2.0])
    logl_t, grad_t, hess_t = truncated.logpdf_gradient_hessian(x)
    logl_n, grad_n, hess_n = normal.logpdf_gradient_hessian(x)

    assert np.allclose(logl_t, logl_n)
    assert np.allclose(grad_t, grad_n)
    assert np.allclose(hess_t, hess_n)


def test_truncated_natural_normal_finite_gradients():
    # Build via natural parameters: eta1 = mu/sigma^2, eta2 = -1/(2 sigma^2).
    mu_underlying, sigma_underlying = 0.0, 1.0
    eta1 = mu_underlying / sigma_underlying ** 2
    eta2 = -0.5 / sigma_underlying ** 2
    msg = TruncatedNaturalNormal(
        eta1, eta2, lower_limit=-1.0, upper_limit=0.5
    )

    x = np.array([-0.5, 0.0, 0.25])
    logl, grad, hess = msg.logpdf_gradient_hessian(x)

    assert np.all(np.isfinite(logl))
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))


def test_truncated_natural_normal_uses_underlying_mu_sigma():
    # The override on TruncatedNaturalNormal must reconstruct the underlying
    # (mu, sigma) from the natural parameters rather than using
    # self.mean / self.sigma (which return *truncated* moments). Verify by
    # comparing against an equivalent TruncatedNormalMessage built from the
    # same underlying parameters.
    mu_underlying, sigma_underlying = 0.2, 0.8
    eta1 = mu_underlying / sigma_underlying ** 2
    eta2 = -0.5 / sigma_underlying ** 2

    natural = TruncatedNaturalNormal(
        eta1, eta2, lower_limit=-1.0, upper_limit=0.5
    )
    standard = TruncatedNormalMessage(
        mean=mu_underlying,
        sigma=sigma_underlying,
        lower_limit=-1.0,
        upper_limit=0.5,
    )

    x = np.array([-0.4, 0.0, 0.3])
    logl_n, grad_n, hess_n = natural.logpdf_gradient_hessian(x)
    logl_s, grad_s, hess_s = standard.logpdf_gradient_hessian(x)

    assert np.allclose(logl_n, logl_s)
    assert np.allclose(grad_n, grad_s)
    assert np.allclose(hess_n, hess_s)


# === PyAutoFit#1559: truncation limits are support state, carried through every rebuild ===


@pytest.fixture(name="m")
def make_m():
    return TruncatedNormalMessage(10, 5, 0, 100)


def _support_ops():
    m = TruncatedNormalMessage(10, 5, 0, 100)
    other = TruncatedNormalMessage(20, 3, 0, 100)
    return [
        pytest.param(lambda m: m ** 0.2, id="pow"),
        pytest.param(lambda m: m * other, id="multiply"),
        pytest.param(lambda m: m / other, id="divide"),
        pytest.param(
            lambda m: TruncatedNormalMessage.from_natural_parameters(
                m.natural_parameters(), **m._support_kwargs
            ),
            id="from_natural_parameters",
        ),
        pytest.param(lambda m: m.copy(), id="copy"),
        pytest.param(
            lambda m: m.update_invalid(TruncatedNormalMessage(1, 1, 0, 100)),
            id="update_invalid",
        ),
        pytest.param(lambda m: m.natural, id="natural"),
        pytest.param(lambda m: m.zeros_like(), id="zeros_like"),
        pytest.param(lambda m: m * 2.0, id="scalar_multiply"),
        pytest.param(lambda m: m[()], id="getitem"),
        pytest.param(lambda m: pickle.loads(pickle.dumps(m)), id="pickle"),
    ]


@pytest.mark.parametrize("op", _support_ops())
def test__support_survives_natural_parameter_ops(m, op):
    result = op(m)

    assert isinstance(result, TruncatedNormalMessage)
    assert (result.lower_limit, result.upper_limit) == (0.0, 100.0)


def test__parameters_exclude_the_limits(m):
    # The limits are support state, not parameters, so ``parameters`` and
    # ``_parameter_support`` line up.
    assert m.parameters == (10, 5)
    assert len(m.parameters) == len(m._parameter_support)


def test__truncated_times_normal_keeps_truncated_support(m):
    result = m * NormalMessage(20, 3)
    assert isinstance(result, TruncatedNormalMessage)
    assert (result.lower_limit, result.upper_limit) == (0.0, 100.0)

    # Documented asymmetry: the result takes the left operand's class, so a
    # NormalMessage times a truncated one stays unbounded. Inside EP every
    # message on a variable shares the prior's class, so this is user-only.
    assert type(NormalMessage(20, 3) * m) is NormalMessage


def test__mismatched_finite_supports():
    overlap = TruncatedNormalMessage(1, 1, 0, 10) * TruncatedNormalMessage(1, 1, 5, 30)
    assert (overlap.lower_limit, overlap.upper_limit) == (5.0, 10.0)

    with pytest.raises(exc.MessageException):
        TruncatedNormalMessage(1, 1, 0, 10) * TruncatedNormalMessage(1, 1, 20, 30)


def test__divide_keeps_own_support(m):
    result = m / TruncatedNormalMessage(1, 1, 5, 30)
    assert (result.lower_limit, result.upper_limit) == (0.0, 100.0)


def test__from_mode_keeps_class_and_limits():
    result = TruncatedNormalMessage.from_mode(
        3.0, 4.0, lower_limit=0.0, upper_limit=100.0
    )
    assert type(result) is TruncatedNormalMessage
    assert (result.lower_limit, result.upper_limit) == (0.0, 100.0)
    assert result.mean == 3.0
    assert result.sigma == 2.0


def test__update_invalid_vector_truncated():
    # Vector-shaped truncated messages used to raise TypeError here because
    # the limits were positional parameters (float() of an array).
    invalid = TruncatedNormalMessage(
        np.array([1.0, 2.0]), np.array([1.0, np.nan]), 0, 100
    )
    safe = TruncatedNormalMessage(np.array([5.0, 5.0]), np.array([1.0, 1.0]), 0, 100)

    result = invalid.update_invalid(safe)

    assert (result.lower_limit, result.upper_limit) == (0.0, 100.0)
    assert result.mean == pytest.approx([1.0, 5.0])
    assert result.sigma == pytest.approx([1.0, 1.0])


def test__truncated_natural_normal_copy_keeps_limits():
    natural = TruncatedNaturalNormal(0.1, -0.5, lower_limit=0.0, upper_limit=100.0)

    for result in (natural.copy(), pickle.loads(pickle.dumps(natural))):
        assert type(result) is TruncatedNaturalNormal
        assert (result.lower_limit, result.upper_limit) == (0.0, 100.0)


def test__old_three_tuple_pickle_payload_still_loads():
    # Pickles written before #1559 carried the limits as the 3rd and 4th
    # positional parameters and no support element.
    result = TruncatedNormalMessage._reconstruct((10.0, 5.0, 0.0, 100.0), 0.0, 7)
    assert (result.lower_limit, result.upper_limit) == (0.0, 100.0)
    assert result.id == 7

    result = TruncatedNormalMessage._reconstruct((10.0, 5.0), 0.0, 7)
    assert (result.lower_limit, result.upper_limit) == (-np.inf, np.inf)
