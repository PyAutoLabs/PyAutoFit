import numpy as np
import pytest
from scipy.stats import norm

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
