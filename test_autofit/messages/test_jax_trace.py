import numpy as np
import pytest

from autofit.messages.beta import BetaMessage
from autofit.messages.gamma import GammaMessage
from autofit.messages.normal import NaturalNormal, NormalMessage
from autofit.messages.truncated_normal import (
    TruncatedNaturalNormal,
    TruncatedNormalMessage,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


MESSAGE_ARRAY_CASES = [
    pytest.param(
        lambda value, xp: NormalMessage.calc_natural_parameters(
            value, value + 1.0, xp=xp
        ),
        id="normal-natural-parameters",
    ),
    pytest.param(
        lambda value, xp: NormalMessage.to_canonical_form(value, xp=xp),
        id="normal-canonical-form",
    ),
    pytest.param(
        lambda value, xp: NaturalNormal.calc_natural_parameters(value, -value, xp=xp),
        id="natural-normal-natural-parameters",
    ),
    pytest.param(
        lambda value, xp: TruncatedNormalMessage.calc_natural_parameters(
            value, value + 1.0, xp=xp
        ),
        id="truncated-normal-natural-parameters",
    ),
    pytest.param(
        lambda value, xp: TruncatedNormalMessage.to_canonical_form(value, xp=xp),
        id="truncated-normal-canonical-form",
    ),
    pytest.param(
        lambda value, xp: TruncatedNaturalNormal.calc_natural_parameters(
            value, -value, xp=xp
        ),
        id="truncated-natural-normal-natural-parameters",
    ),
    pytest.param(
        lambda value, xp: BetaMessage.calc_natural_parameters(
            value + 1.0, value + 2.0, xp=xp
        ),
        id="beta-natural-parameters",
    ),
    pytest.param(
        lambda value, xp: BetaMessage.to_canonical_form(value, xp=xp),
        id="beta-canonical-form",
    ),
    pytest.param(
        lambda value, xp: GammaMessage.calc_natural_parameters(
            value + 1.0, value + 2.0, xp=xp
        ),
        id="gamma-natural-parameters",
    ),
    pytest.param(
        lambda value, xp: GammaMessage.to_canonical_form(value, xp=xp),
        id="gamma-canonical-form",
    ),
]


@pytest.mark.parametrize("message_array", MESSAGE_ARRAY_CASES)
@pytest.mark.parametrize(
    "value",
    [pytest.param(0.25, id="scalar"), pytest.param([0.25, 0.5], id="batched")],
)
def test_message_array_construction_is_jittable_and_matches_numpy(message_array, value):
    numpy_value = np.asarray(value)
    expected = message_array(numpy_value, np)

    actual = jax.jit(lambda traced: message_array(traced, jnp))(jnp.asarray(value))

    assert actual.shape == expected.shape
    np.testing.assert_allclose(np.asarray(actual), expected)


MESSAGE_LOG_PARTITION_CASES = [
    pytest.param(
        lambda value, xp: GammaMessage(value + 1.0, value + 2.0).log_partition(xp=xp),
        id="gamma",
    ),
    pytest.param(
        lambda value, xp: BetaMessage(value + 1.0, value + 2.0).log_partition(xp=xp),
        id="beta",
    ),
]


@pytest.mark.parametrize("message_log_partition", MESSAGE_LOG_PARTITION_CASES)
@pytest.mark.parametrize(
    "value",
    [pytest.param(2.0, id="scalar"), pytest.param([2.0, 3.0], id="batched")],
)
def test_message_log_partition_is_jittable_and_matches_numpy(
    message_log_partition, value
):
    numpy_value = np.asarray(value)
    expected = message_log_partition(numpy_value, np)

    actual = jax.jit(lambda traced: message_log_partition(traced, jnp))(
        jnp.asarray(value)
    )

    assert actual.shape == np.shape(expected)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


MESSAGE_PARITY_CASES = [
    pytest.param(
        lambda params, xp: NormalMessage(xp.asarray(params), xp.asarray(params) + 1.0),
        0.5,
        id="normal",
    ),
    pytest.param(
        lambda params, xp: BetaMessage(
            xp.asarray(params) + 1.0, xp.asarray(params) + 2.0
        ),
        0.25,
        id="beta",
    ),
    pytest.param(
        lambda params, xp: GammaMessage(
            xp.asarray(params) + 1.0, xp.asarray(params) + 2.0
        ),
        0.5,
        id="gamma",
    ),
]


@pytest.mark.parametrize("make_message, x_offset", MESSAGE_PARITY_CASES)
@pytest.mark.parametrize(
    "params",
    [pytest.param(1.0, id="scalar"), pytest.param([1.0, 2.0], id="batched")],
)
def test_message_shape_and_logpdf_match_numpy(make_message, x_offset, params):
    """
    A JAX-backed message must report the same shape/size/ndim as its NumPy
    twin, and batched logpdf must return the same values and shape.

    Guards the `()` shape sentinel regression: with `shape` hard-wired to `()`
    for the JAX branch, `_broadcast_natural_parameters` matched the
    `shape[1:] == self.shape` branch for batched messages, inserted a spurious
    axis and returned an (n, n) matrix of wrong values instead of the (n,)
    vector NumPy produces (#1510).
    """
    numpy_message = make_message(np.asarray(params), np)
    jax_message = make_message(jnp.asarray(params), jnp)

    assert jax_message.shape == numpy_message.shape
    assert jax_message.size == numpy_message.size
    assert jax_message.ndim == numpy_message.ndim

    x = np.asarray(params) * 0.0 + x_offset
    expected = numpy_message.logpdf(x, xp=np)

    actual = jax_message.logpdf(jnp.asarray(x), xp=jnp)

    assert np.shape(actual) == np.shape(expected)
    # rtol reflects float32 accumulation in the JAX natural_logpdf path.
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4)
