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
