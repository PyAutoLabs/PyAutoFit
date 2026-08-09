import numpy as np
import pytest

import autofit as af

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


COMPOUND_PRIOR_CASES = [
    pytest.param(lambda prior: af.Log(prior), id="log"),
    pytest.param(lambda prior: af.Log10(prior), id="log10"),
    pytest.param(
        lambda prior: af.Log(prior) + af.Log10(prior),
        id="compound-children",
    ),
    pytest.param(lambda prior: af.Log(af.Log(prior)), id="nested-modifier"),
]


@pytest.mark.parametrize("compound_prior_from", COMPOUND_PRIOR_CASES)
@pytest.mark.parametrize(
    "value",
    [pytest.param(4.0, id="scalar"), pytest.param([4.0, 10.0], id="batched")],
)
def test_compound_prior_is_jittable_and_matches_numpy(compound_prior_from, value):
    prior = af.GaussianPrior(mean=0.0, sigma=1.0)
    compound_prior = compound_prior_from(prior)

    def evaluate(prior_value, xp):
        return compound_prior.instance_for_arguments(
            {prior: prior_value},
            ignore_assertions=True,
            xp=xp,
        )

    numpy_value = np.asarray(value)
    expected = evaluate(numpy_value, np)
    actual = jax.jit(lambda traced: evaluate(traced, jnp))(jnp.asarray(value))

    assert actual.shape == np.shape(expected)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)
