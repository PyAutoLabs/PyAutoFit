"""Tests for ``autofit.jax.enable_pytrees`` / ``register_model``."""
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import autofit as af
from autofit.jax import enable_pytrees, register_model


@pytest.fixture(name="model")
def make_model():
    return af.Model(
        af.ex.Gaussian,
        centre=af.GaussianPrior(mean=1.0, sigma=1.0),
        normalization=af.GaussianPrior(mean=2.0, sigma=1.0),
        sigma=af.GaussianPrior(mean=3.0, sigma=1.0),
    )


def test_enable_pytrees_returns_true_when_jax_available():
    assert enable_pytrees() is True


def test_register_model_round_trip(model):
    register_model(model)
    instance = model.instance_from_prior_medians()

    leaves, treedef = jax.tree_util.tree_flatten(instance)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, af.ex.Gaussian)
    assert float(rebuilt.centre) == float(instance.centre)
    assert float(rebuilt.normalization) == float(instance.normalization)
    assert float(rebuilt.sigma) == float(instance.sigma)


def test_register_model_works_under_jit(model):
    register_model(model)
    instance = model.instance_from_prior_medians()

    @jax.jit
    def total(inst):
        return inst.centre + inst.normalization + inst.sigma

    assert float(total(instance)) == pytest.approx(
        instance.centre + instance.normalization + instance.sigma
    )


def test_register_model_keeps_constants_static():
    """Constants from the model definition must NOT be traced under JIT.

    Galaxy.redshift is a common case: it sits in the constructor signature
    but gets a fixed value via ``af.Constant``. If it became a JAX tracer,
    code paths like ``sorted(galaxies, key=lambda g: g.redshift)`` would
    blow up with ``TracerBoolConversionError``. The ``aux_data`` partition
    in ``register_model`` keeps it as a Python float.
    """
    class Holder:
        def __init__(self, redshift, scale):
            self.redshift = redshift
            self.scale = scale

    model = af.Model(
        Holder,
        redshift=0.5,
        scale=af.GaussianPrior(mean=1.0, sigma=1.0),
    )
    register_model(model)
    instance = model.instance_from_prior_medians()
    assert isinstance(instance.redshift, float)

    @jax.jit
    def use_redshift_for_control_flow(inst):
        # `if` on a traced value would raise TracerBoolConversionError;
        # this only works if redshift is kept static via aux_data.
        if inst.redshift > 0:
            return inst.scale * 2
        return inst.scale

    result = use_redshift_for_control_flow(instance)
    assert float(result) == pytest.approx(2.0 * instance.scale)


def test_enable_pytrees_idempotent():
    assert enable_pytrees() is True
    assert enable_pytrees() is True


def test_collection_round_trip(model):
    register_model(model)
    collection = af.Collection(g1=model, g2=model)
    register_model(collection)

    instance = collection.instance_from_prior_medians()
    leaves, treedef = jax.tree_util.tree_flatten(instance)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert float(rebuilt.g1.centre) == float(instance.g1.centre)
    assert float(rebuilt.g2.sigma) == float(instance.g2.sigma)
