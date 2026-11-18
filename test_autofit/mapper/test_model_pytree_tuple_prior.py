import pytest

import autofit as af
from autofit.mapper.prior.tuple_prior import TuplePrior

jax = pytest.importorskip("jax")

from autofit.jax import register_model


class DatasetModel:
    """
    The shape of autoarray's ``DatasetModel``: a scalar plus a 2-tuple, so
    the model has no direct priors when the scalar is fixed and its tuple
    priors sit under a ``TuplePrior``.
    """

    def __init__(self, background_sky_level=0.0, grid_offset=(0.0, 0.0)):
        self.background_sky_level = background_sky_level
        self.grid_offset = grid_offset


def _make_model(free_grid_offset, free_sky):
    model = af.Model(DatasetModel)
    model.background_sky_level = 0.0
    model.grid_offset.grid_offset_0 = 0.0
    model.grid_offset.grid_offset_1 = 0.0
    if free_grid_offset:
        model.grid_offset.grid_offset_0 = af.UniformPrior(-1.0, 1.0)
        model.grid_offset.grid_offset_1 = af.UniformPrior(-1.0, 1.0)
    if free_sky:
        model.background_sky_level = af.UniformPrior(0.0, 1.0)
    return model


@pytest.mark.parametrize(
    "free_grid_offset, free_sky, prior_count",
    [
        (False, False, 0),
        (True, False, 2),
        (True, True, 3),
    ],
)
def test_model_with_tuple_prior_round_trips(free_grid_offset, free_sky, prior_count):
    model = _make_model(free_grid_offset, free_sky)
    register_model(model)

    assert model.prior_count == prior_count

    leaves, treedef = jax.tree_util.tree_flatten(model)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.prior_count == prior_count
    assert isinstance(rebuilt.grid_offset, TuplePrior)


def test_tuple_prior_round_trips_free_and_fixed_members():
    register_model(af.Model(DatasetModel))

    tuple_prior = TuplePrior(
        grid_offset_0=af.UniformPrior(-1.0, 1.0),
        grid_offset_1=0.5,
    )

    leaves, treedef = jax.tree_util.tree_flatten(tuple_prior)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, TuplePrior)
    assert len(rebuilt.priors) == 1
    assert rebuilt.grid_offset_1 == 0.5
    assert rebuilt.value_for_arguments({rebuilt.grid_offset_0: 0.25}) == (0.25, 0.5)
