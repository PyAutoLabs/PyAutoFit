import pickle
from copy import copy

import numpy as np
import pytest

import autofit as af
from autofit.mapper.identifier import Identifier


@pytest.fixture(name="log_gaussian")
def make_log_gaussian():
    return af.LogGaussianPrior(mean=1.0, sigma=2.0)


@pytest.mark.parametrize(
    "unit, value",
    [
        # (0.0, 0.0),
        (0.1, 0.2),
        (0.5, 2.7),
        (0.9, 35),
    ],
)
def test_values(log_gaussian, unit, value):
    assert log_gaussian.value_for(unit) == pytest.approx(value, rel=0.1)


def test_pickle(log_gaussian):
    loaded = pickle.loads(pickle.dumps(log_gaussian))
    assert loaded == log_gaussian

def test_identifier(log_gaussian):
    Identifier(log_gaussian)


# === PyAutoFit#1526: the prior declares its own (0, inf) support ===
#
# The support was always (0, inf) -- ``log_prior_from_value`` returns -inf for
# ``value <= 0`` -- but the prior reported (-inf, inf), because ``Prior.__getattr__``
# delegated to a ``TransformedMessage`` whose limits default to +/-inf and were never
# set. ``ClipperPriorBox`` worked around it with an ``isinstance`` switch; every other
# consumer of ``lower_limit`` was simply told the wrong thing.
#
# The pinned values below are the pre-change ones, measured on the commit before the
# fix. They are the guarantee that declaring the support moved *what the prior says*
# and nothing about *what it computes*.


def test__reports_its_own_support(log_gaussian):
    assert log_gaussian.lower_limit == 0.0
    assert log_gaussian.upper_limit == float("inf")
    assert log_gaussian.limits == (0.0, float("inf"))


def test__lower_bound_is_strict__upper_is_not(log_gaussian):
    """
    The support is the *open* (0, inf): ``log_prior_from_value(0.0)`` is -inf, so a
    consumer that clips onto the reported bound must know to stay strictly above it.
    """
    assert log_gaussian.lower_limit_strict is True
    assert log_gaussian.upper_limit_strict is False

    assert log_gaussian.log_prior_from_value(log_gaussian.lower_limit) == -np.inf


def test__other_prior_families_keep_their_limits():
    """
    ``Prior.limits`` now derives from ``lower_limit``/``upper_limit`` rather than
    returning a hardcoded (-inf, inf). GaussianPrior must still be unbounded.
    """
    assert af.GaussianPrior(mean=0.0, sigma=1.0).limits == (-np.inf, np.inf)
    assert af.UniformPrior(lower_limit=0.0, upper_limit=2.0).limits == (0.0, 2.0)
    assert af.LogUniformPrior(lower_limit=0.01, upper_limit=100.0).limits == (
        0.01,
        100.0,
    )

    for cls in (af.GaussianPrior, af.UniformPrior, af.LogUniformPrior):
        assert cls.lower_limit_strict is False
        assert cls.upper_limit_strict is False


@pytest.mark.parametrize(
    "value, expected",
    [
        (-3.0, -np.inf),
        (-1e-09, -np.inf),
        (0.0, -np.inf),
        (1e-12, -204.83588563011642),
        (0.001, -8.892033838618836),
        (0.1, 0.14164835190717184),
        (0.5, 0.3396055360729164),
        (1.0, -0.04733727810650888),
        (2.0, -0.7185718164978876),
        (10.0, -3.3735407249713125),
        (1000.0, -19.437601069254285),
    ],
)
def test__log_prior_from_value_is_unchanged(value, expected):
    """
    The density was always correct; only the reported limits were wrong. Pinned
    against the pre-change values either side of zero.
    """
    prior = af.LogGaussianPrior(mean=0.4, sigma=1.3)
    assert prior.log_prior_from_value(value) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "unit, expected",
    [
        (1e-09, 0.0006129978595719644),
        (0.0001, 0.011858368820420245),
        (0.01, 0.07249394511581292),
        (0.1, 0.2819523947584061),
        (0.25, 0.6207439038545902),
        (0.5, 1.4918246976412703),
        (0.75, 3.5852803622761034),
        (0.9, 7.893321602745905),
        (0.99, 30.69968015862632),
        (0.999999999, 3630.585196715403),
    ],
)
def test__unit_cube_mapping_is_unchanged(unit, expected):
    """
    The nested samplers work in unit-cube coordinates and map through the prior. The
    limits live on the prior while the mapping lives on the message stack, which this
    change does not touch -- so no stored nested-sampling result shifts.
    """
    prior = af.LogGaussianPrior(mean=0.4, sigma=1.3)
    assert prior.value_for(unit) == pytest.approx(expected, rel=1e-12)
    assert prior.unit_value_for(prior.value_for(unit)) == pytest.approx(unit, rel=1e-6)


def test__identifier_is_unchanged():
    """
    If the declared limits fed the identifier, every existing output directory would
    re-key and its stored results would be orphaned. ``__identifier_fields__`` is
    ("mean", "sigma"), and the limits are instance/class attributes outside it, so the
    hash is untouched. Pinned to the pre-change value.
    """
    prior = af.LogGaussianPrior(mean=0.4, sigma=1.3)
    assert str(Identifier(prior)) == "34cb61ade6bafa6050229e8b6b390235"


def test__declared_support_survives_copy_pickle_and_projection(log_gaussian):
    """
    The limits are derived in ``__init__`` rather than carried as parameters, which is
    what makes them survive every path that rebuilds the prior -- including the JAX
    pytree round-trip, where only (mean, sigma, id) are flattened.
    """
    assert copy(log_gaussian).lower_limit == 0.0
    assert pickle.loads(pickle.dumps(log_gaussian)).lower_limit == 0.0

    samples = np.exp(np.random.default_rng(0).normal(1.0, 2.0, 500))
    projected = log_gaussian.project(samples, np.zeros(500))
    assert projected.lower_limit == 0.0

    rebuilt = af.LogGaussianPrior.tree_unflatten((), log_gaussian.tree_flatten()[0])
    assert rebuilt.lower_limit == 0.0
