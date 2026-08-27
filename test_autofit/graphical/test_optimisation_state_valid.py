"""
``OptimisationState.valid`` — the limits guard on the Laplace line search.

Written alongside the guard's rewrite from ``if self.lower_limit`` to
``if self.lower_limit is not None``. The property had **no test at all** before
this file, so the suite would have stayed green through any change to it — the
process lesson recorded in ``prior-support-clipper`` (PyAutoFit#1477), where a
1790-test suite passed against an ``LBFGS._fit`` that raised ``NameError`` on
every real call.

The guards test ``is not None`` rather than truthiness because the intent is
"was a limit supplied?". ``lower_limit`` / ``upper_limit`` are ``VariableData``
(a ``dict`` keyed by free variable) or ``None``; ``if x`` expressed that only by
accident of dict truthiness being non-emptiness.
"""

import numpy as np
import pytest

from autofit.graphical.laplace.line_search import OptimisationState
from autofit.mapper.variable import Variable, VariableData


@pytest.fixture(name="x")
def make_variable():
    return Variable("x")


def _state(parameters, lower_limit=None, upper_limit=None):
    """
    An ``OptimisationState`` carrying only what ``valid`` reads.

    ``__init__`` evaluates ``self.valid`` and, when invalid, replaces value and
    gradient with ``inf`` — which is why the factor callables below must be
    tolerated rather than invoked. ``valid`` itself touches none of them.
    """
    return OptimisationState(
        factor=lambda *_: None,
        factor_gradient=lambda *_: None,
        parameters=parameters,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )


def test__no_limits_supplied__always_valid(x):
    """``check_limits=False`` leaves both limits ``None``; nothing is enforced."""
    state = _state(VariableData({x: np.array([-1e9, 1e9])}))

    assert state.lower_limit is None
    assert state.upper_limit is None
    assert state.valid is True


def test__empty_limits__valid(x):
    """
    The case the old truthiness guard and the new ``is not None`` guard resolve
    differently *in the guard*, and identically *in the answer*.

    An empty ``VariableData`` is falsy, so the old guard skipped the comparison;
    the new guard runs it, and an empty comparison's ``.any()`` is ``False``. A
    model with no free variables is valid either way — this pins that the rewrite
    did not change it.
    """
    state = _state(
        VariableData({x: np.array([-5.0])}),
        lower_limit=VariableData({}),
        upper_limit=VariableData({}),
    )

    assert state.valid is True


def test__inside_the_limits__valid(x):
    state = _state(
        VariableData({x: np.array([0.5, 1.5])}),
        lower_limit=VariableData({x: np.array([0.0, 0.0])}),
        upper_limit=VariableData({x: np.array([2.0, 2.0])}),
    )

    assert state.valid is True


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (np.array([-0.1, 1.0]), False),  # below the lower limit
        (np.array([1.0, 2.1]), False),  # above the upper limit
        (np.array([0.0, 2.0]), True),  # exactly on both — inclusive
    ],
)
def test__limits_are_enforced_and_inclusive(x, parameters, expected):
    state = _state(
        VariableData({x: parameters}),
        lower_limit=VariableData({x: np.array([0.0, 0.0])}),
        upper_limit=VariableData({x: np.array([2.0, 2.0])}),
    )

    assert state.valid is expected


def test__a_zero_lower_limit_is_still_enforced(x):
    """
    The regression the rewrite exists for.

    Under the old ``if self.lower_limit`` guard this is safe only because
    ``VariableData`` is a ``dict``. PyAutoFit#1527's follow-up list read the guard
    as a scalar test and recorded a `0.0` bug that did not exist — but the guard
    was one type change away from making it real. Pinning a lower limit of
    exactly ``0.0`` keeps that honest: a scalar-valued limit of ``0.0`` must
    never mean "no limit".
    """
    state = _state(
        VariableData({x: np.array([-1.0])}),
        lower_limit=VariableData({x: np.array([0.0])}),
    )

    assert state.valid is False


# === VariableData.any ===


def test__variable_data_any_reduces_any_not_all(x):
    """
    ``VariableData.any`` dispatched through ``var_all``, which made it "is there a
    variable whose elements are ALL True" rather than "is ANY element True".

    This is why ``OptimisationState.valid`` above could not see a *partial*
    violation: for ``array([True, False])`` the old reduction answered ``False``,
    so a parameter vector with one component outside its limits was accepted. The
    guard rewrite alone does not fix that — this does.
    """
    assert VariableData({x: np.array([True, False])}).any() is True
    assert VariableData({x: np.array([False, False])}).any() is False
    assert VariableData({x: np.array([True, True])}).any() is True
    assert VariableData({}).any() is False

    # ``all`` is the counterpart and was always correct; pinned so a future
    # edit cannot "fix" one by breaking the other.
    assert VariableData({x: np.array([True, False])}).all() is False
    assert VariableData({x: np.array([True, True])}).all() is True
