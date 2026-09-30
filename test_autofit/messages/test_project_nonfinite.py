"""
A non-finite importance-weighted projection must raise a named, catchable
exception that says which input was non-finite (PyAutoFit#1653).

`AbstractMessage.project` used to end on a bare
`assert np.isfinite(suff_stats).all()`. An `AssertionError` is not in the
recovery tuple of `factor_step`, so one factor's bad projection killed a
51-factor EP run (RAL 342411), and the message gave no hint of the cause.
Under `python -O` the assert vanished and the non-finite statistics flowed on
silently.

The exception is a `ValueError` subclass, which is what `factor_step`
recovers from, so these tests match on `ValueError` and then check the
concrete type: against the old assert they fail with the `AssertionError`
itself.
"""

import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.messages.normal import NormalMessage

nan = np.nan
inf = np.inf


@pytest.mark.parametrize(
    "samples, log_weight_list, match",
    [
        ([1.0, 2.0, nan], [0.0, 0.0, 0.0], "non-finite samples"),
        ([1.0, inf, 3.0], [0.0, 0.0, 0.0], "non-finite samples"),
        ([1.0, 2.0, 3.0], [0.0, nan, 0.0], "log weights are nan or \\+inf"),
        ([1.0, 2.0, 3.0], [0.0, inf, 0.0], "log weights are nan or \\+inf"),
        ([1.0, 2.0, 3.0], [-inf, -inf, -inf], "all log weights are -inf"),
        ([1e200, 2.0, 3.0], [0.0, 0.0, 0.0], "overflow"),
    ],
    ids=[
        "nan_sample",
        "inf_sample",
        "nan_log_weight",
        "pos_inf_log_weight",
        "all_neg_inf_log_weights",
        "overflow",
    ],
)
def test_nonfinite_projection_raises_projection_exception(
    samples, log_weight_list, match
):
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match=match) as info:
            NormalMessage.project(np.array(samples), np.array(log_weight_list))

    assert isinstance(info.value, exc.ProjectionException)
    assert not isinstance(info.value, AssertionError)
    assert issubclass(exc.ProjectionException, ValueError)
    assert "NormalMessage.project: non-finite sufficient statistics" in str(
        info.value
    )


def test_gaussian_prior_projection_names_the_prior_id():
    prior = af.GaussianPrior(mean=0.0, sigma=1.0)

    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="non-finite samples") as info:
            prior.project(np.array([1.0, 2.0, nan]), np.zeros(3))

    assert isinstance(info.value, exc.ProjectionException)
    assert f"id={prior.id}" in str(info.value)
    assert "GaussianPrior" in str(info.value)


def test_uniform_prior_projection_names_the_prior_id():
    """
    `UniformPrior`'s message is a `TransformedMessage`, whose `project` drops
    `id_` before reaching `AbstractMessage.project` — the id must still reach
    the error, so it is added at `Prior.project`.
    """
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    with np.errstate(all="ignore"):
        with pytest.raises(ValueError) as info:
            prior.project(np.array([0.2, 0.5, nan]), np.zeros(3))

    assert isinstance(info.value, exc.ProjectionException)
    assert f"id={prior.id}" in str(info.value)
    assert "UniformPrior" in str(info.value)


def test_finite_projection_is_unchanged():
    projected = NormalMessage.project(np.array([1.0, 2.0, 3.0]), np.zeros(3))

    assert projected.mean == pytest.approx(2.0)
    assert np.isfinite(projected.sigma)
