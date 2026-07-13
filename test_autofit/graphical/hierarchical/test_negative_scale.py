import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit import graphical as g
from autofit.graphical.declarative.factor.hierarchical import Factor
from autofit.messages.normal import NormalMessage


def _factor():
    """A HierarchicalFactor whose scale hyper-prior is truncated at zero, matching
    the EP guide (`GaussianPrior` distribution drawn from truncated mean/sigma)."""
    hf = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.TruncatedGaussianPrior(
            mean=2.0, sigma=1.0, lower_limit=0.0, upper_limit=100.0
        ),
        sigma=af.TruncatedGaussianPrior(
            mean=0.5, sigma=0.5, lower_limit=0.0, upper_limit=100.0
        ),
    )
    hf.add_drawn_variable(af.GaussianPrior(mean=2.0, sigma=1.0))
    return hf


def test_negative_scale_returns_neg_inf_not_raise():
    """
    Regression for the EP nightly crash: an EP factor optimiser proposing a
    (transiently) negative scale for the `GaussianPrior` distribution — just below
    the truncated scale hyper-prior's `lower_limit=0` — must yield a zero-density
    (`-inf`) factor value rather than raising `MessageException` from the strict
    `NormalMessage` sigma guard.
    """
    hf = _factor()
    mean_p, scale_p = hf.mean, hf.sigma
    factor = Factor(hf)

    def call(scale):
        return factor(
            **{
                f"prior_{mean_p.id}": 2.0,
                f"prior_{scale_p.id}": scale,
                "argument": 2.0,
            }
        )

    # a valid scale gives a finite log-density
    assert np.isfinite(call(0.5))

    # negative scales are outside the distribution's support → zero density
    assert call(-0.016) == -np.inf
    assert call(-0.17) == -np.inf


def test_normal_message_negative_sigma_guard_still_raises():
    """The fix relies on the strict guard staying loud for genuine misuse (prior
    passing); constructing a `NormalMessage` with a negative sigma must still raise."""
    with pytest.raises(exc.MessageException):
        NormalMessage(mean=1.0, sigma=-0.1)
