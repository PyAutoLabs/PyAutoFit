"""Regression tests for the width-modifier safety pair (PyAutoFit #1346,
Phase 2 of decision hub #1331 — Decisions 5 + 2). Numpy-only.
"""
import pytest

import autofit as af
from autofit import exc
from autofit.mapper.prior.width_modifier import (
    RelativeWidthModifier,
    WidthModifier,
)
from autofit.messages.normal import NormalMessage
from autofit.messages.truncated_normal import TruncatedNormalMessage


# --- Decision 5: RelativeWidthModifier uses abs(mean), optional absolute_floor ---

def test__relative_width_modifier_abs_mean():
    mod = RelativeWidthModifier(0.5)
    # A negative posterior median previously produced a negative sigma that
    # flowed silently into the passed prior and flipped its scale.
    assert mod(-2.0) == 1.0
    assert mod(2.0) == 1.0
    # The bare modifier still returns 0.0 at mean=0; prior passing rejects it
    # loudly downstream (see the chained tests below).
    assert mod(0.0) == 0.0


def test__relative_width_modifier_floor():
    mod = RelativeWidthModifier(0.5, absolute_floor=0.1)
    assert mod(0.0) == 0.1     # floor engages at zero median
    assert mod(0.1) == 0.1     # 0.5 * 0.1 = 0.05 < floor
    assert mod(10.0) == 5.0    # a floor, not a cap


def test__relative_width_modifier_dict_round_trip():
    mod = RelativeWidthModifier(0.5, absolute_floor=0.1)
    assert mod.dict == {"type": "Relative", "value": 0.5, "absolute_floor": 0.1}
    assert WidthModifier.from_dict(mod.dict) == mod

    bare = RelativeWidthModifier(0.5)
    assert bare.dict == {"type": "Relative", "value": 0.5}
    assert WidthModifier.from_dict(bare.dict) == bare
    assert bare != mod


# --- Decision 2 (evidence-adjusted): both message classes now agree — sigma < 0
# rejected, sigma == 0 permitted as the established point-mass idiom (latent
# variables' simple_model_for_kwargs, from_mode(covariance=0),
# model_centred_relative at mean=0 all depend on it) ---

def test__normal_message_rejects_negative_sigma():
    with pytest.raises(exc.MessageException):
        NormalMessage(mean=0.0, sigma=-1.0)


def test__truncated_normal_message_rejects_negative_sigma():
    with pytest.raises(exc.MessageException):
        TruncatedNormalMessage(
            mean=0.0, sigma=-1.0, lower_limit=-1.0, upper_limit=1.0
        )


def test__sigma_zero_point_mass_still_constructs():
    # The point-mass carrier used by the latent-variables machinery
    # (non_linear/samples/util.py) and from_mode(covariance=0) must keep working.
    m = NormalMessage(mean=3.0, sigma=0.0)
    assert m.sigma == 0.0
    p = af.GaussianPrior(mean=3.0, sigma=0.0)
    assert p.sigma == 0.0


def test__gaussian_prior_rejects_negative_sigma():
    # Previously constructed silently with deceptive variance = sigma**2 > 0
    # and a sign-flipped value_for.
    with pytest.raises(exc.MessageException):
        af.GaussianPrior(mean=0.0, sigma=-0.5)


# --- The mean=0 chained-parameter regression (the Phase-2 sequencing gate:
# yesterday's silent delta-freeze must become a clear, parameter-named error,
# and the floor must be the working remedy) ---

def _mapper_with_relative_widths(absolute_floor=None):
    mapper = af.ModelMapper(mock_class=af.m.MockClassx2)
    for prior in mapper.priors:
        prior.width_modifier = RelativeWidthModifier(
            0.5, absolute_floor=absolute_floor
        )
    return mapper


def test__prior_passing_mean_zero_raises_with_guidance():
    mapper = _mapper_with_relative_widths()
    with pytest.raises(exc.PriorException) as err:
        mapper.mapper_from_prior_means([0.0, 5.0])
    # The error must name the parameter and point at the remedy.
    assert "mock_class" in str(err.value)
    assert "absolute_floor" in str(err.value)


def test__prior_passing_mean_zero_with_floor_passes():
    mapper = _mapper_with_relative_widths(absolute_floor=0.1)
    result = mapper.mapper_from_prior_means([0.0, 5.0])
    assert result.mock_class.one.mean == 0.0
    assert result.mock_class.one.sigma == 0.1   # floor engaged
    assert result.mock_class.two.sigma == 2.5   # 0.5 * 5.0, floor irrelevant


def test__prior_passing_negative_mean_gets_positive_width():
    mapper = _mapper_with_relative_widths()
    result = mapper.mapper_from_prior_means([-2.0, 5.0])
    assert result.mock_class.one.mean == -2.0
    assert result.mock_class.one.sigma == 1.0   # 0.5 * abs(-2.0)
