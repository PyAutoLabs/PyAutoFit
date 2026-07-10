"""Regression tests for the EP statistics fix batch (PyAutoFit #1350;
findings F1/F2/F4/F8 from the #1332 audit). Numpy-only.
"""
import numpy as np
import pytest

from autofit.graphical.mean_field import MeanField
from autofit.mapper.variable import Variable
from autofit.messages.abstract import AbstractMessage
from autofit.messages.beta import BetaMessage
from autofit.messages.gamma import GammaMessage
from autofit.messages.normal import NormalMessage


# --- F1: MeanField __truediv__ / __pow__ keep log_norm in the log_norm slot ---

def _mean_field(log_norm):
    v = Variable("x")
    return MeanField({v: NormalMessage(0.0, 1.0)}, log_norm=log_norm), v


def test__mean_field_truediv_preserves_log_norm():
    mf1, v = _mean_field(3.0)
    mf2 = MeanField({v: NormalMessage(0.5, 2.0)}, log_norm=1.0)
    out = mf1 / mf2
    # Previously: log_norm silently 0.0 and the float 2.0 stored in _plates.
    assert out.log_norm == 3.0 - 1.0
    assert not isinstance(out.plates, float)


def test__mean_field_pow_scalar_preserves_log_norm():
    mf1, _ = _mean_field(3.0)
    out = mf1 ** 2.0
    assert out.log_norm == 6.0
    assert not isinstance(out.plates, float)


def test__mean_field_pow_mean_field_exponent_constructs_cleanly():
    mf1, v = _mean_field(3.0)
    exponents = MeanField({v: 0.5}, log_norm=2.0)
    out = mf1 ** exponents
    # No meaningful scalar aggregate exists for per-variable exponents; the
    # previous self.log_norm * other.log_norm (= 6.0) was meaningless and
    # landed in the plates slot.
    assert out.log_norm == 0.0
    assert not isinstance(out.plates, float)


# --- F2: KL direction contract — message.kl(other) == KL(message || other),
# verified per family by Monte Carlo E_self[logpdf_self - logpdf_other] ---

def _mc_kl(p, q, n=400_000, seed=1):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        x = p.sample(n)
    finally:
        np.random.set_state(rng_state)
    return float(np.mean(p.logpdf(x) - q.logpdf(x)))


@pytest.mark.parametrize(
    "p, q",
    [
        (NormalMessage(0.0, 1.0), NormalMessage(1.0, 2.0)),
        (GammaMessage(2.0, 1.0), GammaMessage(5.0, 3.0)),
        (BetaMessage(2.0, 5.0), BetaMessage(4.0, 2.0)),
    ],
    ids=["normal", "gamma", "beta"],
)
def test__kl_direction_contract(p, q):
    analytic = float(p.kl(q))
    mc = _mc_kl(p, q)
    # Direction reversal changes these asymmetric KLs by far more than the MC
    # noise (~1e-2), so this test pins the KL(self || other) contract.
    assert analytic == pytest.approx(mc, abs=0.05)
    # And the reverse direction is genuinely different (asymmetric cases).
    assert abs(float(q.kl(p)) - analytic) > 0.05


# --- F4: update_invalid — both branches, scalar branch un-flagged ---

def test__update_invalid_scalar_branch():
    good = NormalMessage(1.0, 2.0)
    invalid = NormalMessage.from_natural_parameters(np.array([np.nan, -0.5]))
    assert not invalid.check_valid()
    repaired = invalid.update_invalid(good)
    assert repaired.check_valid()
    assert (repaired.mean, repaired.sigma) == (1.0, 2.0)
    # valid scalar keeps itself
    kept = good.update_invalid(NormalMessage(9.0, 9.0))
    assert (kept.mean, kept.sigma) == (1.0, 2.0)


def test__update_invalid_array_branch():
    good = NormalMessage(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    mixed = NormalMessage.from_natural_parameters(
        np.array([[0.5, np.nan], [-0.5, -0.5]])
    )
    valid = mixed.check_valid()
    assert valid.tolist() == [True, False]
    repaired = mixed.update_invalid(good)
    assert repaired.check_valid().all()
    # element 0 kept, element 1 replaced by good's parameters
    assert repaired.mean[1] == 1.0 and repaired.sigma[1] == 2.0


# --- F8: dead quasi-Newton variants are gone; exported ones remain ---

def test__dead_newton_variants_removed():
    from autofit.graphical.laplace import newton

    for dead in ("diag_sr1_update_", "diag_sr1_bfgs_update", "bfgs1_update"):
        assert not hasattr(newton, dead)
    for alive in ("bfgs_update", "sr1_update", "full_bfgs_update"):
        assert hasattr(newton, alive)
