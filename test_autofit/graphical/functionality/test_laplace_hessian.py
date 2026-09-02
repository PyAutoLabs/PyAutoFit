"""
The Laplace projection's covariance is the inverse of a finite-difference
Hessian of the tilted log-density at the mode (PyAutoFit#1561, D2), and a
failed or invalid update leaves the factor's message untouched (D3).
"""
import pickle

import numpy as np
import pytest

from autofit import graphical as graph
from autofit.graphical.laplace import newton
from autofit.graphical.laplace.optimiser import LaplaceOptimiser
from autofit.graphical.utils import Status, StatusFlag
from autofit.mapper.variable import Variable
from autofit.messages import NormalMessage

# Tilted precision of `make_approx`: factor curvature 1/sigma_f^2 = 0.01 on
# both variables (coupled), plus the cavity precisions 1/10^2 and 1/20^2.
P = np.array([[0.02, -0.01], [-0.01, 0.0125]])
COV = np.linalg.inv(P)
P_CAVITY = np.diag([0.01, 0.0025])
MODE = np.linalg.solve(P, P_CAVITY @ np.array([50.0, 55.0]))


def make_approx(sigma_f=10.0, convex=False):
    """
    A two-variable Gaussian factor N(x | mu, sigma_f) with an analytic
    Jacobian, so the finite-difference Hessian is exact to roundoff.
    `convex=True` flips the sign: the tilted density is then not concave.
    """
    mu_, x_ = Variable("mu"), Variable("x")
    sign = 1.0 if convex else -1.0

    def f(mu, x):
        return sign * 0.5 * ((x - mu) / sigma_f) ** 2

    def f_jac(mu, x):
        d = -sign * (x - mu) / sigma_f**2
        return f(mu, x), (d, -d)

    factor = graph.Factor(f, mu_, x_, factor_jacobian=f_jac)
    cavity = graph.MeanField(
        {mu_: NormalMessage(50.0, 10.0), x_: NormalMessage(55.0, 20.0)}
    )
    return graph.FactorApproximation(factor, cavity, cavity, cavity), mu_, x_


def _projection_bits(proj, mu_, x_):
    return tuple(
        float(getattr(proj[v], attr)) for v in (mu_, x_) for attr in ("mean", "variance")
    )


def test__analytic_inverse_hessian_rng_independent():
    results = []
    for seed in (0, 1, 12345):
        fa, mu_, x_ = make_approx()
        np.random.seed(seed)
        proj, status = LaplaceOptimiser().optimise_approx(fa)
        assert status.success
        assert proj[mu_].variance == pytest.approx(COV[0, 0], abs=1e-8)
        assert proj[x_].variance == pytest.approx(COV[1, 1], abs=1e-8)
        assert proj[mu_].mean == pytest.approx(MODE[0], abs=1e-3)
        assert proj[x_].mean == pytest.approx(MODE[1], abs=1e-3)
        results.append(_projection_bits(proj, mu_, x_))
    # bit-equal across seeds: the default path draws no random numbers
    assert results[0] == results[1] == results[2]


def test__analytic_inverse_hessian_variable_id_independent():
    reference = None
    for n_throwaway in (1, 3, 6):
        _ = [Variable("k") for _ in range(n_throwaway)]
        fa, mu_, x_ = make_approx()
        np.random.seed(0)
        proj, status = LaplaceOptimiser().optimise_approx(fa)
        assert status.success
        bits = _projection_bits(proj, mu_, x_)
        if reference is None:
            reference = bits
        assert bits == reference


def test__finite_difference_hessian_matches_analytic():
    fa, mu_, x_ = make_approx()
    opt = LaplaceOptimiser()
    state = opt.prepare_state(fa, fa.model_dist)
    step = graph.VariableData({mu_: 0.1, x_: 0.2})
    H, shapes = newton.finite_difference_hessian(state, step)
    order = [shapes_v for shapes_v in shapes]
    assert order == sorted(order, key=lambda v: v.id)
    assert np.allclose(H, H.T)
    P_ordered = P if order[0] is mu_ else P[::-1, ::-1]
    assert np.allclose(-H, P_ordered, atol=1e-10)
    # the state passed in is untouched
    assert state.parameters is not None and state.hessian is not None


def test__line_search_failure_returns_last_dist_untouched():
    fa, mu_, x_ = make_approx()
    # stepsize None -> "Line search failed"
    opt = LaplaceOptimiser(calc_line_search=lambda state, old_state, **kw: (None, state))
    np.random.seed(0)
    proj, status = opt.optimise(fa)
    assert not status.success
    assert status.flag is StatusFlag.FAILURE
    # the mean field handed back is the one passed in, unchanged
    assert proj is fa.model_dist

    new_fa, st = fa.project(proj, status=status)
    assert new_fa.factor_dist is fa.factor_dist
    assert st.updated is False
    assert st.flag is StatusFlag.FAILURE
    assert any("skipped" in m for m in st.messages)


def test__line_search_failure_ep_mean_field_byte_identical():
    from autofit.graphical.expectation_propagation.optimiser import SimplerUpdater

    fa, mu_, x_ = make_approx()
    factor = fa.factor
    model_approx = graph.EPMeanField.from_approx_dists(
        graph.FactorGraph([factor]),
        {mu_: NormalMessage(50.0, 10.0), x_: NormalMessage(55.0, 20.0)},
    )
    factor_approx = model_approx.factor_approximation(factor)
    before = pickle.dumps(model_approx.factor_mean_field[factor])

    opt = LaplaceOptimiser(calc_line_search=lambda state, old_state, **kw: (None, state))
    proj, status = opt.optimise(factor_approx)
    assert status.flag is StatusFlag.FAILURE

    new_approx, st = SimplerUpdater().update_model_approx(
        proj, factor_approx, model_approx, status
    )
    assert st.updated is False
    assert pickle.dumps(new_approx.factor_mean_field[factor]) == before


def test__non_concave_mode_is_bad_projection():
    fa, mu_, x_ = make_approx(convex=True)
    # the optimiser would walk off the convex bowl: stop it at the start so the
    # Hessian is evaluated there (a successful, NO_CHANGE optimisation)
    opt = LaplaceOptimiser(stop_conditions=(lambda *a, **k: (True, "forced stop"),))
    proj, status = opt.optimise(fa)
    assert not status.success
    assert status.flag is StatusFlag.BAD_PROJECTION
    assert status.updated is False
    assert any("not concave" in m for m in status.messages)
    assert proj is fa.model_dist

    new_fa, st = fa.project(proj, status=status)
    assert new_fa.factor_dist is fa.factor_dist
    assert st.updated is False
    assert st.flag is StatusFlag.BAD_PROJECTION


def test__invalid_projection_reverts_variable_and_names_it():
    v, w = Variable("v"), Variable("w")
    last = graph.MeanField({v: NormalMessage(0.0, 1.0), w: NormalMessage(0.0, 1.0)})
    # v's projection is wider than its cavity (negative factor precision);
    # w's is a valid update
    projected = graph.MeanField({v: NormalMessage(0.0, 3.0), w: NormalMessage(1.0, 1.0)})
    cavity = graph.MeanField({v: NormalMessage(0.0, 2.0), w: NormalMessage(0.0, 2.0)})
    factor_dist, status = projected.update_factor_mean_field(cavity, last_dist=last)
    assert status.flag is StatusFlag.BAD_PROJECTION
    assert not status.success
    # per-variable backstop: v reverted to its previous message, w updated
    assert status.updated is True
    assert factor_dist[v].mean == last[v].mean and factor_dist[v].sigma == last[v].sigma
    assert factor_dist[w].sigma == pytest.approx(np.sqrt(1 / (1 - 1 / 4)))
    named = [m for m in status.messages if "for v " in m and "precision(q*)" in m]
    assert len(named) == 1
    assert "precision(cavity)=0.25" in named[0]
    assert not any("for w " in m for m in status.messages)


def test__n_refine_zero_is_honoured():
    fa, mu_, x_ = make_approx()
    opt = LaplaceOptimiser(hessian="quasi", n_refine=0)
    calls = []
    state = opt.prepare_state(fa, fa.model_dist)

    def new_param():
        calls.append(1)
        return fa.model_dist.sample()

    refined = opt.refine_state(state, new_param, n_refine=0)
    assert refined is state
    assert calls == []
    opt.refine_state(state, new_param)
    assert len(calls) == 0
    opt3 = LaplaceOptimiser(hessian="quasi", n_refine=3)
    opt3.refine_state(state, new_param)
    assert len(calls) == 3


def test__quasi_hessian_refinement_chains_secants():
    fa, mu_, x_ = make_approx()
    seen = []

    def spy(state1, state, **kwargs):
        seen.append(state.hessian)  # B_k the secant is applied to
        out = newton.full_diag_update(state1, state, **kwargs)
        seen.append(out.hessian)  # B_{k+1}
        return out

    opt = LaplaceOptimiser(hessian="quasi", n_refine=3, quasi_newton_update=spy)
    state = opt.prepare_state(fa, fa.model_dist)
    np.random.seed(0)
    refined = opt.refine_state(state, fa.model_dist.sample)
    assert len(seen) == 6
    # each refinement starts from the previous one's estimate, not the original
    assert seen[2] is seen[1]
    assert seen[4] is seen[3]
    assert refined.hessian is seen[5]


def test__hessian_option_validated():
    with pytest.raises(ValueError):
        LaplaceOptimiser(hessian="exact")
