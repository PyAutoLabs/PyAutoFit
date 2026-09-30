"""
``LaplaceOptimiser(projection="moments")`` projects a factor's tilted
distribution by matching moments computed with nested quadrature: an outer
Gauss-Legendre rule over the factor's scale (bounded-support) variables and an
inner conditional Laplace approximation over the remaining variables at each
outer node (PyAutoFit#1654).
"""
import logging

import numpy as np
import pytest
from scipy import integrate, stats

from autofit import graphical as graph
from autofit.graphical.laplace.optimiser import LaplaceOptimiser
from autofit.graphical.utils import StatusFlag
from autofit.mapper.variable import Variable
from autofit.messages import NormalMessage
from autofit.messages.truncated_normal import TruncatedNormalMessage

# Cavity of `make_scale_approx`: x ~ N(M, A), s ~ TruncatedNormal(C, sqrt(CV), 0, 100)
M, A = 1.0, 0.5
C, CV = 0.8, 0.36


def make_scale_approx(c=C, cv=CV):
    """
    A two-variable factor N(x | 0, s) with an analytic Jacobian: `s` is a
    scale with a truncated (bounded-support) cavity, so it is the outer
    quadrature variable and `x` the inner Laplace variable, whose conditional
    tilted density is exactly Gaussian.
    """
    x_, s_ = Variable("x"), Variable("s")

    def f(x, s):
        return -0.5 * (x / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)

    def f_jac(x, s):
        return f(x, s), (-x / s**2, x**2 / s**3 - 1 / s)

    factor = graph.Factor(f, x_, s_, factor_jacobian=f_jac)
    cavity = graph.MeanField(
        {
            x_: NormalMessage(M, np.sqrt(A)),
            s_: TruncatedNormalMessage(
                c, np.sqrt(cv), lower_limit=0.0, upper_limit=100.0
            ),
        }
    )
    return graph.FactorApproximation(factor, cavity, cavity, cavity), x_, s_


def exact_scale_moments():
    """
    The tilted moments of `make_scale_approx` by 1-D quadrature over s:
    p(s) ∝ TN(s | C, CV) N(M | 0, s² + A), E[x | s] = M s²/(s²+A),
    Var[x | s] = A s²/(s²+A).
    """
    a, b = (0.0 - C) / np.sqrt(CV), (100.0 - C) / np.sqrt(CV)
    prior = stats.truncnorm(a, b, loc=C, scale=np.sqrt(CV))

    def p(s):
        return prior.pdf(s) * stats.norm.pdf(M, 0.0, np.sqrt(s**2 + A))

    hi = C + 12 * np.sqrt(CV)
    kw = dict(epsabs=1e-14, epsrel=1e-12, limit=200)

    def integral(g):
        return integrate.quad(lambda s: g(s) * p(s), 0.0, hi, **kw)[0]

    z = integral(lambda s: 1.0)
    e_s = integral(lambda s: s) / z
    v_s = integral(lambda s: s**2) / z - e_s**2
    ex = lambda s: M * s**2 / (s**2 + A)
    vx = lambda s: A * s**2 / (s**2 + A)
    e_x = integral(ex) / z
    v_x = integral(lambda s: vx(s) + ex(s) ** 2) / z - e_x**2
    return dict(log_z=np.log(z), e_s=e_s, v_s=v_s, e_x=e_x, v_x=v_x)


def _bits(proj, *variables):
    return tuple(
        float(getattr(proj[v], attr))
        for v in variables
        for attr in ("mean", "variance")
    ) + (float(proj.log_norm),)


def test__known_moments_match_one_dimensional_quadrature():
    fa, x_, s_ = make_scale_approx()
    proj, status = LaplaceOptimiser(projection="moments").optimise(fa)
    assert status.success, status.messages
    assert status.flag is StatusFlag.SUCCESS
    assert any(m.startswith("moments: n_nodes=") for m in status.messages)

    exact = exact_scale_moments()
    assert proj[x_].mean == pytest.approx(exact["e_x"], abs=1e-6)
    assert proj[x_].variance == pytest.approx(exact["v_x"], abs=1e-6)
    # Truncated scale: the (E, Var)-as-parent convention of
    # TruncatedNormalMessage.invert_sufficient_statistics
    assert isinstance(proj[s_], TruncatedNormalMessage)
    assert proj[s_].mean == pytest.approx(exact["e_s"], abs=1e-6)
    assert proj[s_].variance == pytest.approx(exact["v_s"], abs=1e-6)
    assert (proj[s_].lower_limit, proj[s_].upper_limit) == (0.0, 100.0)
    # the tilted normalisation Z = ∫ f q_cavity
    assert proj.log_norm == pytest.approx(exact["log_z"], abs=1e-6)


def make_gaussian_approx():
    """
    `test_laplace_hessian.make_approx`, with `x`'s cavity a truncated normal
    whose limits sit hundreds of sigma out: the tilted density is Gaussian to
    roundoff, but `x` has a bounded support and so takes the moments path.
    """
    mu_, x_ = Variable("mu"), Variable("x")
    sigma_f = 10.0

    def f(mu, x):
        return -0.5 * ((x - mu) / sigma_f) ** 2

    def f_jac(mu, x):
        d = (x - mu) / sigma_f**2
        return f(mu, x), (d, -d)

    factor = graph.Factor(f, mu_, x_, factor_jacobian=f_jac)
    cavity = graph.MeanField(
        {
            mu_: NormalMessage(50.0, 10.0),
            x_: TruncatedNormalMessage(
                55.0, 20.0, lower_limit=-1e4, upper_limit=1e4
            ),
        }
    )
    return graph.FactorApproximation(factor, cavity, cavity, cavity), mu_, x_


def test__gaussian_tilted_moments_match_mode_and_covariance():
    P = np.array([[0.02, -0.01], [-0.01, 0.0125]])
    cov = np.linalg.inv(P)
    mode = np.linalg.solve(P, np.diag([0.01, 0.0025]) @ np.array([50.0, 55.0]))

    fa, mu_, x_ = make_gaussian_approx()
    proj, status = LaplaceOptimiser(projection="moments").optimise(fa)
    assert status.flag is StatusFlag.SUCCESS, status.messages
    assert proj[mu_].mean == pytest.approx(mode[0], abs=1e-6)
    assert proj[x_].mean == pytest.approx(mode[1], abs=1e-6)
    assert proj[mu_].variance == pytest.approx(cov[0, 0], abs=1e-6)
    assert proj[x_].variance == pytest.approx(cov[1, 1], abs=1e-6)


def test__narrow_tilted_density_is_rewindowed():
    """
    A tilted density ~14x narrower than the cavity window (x: cavity std 20,
    tilted std 1.4) is under-resolved by the first Gauss-Legendre pass; the
    rule is re-windowed onto the tilted density until it is resolved.
    """
    mu_, x_ = Variable("mu"), Variable("x")

    def f(mu, x):
        return -0.5 * (x - mu) ** 2

    def f_jac(mu, x):
        return f(mu, x), (x - mu, mu - x)

    factor = graph.Factor(f, mu_, x_, factor_jacobian=f_jac)
    cavity = graph.MeanField(
        {
            mu_: NormalMessage(50.0, 1.0),
            x_: TruncatedNormalMessage(55.0, 20.0, lower_limit=-1e4, upper_limit=1e4),
        }
    )
    fa = graph.FactorApproximation(factor, cavity, cavity, cavity)
    P = np.array([[2.0, -1.0], [-1.0, 1.0 + 1 / 400]])
    cov = np.linalg.inv(P)
    mode = np.linalg.solve(P, np.array([50.0, 55.0 / 400]))

    proj, status = LaplaceOptimiser(projection="moments").optimise(fa)
    assert status.flag is StatusFlag.SUCCESS, status.messages
    assert "passes=1," not in status.messages[0]
    assert proj[mu_].mean == pytest.approx(mode[0], abs=1e-6)
    assert proj[x_].mean == pytest.approx(mode[1], abs=1e-6)
    assert proj[mu_].variance == pytest.approx(cov[0, 0], abs=1e-6)
    assert proj[x_].variance == pytest.approx(cov[1, 1], abs=1e-6)


def test__moments_rng_independent():
    results = []
    for seed in (0, 1, 12345):
        fa, x_, s_ = make_scale_approx()
        np.random.seed(seed)
        proj, status = LaplaceOptimiser(projection="moments").optimise(fa)
        assert status.flag is StatusFlag.SUCCESS
        results.append(_bits(proj, x_, s_))
    assert results[0] == results[1] == results[2]


def test__moments_variable_id_independent():
    reference = None
    for n_throwaway in (1, 3, 6):
        _ = [Variable("k") for _ in range(n_throwaway)]
        fa, x_, s_ = make_scale_approx()
        np.random.seed(0)
        proj, status = LaplaceOptimiser(projection="moments").optimise(fa)
        assert status.flag is StatusFlag.SUCCESS
        bits = _bits(proj, x_, s_)
        if reference is None:
            reference = bits
        assert bits == reference


def _laplace_hessian_make_approx():
    from test_autofit.graphical.functionality.test_laplace_hessian import make_approx

    return make_approx()


def test__no_outer_variable_falls_back_to_mode_bit_for_bit():
    fa, mu_, x_ = _laplace_hessian_make_approx()
    np.random.seed(0)
    mode_proj, mode_status = LaplaceOptimiser().optimise(fa)
    mode_bits = _bits(mode_proj, mu_, x_)

    fa, mu_, x_ = _laplace_hessian_make_approx()
    np.random.seed(0)
    proj, status = LaplaceOptimiser(projection="moments").optimise(fa)

    assert status.flag is mode_status.flag is StatusFlag.SUCCESS
    assert status.messages == mode_status.messages
    assert _bits(proj, mu_, x_) == mode_bits


def test__over_moment_max_size_falls_back_and_logs(caplog):
    fa, x_, s_ = make_scale_approx()
    np.random.seed(0)
    mode_proj, mode_status = LaplaceOptimiser().optimise(fa)
    mode_bits = _bits(mode_proj, x_, s_) if mode_status.success else None

    fa, x_, s_ = make_scale_approx()
    np.random.seed(0)
    with caplog.at_level(logging.INFO, logger="autofit.graphical.laplace"):
        proj, status = LaplaceOptimiser(
            projection="moments", moment_max_size=1
        ).optimise(fa)

    assert any("moment_max_size" in r.getMessage() for r in caplog.records)
    assert status.flag is mode_status.flag
    assert status.messages == mode_status.messages
    if status.success:
        assert _bits(proj, x_, s_) == mode_bits
    else:
        assert proj is fa.model_dist


def test__too_many_outer_variables_falls_back():
    fa, x_, s_ = make_scale_approx()
    np.random.seed(0)
    _, mode_status = LaplaceOptimiser().optimise(fa)
    fa, x_, s_ = make_scale_approx()
    np.random.seed(0)
    _, status = LaplaceOptimiser(projection="moments", moment_max_outer=0).optimise(
        fa
    )
    assert status.messages == mode_status.messages


def test__bad_projection_value_raises():
    with pytest.raises(ValueError, match="projection"):
        LaplaceOptimiser(projection="median")


def test__cavity_outside_support_is_bad_projection():
    # The cavity window C ± 8 sd lies wholly below the scale's lower limit 0
    fa, x_, s_ = make_scale_approx(c=-100.0, cv=1.0)
    proj, status = LaplaceOptimiser(projection="moments").optimise(fa)
    assert status.flag is StatusFlag.BAD_PROJECTION
    assert not status.success
    assert status.updated is False
    assert proj is fa.model_dist
    assert any("window" in m for m in status.messages)


# --------------------------------------------------------------------------
# End to end: the hierarchical toy of hierarchical/test_truncated_support.py
# --------------------------------------------------------------------------


def make_hierarchical_toy(truths, noise=2.0):
    """
    Three groups with a `TruncatedGaussianPrior` scatter, as
    `test_truncated_support.py`. Returns the scatter/mean priors, the graph and
    each group's data.
    """
    import autofit as af
    from test_autofit.graphical.hierarchical.test_truncated_support import (
        GroupAnalysis,
        Level,
    )

    rng = np.random.default_rng(0)
    hf = af.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=50.0, sigma=10.0),
        sigma=af.TruncatedGaussianPrior(
            mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0
        ),
    )
    analysis_factors, ys = [], []
    for truth in truths:
        model = af.Model(Level)
        model.x = af.GaussianPrior(50.0, 20.0)
        y = truth + noise * rng.standard_normal(20)
        ys.append(y)
        analysis_factors.append(af.AnalysisFactor(model, GroupAnalysis(y, noise)))
        hf.add_drawn_variable(model.x)
    return hf, af.FactorGraphModel(*analysis_factors, hf), ys


def exact_scatter_posterior(ys, noise=2.0):
    """
    Mean and std of the toy's exact σ posterior: given σ the model is Gaussian
    in (μ, x₁..x₃), so p(σ | y) ∝ TN(σ | 10, 5; 0, 100) · p(y | σ) with the
    marginal likelihood in closed form, then 1-D quadrature over σ.
    """
    n = len(ys)

    def log_marginal(s):
        L = np.zeros((n + 1, n + 1))
        h = np.zeros(n + 1)
        L[0, 0] += 1 / 10.0**2
        h[0] += 50.0 / 10.0**2
        for i, y in enumerate(ys):
            L[i + 1, i + 1] += 1 / 20.0**2 + len(y) / noise**2 + 1 / s**2
            h[i + 1] += 50.0 / 20.0**2 + y.sum() / noise**2
            L[0, 0] += 1 / s**2
            L[0, i + 1] = L[i + 1, 0] = L[0, i + 1] - 1 / s**2
        return (
            0.5 * h @ np.linalg.solve(L, h)
            - 0.5 * np.linalg.slogdet(L)[1]
            - n * np.log(s)
        )

    s = np.linspace(1e-4, 60.0, 30001)
    lp = np.array([log_marginal(v) for v in s]) + stats.norm.logpdf(s, 10.0, 5.0)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean = float(np.sum(w * s))
    return mean, float(np.sqrt(np.sum(w * (s - mean) ** 2)))


def run_manual_ep(graph_model, optimiser, n_sweeps):
    """
    The manual factor_approximation -> optimise -> project_mean_field loop of
    `test_truncated_support.py`. Returns the final mean field and, per
    hierarchical-factor update, the flag after projection.
    """
    approx = graph_model.mean_field_approximation()
    flags = []
    for _ in range(n_sweeps):
        for factor in graph_model.graph.factors:
            factor_approx = approx.factor_approximation(factor)
            new_dist, status = optimiser.optimise(factor_approx)
            approx, status = approx.project_mean_field(
                new_dist, factor_approx, status=status
            )
            if hasattr(factor, "scale_variables"):
                flags.append(status.flag)
    return approx.mean_field, flags


def test__hierarchical_scale_variables():
    hf, graph_model, _ = make_hierarchical_toy((45.0, 52.0, 58.0))
    hierarchical = [f for f in graph_model.graph.factors if hasattr(f, "scale_variables")]
    assert len(hierarchical) == 3
    for factor in hierarchical:
        assert factor.scale_variables == frozenset({hf.sigma})


def test__hierarchical_scatter_moments_match_exact_posterior():
    np.random.seed(0)
    hf, graph_model, ys = make_hierarchical_toy((45.0, 52.0, 58.0))
    mean_field, flags = run_manual_ep(
        graph_model, LaplaceOptimiser(projection="moments"), n_sweeps=2
    )
    assert flags.count(StatusFlag.SUCCESS) >= 1, flags
    assert StatusFlag.EXCEPTION not in flags

    scatter = mean_field[hf.sigma]
    assert isinstance(scatter, TruncatedNormalMessage)
    assert (scatter.lower_limit, scatter.upper_limit) == (0.0, 100.0)

    exact_mean, exact_std = exact_scatter_posterior(ys)
    assert abs(scatter.mean - exact_mean) < 0.5 * exact_std, (
        scatter.mean,
        exact_mean,
        exact_std,
    )


def test__near_zero_scatter_updates_by_moments_not_by_mode():
    """
    Groups consistent with no scatter: the tilted density in σ piles up on
    σ = 0, so the mode path never lands an update of a hierarchical factor
    (each projection is rejected) while the moments path does.
    """
    truths = (50.0, 50.5, 49.5)

    np.random.seed(0)
    hf, graph_model, _ = make_hierarchical_toy(truths)
    mode_mean_field, mode_flags = run_manual_ep(
        graph_model, LaplaceOptimiser(), n_sweeps=2
    )
    assert StatusFlag.SUCCESS not in mode_flags, mode_flags

    np.random.seed(0)
    hf, graph_model, _ = make_hierarchical_toy(truths)
    mean_field, flags = run_manual_ep(
        graph_model, LaplaceOptimiser(projection="moments"), n_sweeps=1
    )
    assert flags == [StatusFlag.SUCCESS] * 3, flags
    # the scatter has moved off its prior (10 ± 5) towards 0
    assert mean_field[hf.sigma].mean < 9.0
    assert (mean_field[hf.sigma].lower_limit, mean_field[hf.sigma].upper_limit) == (
        0.0,
        100.0,
    )
