"""
Moment-matching projection of a factor's tilted distribution by nested
quadrature (PyAutoFit#1654).

The Laplace ("mode") projection of `LaplaceOptimiser` fails on the factor it is
most needed for: a ``HierarchicalFactor`` whose parent scale σ is poorly
constrained. Its tilted density in σ piles up against σ = 0, so there is no
interior mode and no negative-definite Hessian (``BAD_PROJECTION``), or the
line search walks into the boundary (``FAILURE``). EP's projection is, by
definition, a *moment* match — the Gaussian closest in KL(p̃ ‖ q) to the tilted
density p̃ — and the mode/curvature pair is only its large-data approximation.

This module computes those moments directly, INLA-style:

- **Outer** variables — the factor's scale variables (``scale_variables`` of a
  ``_HierarchicalFactor``) and any variable whose message has a bounded support
  (e.g. a ``TruncatedGaussianPrior`` σ) — are integrated with a tensor
  Gauss–Legendre rule of ``n_quadrature`` nodes per variable, in the message's
  base coordinate u (identity for Normal/TruncatedNormal messages, log σ for a
  ``LogGaussianPrior``), on the intersection of the variable's support with the
  cavity window ``mean ± half_width · std``. Further passes (at most
  ``MAX_PASSES``) re-window the rule onto the tilted
  ``mean ± half_width · max(std, node spacing)`` while a pass finds the tilted
  density narrower than a quarter of its window's scale (under-resolved) or
  finds mass at a window edge that is not a support edge.
- **Inner** variables — everything else — are integrated at each outer node
  s_j by a conditional Laplace approximation: the conditional mode m_j
  (quasi-Newton, warm-started from the previous node, polished by Newton steps
  on the finite-difference Hessian H_j) and covariance Σ_j = (−H_j)⁻¹. For a
  hierarchical Gaussian model the conditional density is exactly Gaussian, so
  this step is exact.

Each outer node carries the log weight

    log w_j = log w_GL,j + log |dx/du|_j + ℓ(m_j, s_j) + (d/2) log 2π − ½ log det(−H_j)

where ℓ is the tilted log-density (factor plus cavity) and d the number of inner
parameters, so that ``logsumexp(log w)`` is the tilted normalisation
Ẑ = ∫ f q_cavity. Moments are then handed to each message's own ``project``
(`MeanField.from_weighted_nodes`) with the outer nodes at s_j and the inner
nodes an order-3 Gauss–Hermite expansion of each conditional N(m_j, Σ_j) in
whitened coordinates (3^d points, exact for E[x] and E[x xᵀ]), which reproduces
the law of total variance.

No random numbers are drawn: the projection is bit-for-bit deterministic.
"""
import itertools
import logging
import math
from operator import attrgetter
from typing import Dict, List, Optional, Tuple

import numpy as np

from autofit import exc
from autofit.graphical.laplace import newton
from autofit.graphical.mean_field import MeanField
from autofit.graphical.utils import FlattenArrays, Status, StatusFlag
from autofit.mapper.variable import Variable
from autofit.mapper.variable_operator import VariableData
from autofit.messages.composed_transform import TransformedMessage

logger = logging.getLogger(__name__)

#: The parameter support of a distribution's scale argument (``sigma`` of
#: ``NormalMessage._parameter_support``): a scale is positive.
SCALE_SUPPORT = (0.0, math.inf)

#: A node whose share of the tilted mass is above this is "weighted": a failed
#: inner search or a non-concave inner Hessian there invalidates the projection.
WEIGHTED_NODE_MASS = 1e-10

#: Mass on the outermost node of a window edge that is not a support edge,
#: above which the window is judged to have cut off tilted mass.
EDGE_MASS = 1e-6

#: A pass is re-windowed when the tilted std in base space is below this
#: fraction of the scale its window was built from (the cavity std on the
#: first pass): the tilted density is under-resolved by the rule.
NARROW_FRACTION = 0.25

#: At most this many quadrature passes (each re-window shrinks the node
#: spacing ~2.5x); still unresolved after the last is a BAD_PROJECTION.
MAX_PASSES = 4

#: Newton polish of the conditional mode stops once the predicted increase of
#: the log-density, ½ gᵀ(−H)⁻¹g, is below this (nats).
NEWTON_DECREMENT_TOL = 1e-12
MAX_NEWTON_POLISH = 5

# Order-3 (probabilists') Gauss–Hermite rule: exact to the fifth moment of N(0, 1)
_GH_NODES = np.array([-math.sqrt(3.0), 0.0, math.sqrt(3.0)])
_GH_LOG_WEIGHTS = np.log(np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]))


def _logsumexp(a) -> float:
    # numpy only: `autofit` must not import scipy.special at import time
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(a - m))))


def _cho_solve(L, b):
    """Solve (L Lᵀ) x = b for a lower-triangular Cholesky factor L."""
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def _scale_variables(factor_approx) -> frozenset:
    factor = getattr(factor_approx, "factor", factor_approx)
    try:
        return frozenset(getattr(factor, "scale_variables", ()) or ())
    except Exception:  # a factor that cannot resolve its priors is not a scale factor
        return frozenset()


def _has_bounded_support(message) -> bool:
    kw = message._support_kwargs
    return bool(kw) and (
        np.isfinite(kw.get("lower_limit", -math.inf))
        or np.isfinite(kw.get("upper_limit", math.inf))
    )


def split_variables(
    factor_approx, mean_field
) -> Tuple[List[Variable], List[Variable], frozenset]:
    """
    Split a factor's free variables into outer (scale / bounded-support) and
    inner variables, each sorted by ``Variable.id`` so that the node order does
    not depend on dict order.
    """
    scales = _scale_variables(factor_approx)
    free = sorted(factor_approx.free_variables, key=attrgetter("id"))
    outer = [v for v in free if v in scales or _has_bounded_support(mean_field[v])]
    inner = [v for v in free if v not in outer]
    return outer, inner, scales


def fallback_reason(
    factor_approx, mean_field, max_size: int, max_outer: int
) -> Optional[str]:
    """
    Why the moments path does not apply to this factor (so the mode path runs
    unchanged), or ``None`` when it does.
    """
    if not (
        hasattr(factor_approx, "cavity_dist")
        and hasattr(factor_approx, "func_gradient")
    ):
        return "not a FactorApproximation"
    if factor_approx.deterministic_variables:
        return "factor has deterministic variables"

    outer, inner, _ = split_variables(factor_approx, mean_field)
    if not outer:
        return "no scale or bounded-support variable"
    if len(outer) > max_outer:
        return f"{len(outer)} outer variables exceed moment_max_outer={max_outer}"
    for v in outer:
        if np.shape(mean_field[v].mean) != ():
            return f"outer variable {v.name} is not a scalar"
    n_params = sum(np.size(mean_field[v].mean) for v in outer + inner)
    if n_params > max_size:
        return f"{n_params} free parameters exceed moment_max_size={max_size}"
    return None


def gauss_legendre(n: int, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre nodes and log weights on [lo, hi]."""
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (hi - lo)
    return lo + half * (x + 1.0), np.log(half * w)


class _OuterAxis:
    def __init__(self, variable: Variable, message, cavity, is_scale: bool):
        """
        One outer quadrature axis, in the message's base coordinate u.

        ``lo``/``hi`` are the support in u: the message's own support (the
        truncation limits of a TruncatedNormal, in base space for a
        TransformedMessage) intersected, for a scale variable, with the
        positive half-line mapped through the transform.
        """
        self.variable = variable
        self.message = message
        self.transformed = isinstance(message, TransformedMessage)

        lo, hi = -math.inf, math.inf
        kw = message._support_kwargs
        lo = max(lo, float(kw.get("lower_limit", -math.inf)))
        hi = min(hi, float(kw.get("upper_limit", math.inf)))
        if is_scale:
            ends = np.array(SCALE_SUPPORT, dtype=float)
            if self.transformed:
                with np.errstate(all="ignore"):
                    ends = np.asarray(message._transform(ends), dtype=float)
                ends = np.nan_to_num(ends, nan=-math.inf)
            lo = max(lo, float(np.min(ends)))
            hi = min(hi, float(np.max(ends)))
        self.lo, self.hi = lo, hi

        self.mean = self.std = math.nan
        if cavity is not None and hasattr(cavity, "variance"):
            base = cavity.base_message if isinstance(cavity, TransformedMessage) else cavity
            try:
                mean = float(base.mean)
                variance = float(base.variance)
            except (TypeError, ValueError, AttributeError):
                mean = variance = math.nan
            if np.isfinite(mean) and np.isfinite(variance) and variance > 0:
                self.mean, self.std = mean, math.sqrt(variance)

    def window(self, centre: float, scale: float, half_width: float):
        """
        ``centre ± half_width·scale`` clipped to the support; an edge that lies
        on a (finite) support limit is lifted inside it by
        ``1e-12·max(1, |centre|)`` — a scale of exactly 0 has zero measure and
        N(x | μ, 0) is undefined. Returns ``(lo, hi, lo_is_support,
        hi_is_support)`` or ``None`` when the window is empty.
        """
        if not (np.isfinite(centre) and np.isfinite(scale) and scale > 0):
            if np.isfinite(self.lo) and np.isfinite(self.hi):
                g_lo, g_hi = self.lo, self.hi
            else:
                return None
        else:
            g_lo = max(centre - half_width * scale, self.lo)
            g_hi = min(centre + half_width * scale, self.hi)
        lift = 1e-12 * max(1.0, abs(centre) if np.isfinite(centre) else 1.0)
        lo_is_support = g_lo == self.lo
        hi_is_support = g_hi == self.hi
        if lo_is_support:
            g_lo = g_lo + lift
        if hi_is_support:
            g_hi = g_hi - lift
        if not g_hi > g_lo:
            return None
        return g_lo, g_hi, lo_is_support, hi_is_support

    def to_physical(self, u: float) -> Tuple[float, float]:
        """The physical value at base coordinate u and log |dx/du|."""
        if not self.transformed:
            return u, 0.0
        x = float(self.message._inverse_transform(np.asarray(u, dtype=float)))
        _, log_du_dx = self.message._transform_det(np.asarray(x, dtype=float))
        return x, -float(log_du_dx)


class _Conditional:
    def __init__(self, factor_approx, outer_values: Dict[Variable, float]):
        """The tilted log-density with the outer variables held fixed."""
        self.factor_approx = factor_approx
        self.outer_values = outer_values
        self.f_count = 0
        self.g_count = 0

    def __call__(self, parameters):
        self.f_count += 1
        return self.factor_approx({**parameters, **self.outer_values})

    def gradient(self, parameters):
        self.g_count += 1
        value, gradient = self.factor_approx.func_gradient(
            {**parameters, **self.outer_values}
        )
        return value, VariableData({v: gradient[v] for v in parameters})


class _NodeResult:
    __slots__ = ("ok", "kind", "log_weight", "mode", "chol_cov", "reason")

    def __init__(self, ok, log_weight, mode=None, chol_cov=None, kind="", reason=""):
        self.ok = ok
        self.kind = kind  # "search" or "hessian" on failure
        self.log_weight = log_weight
        self.mode = mode
        self.chol_cov = chol_cov
        self.reason = reason


class MomentProjection:
    def __init__(self, optimiser, factor_approx, mean_field, params=None, **kwargs):
        """
        One moment-matching projection of ``factor_approx``'s tilted density
        with the settings of ``optimiser`` (a ``LaplaceOptimiser``).
        """
        self.optimiser = optimiser
        self.factor_approx = factor_approx
        self.mean_field = mean_field
        self.kwargs = kwargs

        self.outer, self.inner, scales = split_variables(factor_approx, mean_field)
        cavity = factor_approx.cavity_dist
        self.axes = [
            _OuterAxis(v, mean_field[v], cavity.get(v), v in scales)
            for v in self.outer
        ]

        parameters = MeanField.mean.fget(mean_field)
        if params:
            for v, p in params.items():
                parameters[v] = p
        self.start = VariableData({v: parameters[v] for v in self.inner})
        self.shapes = FlattenArrays({v: np.shape(self.start[v]) for v in self.inner})
        self.n_inner = int(sum(np.size(self.start[v]) for v in self.inner))

        if self.inner:
            self.hessian0 = optimiser.make_hessian(
                mean_field, self.inner, **optimiser.hessian_kws
            )
            if optimiser.check_limits:
                lower = MeanField.lower_limit.fget(mean_field)
                upper = MeanField.upper_limit.fget(mean_field)
                self.limits = dict(
                    lower_limit=VariableData({v: lower[v] for v in self.inner}),
                    upper_limit=VariableData({v: upper[v] for v in self.inner}),
                )
            else:
                self.limits = {}

        self.n_passes = 0
        self.n_nodes = 0
        self.n_polished = 0
        self.f_count = 0
        self.g_count = 0

    @property
    def factor_name(self):
        factor = getattr(self.factor_approx, "factor", self.factor_approx)
        return getattr(factor, "name", factor)

    # ----------------------------------------------------------------- nodes

    def _solve_node(self, outer_values, log_w_outer, start) -> _NodeResult:
        conditional = _Conditional(self.factor_approx, outer_values)
        if not self.inner:
            value = float(conditional({}))
            self.f_count += conditional.f_count
            return _NodeResult(True, log_w_outer + value)

        state = newton.OptimisationState(
            conditional,
            conditional.gradient,
            start,
            self.hessian0,
            None,
            **self.limits,
        )
        with np.errstate(all="ignore"):
            next_state, status = self.optimiser.optimise_state(state, **self.kwargs)
        best = max(state, next_state, key=lambda s: s.value)

        # Newton polish on central differences of the conditional tilted
        # log-density *values*. The factor gradients are forward differences
        # (eps=1e-8) of a log-density whose exponential-family terms cancel
        # catastrophically at a small scale node (η₁x, η₂x², A(η) ~ 1e6-1e7 at
        # σ ~ 0.02): their ~1e-1 noise swamps the soft direction of the
        # conditional (curvature ~1e-2) and made `finite_difference_hessian`
        # indefinite there. Value differences on the `fd_steps` scale carry
        # ~1e-9 noise and are exact for the (Gaussian) conditional of a
        # hierarchical Gaussian model.
        h = self.shapes.flatten(self.optimiser.fd_steps(best, self.mean_field))
        x = self.shapes.flatten(best.parameters)
        f0 = float(best.value)
        converged = False
        chol = None
        reason = ""
        for _ in range(MAX_NEWTON_POLISH + 1):
            with np.errstate(all="ignore"):
                f0, g, H = self._value_derivatives(conditional, x, f0, h)
            if not (np.isfinite(f0) and np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
                chol, reason = None, "non-finite inner Hessian"
                break
            try:
                chol = np.linalg.cholesky(-H)
            except np.linalg.LinAlgError:
                min_eig = np.linalg.eigvalsh(-H).min()
                chol = None
                reason = f"inner tilted log-density not concave (min eig {min_eig:.3g})"
                break
            step = _cho_solve(chol, g)
            decrement = 0.5 * float(g @ step)
            if decrement < NEWTON_DECREMENT_TOL:
                converged = True
                break
            for _halving in range(20):
                f1 = float(conditional(self._unflatten(x + step)))
                if f1 > f0:
                    break
                step = 0.5 * step
            else:
                # No further increase is available at this resolution
                converged = decrement < 1e-6
                if not converged:
                    reason = "Newton polish of the conditional mode stalled"
                break
            x, f0 = x + step, f1
            self.n_polished += 1
        else:
            reason = "Newton polish did not converge"

        self.f_count += conditional.f_count
        self.g_count += conditional.g_count
        proxy = log_w_outer + f0
        if chol is None:
            return _NodeResult(False, proxy, kind="hessian", reason=reason)
        if not converged:
            return _NodeResult(
                False,
                proxy,
                kind="search",
                reason=f"{reason}; {status.messages[-1] if status.messages else ''}",
            )

        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        covariance = _cho_solve(chol, np.eye(len(chol)))
        chol_cov = np.linalg.cholesky(0.5 * (covariance + covariance.T))
        log_weight = (
            proxy + 0.5 * self.n_inner * math.log(2 * math.pi) - 0.5 * log_det
        )
        return _NodeResult(
            True,
            log_weight,
            mode=x,
            chol_cov=chol_cov,
        )

    def _unflatten(self, x):
        return VariableData(self.shapes.unflatten(x))

    def _value_derivatives(self, conditional, x, f0, h):
        """
        The value, gradient and Hessian of the conditional tilted log-density
        at flat parameters ``x`` by central differences of its values with
        per-parameter steps ``h``: 1 + 2d + 2d(d-1) evaluations.
        """
        d = x.size
        f = lambda y: float(conditional(self._unflatten(y)))
        E = np.diag(h)
        fp = np.array([f(x + E[i]) for i in range(d)])
        fm = np.array([f(x - E[i]) for i in range(d)])
        g = (fp - fm) / (2 * h)
        H = np.empty((d, d))
        H[np.diag_indices(d)] = (fp - 2 * f0 + fm) / h**2
        for i in range(d):
            for j in range(i + 1, d):
                fpp = f(x + E[i] + E[j])
                fpm = f(x + E[i] - E[j])
                fmp = f(x - E[i] + E[j])
                fmm = f(x - E[i] - E[j])
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
        return f0, g, H

    def _integrate(self, windows):
        """One tensor-product pass over the outer windows."""
        n = self.optimiser.n_quadrature
        rules = [gauss_legendre(n, lo, hi) for lo, hi, _, _ in windows]
        grid = []
        for index in itertools.product(range(n), repeat=len(rules)):
            u = tuple(rules[k][0][i] for k, i in enumerate(index))
            log_w = sum(rules[k][1][i] for k, i in enumerate(index))
            grid.append((index, u, log_w))

        results = []
        start = self.start
        for index, u, log_w in grid:
            outer_values = {}
            log_j = 0.0
            for axis, u_k in zip(self.axes, u):
                x, lj = axis.to_physical(u_k)
                outer_values[axis.variable] = x
                log_j += lj
            result = self._solve_node(outer_values, log_w + log_j, start)
            if result.ok and result.mode is not None:
                start = VariableData(self.shapes.unflatten(result.mode))
            results.append((index, u, outer_values, result))

        self.n_passes += 1
        return results

    # ------------------------------------------------------------- statistics

    def _check_nodes(self, results) -> Tuple[Optional[Status], Optional[float]]:
        good = [r.log_weight for *_, r in results if r.ok]
        failed = [r for *_, r in results if not r.ok]
        log_z = _logsumexp(good) if good else -math.inf
        if not np.isfinite(log_z):
            if any(r.kind == "search" for r in failed if np.isfinite(r.log_weight)):
                return self._status(StatusFlag.FAILURE, "inner search failed at every weighted node"), None
            return self._status(StatusFlag.BAD_PROJECTION, "zero tilted mass in the window"), None

        threshold = log_z + math.log(WEIGHTED_NODE_MASS)
        for r in failed:
            if np.isfinite(r.log_weight) and r.log_weight > threshold:
                flag = StatusFlag.FAILURE if r.kind == "search" else StatusFlag.BAD_PROJECTION
                return self._status(flag, f"at a weighted node: {r.reason}"), None
        return None, log_z

    def _axis_statistics(self, results, log_z, windows):
        """
        Tilted mean/std in base space, edge-mass flag and node spacing at the
        peak, per outer axis.
        """
        n = self.optimiser.n_quadrature
        stats = []
        for k, axis in enumerate(self.axes):
            marginal = np.zeros(n)
            u_k = np.zeros(n)
            for index, u, _, r in results:
                u_k[index[k]] = u[k]
                if r.ok:
                    marginal[index[k]] += math.exp(r.log_weight - log_z)
            mean = float(np.sum(marginal * u_k))
            std = math.sqrt(max(float(np.sum(marginal * (u_k - mean) ** 2)), 0.0))
            _, _, lo_support, hi_support = windows[k]
            edge = (not lo_support and marginal[0] > EDGE_MASS) or (
                not hi_support and marginal[-1] > EDGE_MASS
            )
            # The rule's node spacing at the peak: a floor for the re-window
            # scale when the first pass under-resolves a narrow tilted density
            peak = int(np.argmax(marginal))
            gaps = np.diff(u_k)
            spacing = float(np.max(gaps[max(peak - 1, 0) : peak + 1]))
            stats.append((mean, std, edge, spacing))
        return stats

    def _status(self, flag, reason, success=False):
        return Status(
            success=success,
            messages=(f"moment projection for {self.factor_name}: {reason}",),
            updated=False,
            flag=flag,
        )

    # ------------------------------------------------------------------- run

    def __call__(self):
        half_width = self.optimiser.quadrature_half_width
        windows, scales = [], []
        for axis in self.axes:
            window = axis.window(axis.mean, axis.std, half_width)
            if window is None:
                return self.mean_field, self._status(
                    StatusFlag.BAD_PROJECTION,
                    f"cavity window of {axis.variable.name} lies outside its support",
                )
            windows.append(window)
            scales.append(
                axis.std
                if np.isfinite(axis.std)
                else (window[1] - window[0]) / (2 * half_width)
            )

        # Pass 1 on the cavity window; re-window onto the tilted
        # mean ± half_width·max(std, node spacing) while the tilted density is
        # under-resolved (std < NARROW_FRACTION of the window's scale) or has
        # mass on a window edge that is not a support edge.
        for n_pass in range(MAX_PASSES):
            results = self._integrate(windows)
            status, log_z = self._check_nodes(results)
            if status is not None:
                return self.mean_field, status

            stats = self._axis_statistics(results, log_z, windows)
            unresolved = [
                edge or not std >= NARROW_FRACTION * scale
                for (mean, std, edge, _), scale in zip(stats, scales)
            ]
            if not any(unresolved):
                break
            if n_pass == MAX_PASSES - 1:
                names = ", ".join(
                    axis.variable.name
                    for axis, bad in zip(self.axes, unresolved)
                    if bad
                )
                return self.mean_field, self._status(
                    StatusFlag.BAD_PROJECTION,
                    f"residual edge mass / unresolved tilted density for {names} "
                    f"after {MAX_PASSES} passes",
                )

            windows, scales = [], []
            for axis, (mean, std, _, spacing) in zip(self.axes, stats):
                scale = max(std, spacing)
                window = axis.window(mean, scale, half_width) if scale > 0 else None
                if window is None:
                    return self.mean_field, self._status(
                        StatusFlag.BAD_PROJECTION,
                        f"empty tilted window for {axis.variable.name}",
                    )
                windows.append(window)
                scales.append(scale)

        return self._project(results, log_z)

    def _project(self, results, log_z):
        nodes, log_weights = self._expand_nodes(results)
        self.n_nodes = len(log_weights)

        for v, values in nodes.items():
            message = self.mean_field[v]
            if isinstance(message, TransformedMessage):
                with np.errstate(all="ignore"):
                    base = message._transform(values)
                if not np.all(np.isfinite(base)):
                    return self.mean_field, self._status(
                        StatusFlag.BAD_PROJECTION,
                        f"quadrature nodes of {v.name} fall outside its support",
                    )

        try:
            with np.errstate(all="ignore"):
                projection = MeanField.from_weighted_nodes(
                    self.mean_field, nodes, log_weights, log_norm=log_z
                )
        except (AssertionError, ValueError, ArithmeticError, exc.MessageException) as e:
            return self.mean_field, self._status(
                StatusFlag.BAD_PROJECTION, f"moment inversion failed: {e}"
            )

        for v in nodes:
            message = projection[v]
            mean = np.asarray(message.mean, dtype=float)
            variance = np.asarray(message.variance, dtype=float)
            if not (
                np.all(np.isfinite(mean))
                and np.all(np.isfinite(variance))
                and np.all(variance > 0)
            ):
                return self.mean_field, self._status(
                    StatusFlag.BAD_PROJECTION, f"non-finite moment for {v.name}"
                )
        if not np.isfinite(projection.log_norm):
            return self.mean_field, self._status(
                StatusFlag.BAD_PROJECTION, "non-finite tilted normalisation"
            )

        return projection, Status(
            success=True,
            messages=(
                f"moments: n_nodes={self.n_nodes}, passes={self.n_passes}, "
                f"f_count={self.f_count}, g_count={self.g_count}",
            ),
            updated=True,
            flag=StatusFlag.SUCCESS,
        )

    def _expand_nodes(self, results):
        """
        Outer nodes at s_j; each inner conditional N(m_j, Σ_j) expanded into the
        3^d order-3 Gauss–Hermite points m_j + L_j z in whitened coordinates.
        """
        good = [(outer_values, r) for _, _, outer_values, r in results if r.ok]
        d = self.n_inner
        if d:
            z = np.array(list(itertools.product(_GH_NODES, repeat=d)))
            log_gh = np.array(
                [sum(t) for t in itertools.product(_GH_LOG_WEIGHTS, repeat=d)]
            )
        else:
            z = np.zeros((1, 0))
            log_gh = np.zeros(1)
        k = len(log_gh)

        log_weights = np.concatenate([r.log_weight + log_gh for _, r in good])
        nodes = {
            axis.variable: np.repeat(
                np.array([ov[axis.variable] for ov, _ in good], dtype=float), k
            )
            for axis in self.axes
        }
        if d:
            flat = np.concatenate([r.mode[None, :] + z @ r.chol_cov.T for _, r in good])
            n_nodes = flat.shape[0]
            for v, (_, s) in zip(self.inner, self._slices()):
                nodes[v] = flat[:, s].reshape((n_nodes,) + self.shapes[v])
        return VariableData(nodes), log_weights

    def _slices(self):
        offset = 0
        for v in self.inner:
            size = int(np.prod(self.shapes[v], dtype=int))
            yield v, slice(offset, offset + size)
            offset += size


def moment_projection(optimiser, factor_approx, mean_field, params=None, **kwargs):
    """
    The moments projection of ``factor_approx`` or ``None`` when it does not
    apply, in which case the caller runs the mode path unchanged.
    """
    reason = fallback_reason(
        factor_approx,
        mean_field,
        optimiser.moment_max_size,
        optimiser.moment_max_outer,
    )
    if reason is not None:
        factor = getattr(factor_approx, "factor", factor_approx)
        logger.info(
            "moment projection for %s: %s; using the mode (Laplace) projection",
            getattr(factor, "name", factor),
            reason,
        )
        return None
    return MomentProjection(optimiser, factor_approx, mean_field, params, **kwargs)()
