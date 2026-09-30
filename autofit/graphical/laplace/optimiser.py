import logging
from operator import attrgetter
from typing import Optional, Dict, Tuple, Any, Union

import numpy as np

from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.expectation_propagation.optimiser import AbstractFactorOptimiser
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.laplace import moments, newton
from autofit.graphical.mean_field import MeanField, FactorApproximation
from autofit.graphical.utils import FlattenArrays, Status, StatusFlag
from autofit.mapper.variable_operator import VariableData, VariableFullOperator

FactorApprox = Union[EPMeanField, FactorApproximation, Factor]

logger = logging.getLogger(__name__)


def make_posdef_hessian(mean_field, variables):
    return MeanField.precision(mean_field, variables)


class LaplaceOptimiser(AbstractFactorOptimiser):
    """
    Laplace projection of a factor's tilted distribution.

    The mode is found by quasi-Newton ascent (`newton.optimise_quasi_newton`).
    The covariance projected onto the mean field is, by default
    (``hessian="fd"``), the inverse of a central finite-difference Hessian of
    the tilted gradient assembled at that mode (`newton.finite_difference_hessian`),
    factorised as a full matrix so that its marginal blocks are exact. A Hessian
    that is not finite (mode on a limit) or not negative definite (tilted density
    not concave at the mode, e.g. a scale parameter driven to 0) is a
    ``BAD_PROJECTION``: the factor's update is skipped rather than written from a
    guess. Factors with more than ``hessian_max_size`` flattened free parameters
    fall back to ``hessian="quasi"`` — the quasi-Newton secant estimate refined
    with ``n_refine`` random draws from the mean field, which is *not*
    deterministic across runs (PyAutoFit#1561).

    A failed optimisation (line-search failure) returns the mean field it was
    handed, unchanged, so that the caller's projection reproduces the previous
    message exactly.

    ``projection="moments"`` replaces the mode/curvature pair by a moment match
    of the tilted distribution, computed by nested quadrature
    (`autofit.graphical.laplace.moments`): an outer Gauss–Legendre rule of
    ``n_quadrature`` nodes over each scale variable of a hierarchical factor
    (``_HierarchicalFactor.scale_variables``) or bounded-support variable
    (e.g. a ``TruncatedGaussianPrior`` σ), windowed to the support and to the
    cavity ``mean ± quadrature_half_width · std``, and an inner conditional
    Laplace approximation over the remaining variables at each node. This is
    the projection that recovers a hierarchical scatter whose tilted density
    sits on σ = 0, where the mode path has no interior mode (PyAutoFit#1654).
    It is deterministic (no random draws). A factor with no such variable, more
    than ``moment_max_outer`` of them, a non-scalar one, more than
    ``moment_max_size`` flattened free parameters, or deterministic variables
    takes the mode path unchanged. The default ``"mode"`` is the Laplace
    projection described above.
    """

    def __init__(
        self,
        make_hessian=make_posdef_hessian,
        search_direction=newton.newton_abs_direction,
        calc_line_search=newton.line_search,
        quasi_newton_update=newton.full_diag_update,
        stop_conditions=newton.stop_conditions,
        make_det_hessian=None,
        max_iter: int = 100,
        n_refine: int = 3,
        hessian: str = "fd",
        fd_step: float = 1e-2,
        fd_min_step: float = 1e-8,
        hessian_max_size: int = 64,
        hessian_kws: Optional[Dict[str, Any]] = None,
        det_hessian_kws: Optional[Dict[str, Any]] = None,
        search_direction_kws: Optional[Dict[str, Any]] = None,
        line_search_kws: Optional[Dict[str, Any]] = None,
        quasi_newton_kws: Optional[Dict[str, Any]] = None,
        stop_kws: Optional[Dict[str, Any]] = None,
        check_limits=True,
        projection: str = "mode",
        n_quadrature: int = 64,
        quadrature_half_width: float = 8.0,
        moment_max_size: int = 4,
        moment_max_outer: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)

        if hessian not in ("fd", "quasi"):
            raise ValueError(
                f"hessian must be 'fd' or 'quasi', got {hessian!r}"
            )
        if projection not in ("mode", "moments"):
            raise ValueError(
                f"projection must be 'mode' or 'moments', got {projection!r}"
            )

        self.make_hessian = make_hessian
        self.make_det_hessian = make_det_hessian or make_hessian
        self.search_direction = search_direction
        self.calc_line_search = calc_line_search
        self.quasi_newton_update = quasi_newton_update
        self.stop_conditions = stop_conditions

        self.max_iter = max_iter
        self.n_refine = n_refine
        # Covariance at the mode: "fd" (finite-difference Hessian, default) or
        # "quasi" (the secant estimate refined with `n_refine` random draws).
        self.hessian = hessian
        # Finite-difference step per parameter, relative to the mean-field std
        # (floored at `fd_min_step`). With forward-difference factor gradients
        # (eps=1e-8) the gradient noise is ~2e-8·|f|, so the step should stay
        # well above 1e-4·std; 1e-2·std keeps the central-difference truncation
        # error below 1e-4 relative for a Gaussian.
        self.fd_step = fd_step
        self.fd_min_step = fd_min_step
        # Above this many flattened free parameters the 2n gradient evaluations
        # of the finite-difference Hessian are traded for the quasi-Newton path.
        self.hessian_max_size = hessian_max_size

        self.hessian_kws = hessian_kws or {}
        self.det_hessian_kws = det_hessian_kws or hessian_kws or {}
        self.search_direction_kws = search_direction_kws or {}
        self.line_search_kws = line_search_kws or {}
        self.quasi_newton_kws = quasi_newton_kws or {}
        self.stop_kws = stop_kws or {}
        self.check_limits = check_limits

        # Tilted-distribution projection: "mode" (Laplace, default) or
        # "moments" (nested-quadrature moment match, `laplace.moments`).
        self.projection = projection
        self.n_quadrature = n_quadrature
        self.quadrature_half_width = quadrature_half_width
        self.moment_max_size = moment_max_size
        self.moment_max_outer = moment_max_outer

    @property
    def default_kws(self):
        return dict(
            max_iter=self.max_iter,
            n_refine=self.n_refine,
            search_direction=self.search_direction,
            calc_line_search=self.calc_line_search,
            quasi_newton_update=self.quasi_newton_update,
            stop_conditions=self.stop_conditions,
            search_direction_kws=self.search_direction_kws,
            line_search_kws=self.line_search_kws,
            quasi_newton_kws=self.quasi_newton_kws,
            stop_kws=self.stop_kws,
        )

    def prepare_state(
        self,
        factor_approx: FactorApprox,
        mean_field: MeanField = None,
        params: VariableData = None,
    ) -> newton.OptimisationState:
        mean_field = mean_field or factor_approx.model_dist

        free_variables = factor_approx.free_variables
        det_variables = factor_approx.deterministic_variables

        parameters = MeanField.mean.fget(mean_field)
        if params:
            for v, p in params.items():
                parameters[v] = p

        hessian = self.make_hessian(mean_field, free_variables, **self.hessian_kws)
        if det_variables:
            det_hessian = self.make_hessian(mean_field, det_variables)
        else:
            det_hessian = None

        kws = {}
        if self.check_limits:
            kws["upper_limit"] = MeanField.upper_limit.fget(mean_field)
            kws["lower_limit"] = MeanField.lower_limit.fget(mean_field)

        return newton.OptimisationState(
            factor_approx,
            factor_approx.func_gradient,
            parameters.subset(free_variables),
            hessian,
            det_hessian,
            **kws
        )

    def optimise_state(
        self,
        state: newton.OptimisationState,
        old_state: Optional[newton.OptimisationState] = None,
        **kwargs
    ) -> Tuple[bool, newton.OptimisationState, str]:
        kws = {**self.default_kws, **kwargs}
        return newton.optimise_quasi_newton(state, old_state, **kws)

    def fd_steps(
        self, state: newton.OptimisationState, mean_field: MeanField
    ) -> VariableData:
        """
        The central-difference step for each free parameter: a fixed fraction
        `fd_step` of the mean field's standard deviation, floored at
        `fd_min_step`. Shared by every finite difference this optimiser takes,
        so the mode Hessian and the deterministic Jacobian are differenced on
        the same scale.
        """
        # `variance` rather than `std`: TransformedMessage exposes only the former
        variance = MeanField.variance.fget(mean_field)
        return VariableData(
            {
                v: np.maximum(
                    self.fd_step
                    * np.sqrt(np.abs(np.asanyarray(variance[v], dtype=float))),
                    self.fd_min_step,
                )
                for v in state.parameters
            }
        )

    def make_mode_hessian(
        self, state: newton.OptimisationState, mean_field: MeanField
    ) -> Tuple[Optional[VariableFullOperator], str]:
        """
        The negative finite-difference Hessian of the tilted log-density at
        `state`, as a Cholesky-factorised full operator over the free variables.

        Returns ``(operator, "")`` or ``(None, reason)`` when the Hessian is not
        finite or not negative definite.
        """
        H, shapes = newton.finite_difference_hessian(
            state, self.fd_steps(state, mean_field)
        )
        if not np.all(np.isfinite(H)):
            return None, "non-finite Hessian at mode (mode on a limit / outside support)"
        try:
            # cho_factor raises LinAlgError if -H is not positive definite
            return VariableFullOperator.from_posdef(-H, shapes), ""
        except np.linalg.LinAlgError:
            min_eig = np.linalg.eigvalsh(-H).min()
            return (
                None,
                f"tilted log-density not concave at mode (min eig {min_eig:.3g})",
            )

    def make_deterministic_hessian(
        self,
        state: newton.OptimisationState,
        hessian: VariableFullOperator,
        mean_field: MeanField,
    ) -> VariableFullOperator:
        """
        The precision of the factor's deterministic outputs implied by the free
        variables' mode covariance, ``diag(J Sigma J.T) ** -1``.

        `J` is the central-difference Jacobian of the factor's deterministic
        outputs with respect to the free parameters, differenced on the same
        steps as the mode Hessian (`fd_steps`); ``Sigma`` is the inverse of
        `hessian`.

        Only the diagonal is written. ``J Sigma J.T`` is rank deficient whenever
        there are more deterministic outputs than free parameters — the common
        case, e.g. a regression factor with 50 outputs and 3 parameters — so the
        full block is not a valid precision, and `MeanField.from_mode` reads only
        the per-variable marginals in any case. A deterministic output the free
        parameters do not move (an all-zero Jacobian row, so zero or non-finite
        variance) keeps the cavity curvature it came in with, rather than being
        handed an infinite precision.
        """
        deterministic = state.value.deterministic_values
        det_variables = sorted(deterministic.keys(), key=attrgetter("id"))
        det_shapes = FlattenArrays(
            {v: np.shape(deterministic[v]) for v in det_variables}
        )

        shapes = hessian.param_shapes
        x = shapes.flatten(state.parameters)
        step = shapes.flatten(self.fd_steps(state, mean_field))

        jacobian = np.empty((det_shapes.splits[-1], x.size))
        for i in range(x.size):
            offset = np.zeros(x.size)
            offset[i] = step[i]
            plus = state.update(
                parameters=shapes.unflatten(x + offset)
            ).value.deterministic_values
            minus = state.update(
                parameters=shapes.unflatten(x - offset)
            ).value.deterministic_values
            jacobian[:, i] = (
                det_shapes.flatten(plus) - det_shapes.flatten(minus)
            ) / (2 * step[i])

        covariance = hessian.inv().operator.to_dense()
        variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)

        moved = np.isfinite(variance) & (variance > 0)
        cavity = det_shapes.flatten(state.det_hessian.diagonal())
        precision = np.where(moved, 1.0 / np.where(moved, variance, 1.0), cavity)

        return VariableFullOperator.from_diagonal(det_shapes.unflatten(precision))

    def optimise_approx(
        self,
        factor_approx: FactorApprox,
        mean_field: MeanField = None,
        params: VariableData = None,
        **kwargs
    ) -> Tuple[MeanField, Status]:

        mean_field = mean_field or factor_approx.model_dist
        if self.projection == "moments":
            result = self._moment_projection(
                factor_approx, mean_field, params, **kwargs
            )
            if result is not None:
                return result

        state = self.prepare_state(factor_approx, mean_field, params)
        next_state, status = self.optimise_state(state, **kwargs)
        if not status.success:
            # The optimiser did not land anywhere: hand back the mean field
            # unchanged so the projection reproduces the previous message.
            return mean_field, status

        next_state = max(state, next_state, key=lambda x: x.value)

        n_params = sum(np.size(p) for p in next_state.parameters.values())
        if self.hessian == "fd" and n_params <= self.hessian_max_size:
            hessian, reason = self.make_mode_hessian(next_state, mean_field)
            if hessian is None:
                factor_name = getattr(
                    getattr(factor_approx, "factor", factor_approx), "name", factor_approx
                )
                return (
                    mean_field,
                    Status(
                        success=False,
                        messages=status.messages
                        + (f"Laplace Hessian for {factor_name}: {reason}",),
                        updated=False,
                        flag=StatusFlag.BAD_PROJECTION,
                        result=status.result,
                    ),
                )
            next_state.hessian = hessian
            if next_state.det_hessian is not None:
                # The fd branch never reaches `newton.quasi_deterministic_update`
                # (that runs in the line search and in `refine_state`, both of
                # which live in the `else` below), so without this the
                # deterministic variables would be projected at the cavity
                # precision `prepare_state` built — PyAutoFit#1570.
                next_state.det_hessian = self.make_deterministic_hessian(
                    next_state, hessian, mean_field
                )
        else:
            if self.hessian == "fd":
                logger.info(
                    "Laplace Hessian: %d free parameters exceed hessian_max_size=%d; "
                    "falling back to the quasi-Newton estimate (hessian='quasi')",
                    n_params,
                    self.hessian_max_size,
                )
            next_state = self.refine_state(
                next_state, mean_field.sample, n_refine=kwargs.get("n_refine")
            )

        projection = mean_field.from_opt_state(next_state)
        return projection, status

    def _moment_projection(
        self,
        factor_approx: FactorApprox,
        mean_field: MeanField,
        params: VariableData = None,
        **kwargs
    ) -> Optional[Tuple[MeanField, Status]]:
        """
        The moment-matching projection of `factor_approx`'s tilted distribution
        (`laplace.moments.moment_projection`), or ``None`` when the factor has
        no scale / bounded-support variable to integrate over (or too many, or
        deterministic variables), in which case the mode path runs unchanged.
        """
        return moments.moment_projection(
            self, factor_approx, mean_field, params, **kwargs
        )

    def refine_state(self, state, new_param, n_refine=None):
        """
        Refine the quasi-Newton Hessian estimate of `state` with `n_refine`
        secant updates at parameters drawn from `new_param()`. The updates
        chain: each draw's secant is applied to the previously refined
        estimate, not to the original one.
        """
        if n_refine is None:
            n_refine = self.n_refine
        next_state = state
        for _ in range(n_refine):
            new_state = next_state.update(parameters=new_param())
            next_state = self.quasi_newton_update(
                next_state, new_state, **self.quasi_newton_kws
            )

        return next_state

    def refine_approx(
        self,
        factor_approx: FactorApprox,
        mean_field: MeanField = None,
        params: VariableData = None,
        n_refine=None,
    ) -> Tuple[MeanField, Status]:
        mean_field = mean_field or factor_approx.model_dist
        state = self.prepare_state(factor_approx, mean_field, params)
        if n_refine is None:
            n_refine = self.n_refine
        next_state = self.refine_state(state, mean_field.sample, n_refine=n_refine)
        return mean_field.from_opt_state(next_state)

    def optimise(
        self,
        factor_approx: FactorApproximation,
        status: Optional[Status] = Status(),
        **kwargs
    ) -> Tuple[MeanField, Status]:
        return self.optimise_approx(factor_approx, **kwargs)
