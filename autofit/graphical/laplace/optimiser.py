import logging
from typing import Optional, Dict, Tuple, Any, Union

import numpy as np

from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.expectation_propagation.optimiser import AbstractFactorOptimiser
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.laplace import newton
from autofit.graphical.mean_field import MeanField, FactorApproximation
from autofit.graphical.utils import Status, StatusFlag
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
        **kwargs
    ):
        super().__init__(**kwargs)

        if hessian not in ("fd", "quasi"):
            raise ValueError(
                f"hessian must be 'fd' or 'quasi', got {hessian!r}"
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

    def make_mode_hessian(
        self, state: newton.OptimisationState, mean_field: MeanField
    ) -> Tuple[Optional[VariableFullOperator], str]:
        """
        The negative finite-difference Hessian of the tilted log-density at
        `state`, as a Cholesky-factorised full operator over the free variables.

        Returns ``(operator, "")`` or ``(None, reason)`` when the Hessian is not
        finite or not negative definite.
        """
        # `variance` rather than `std`: TransformedMessage exposes only the former
        variance = MeanField.variance.fget(mean_field)
        step = VariableData(
            {
                v: np.maximum(
                    self.fd_step
                    * np.sqrt(np.abs(np.asanyarray(variance[v], dtype=float))),
                    self.fd_min_step,
                )
                for v in state.parameters
            }
        )
        H, shapes = newton.finite_difference_hessian(state, step)
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

    def optimise_approx(
        self,
        factor_approx: FactorApprox,
        mean_field: MeanField = None,
        params: VariableData = None,
        **kwargs
    ) -> Tuple[MeanField, Status]:

        mean_field = mean_field or factor_approx.model_dist
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
