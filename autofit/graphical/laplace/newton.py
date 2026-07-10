import logging
import warnings
from typing import Optional, Dict, Tuple, Any, Callable

import numpy as np

from autofit import exc
from autofit.graphical.laplace.line_search import line_search, OptimisationState
from autofit.graphical.utils import Status, StatusFlag, LogWarnings
from autofit.mapper.variable_operator import VariableData


## get ascent direction


def gradient_ascent(state: OptimisationState, **kwargs) -> VariableData:
    return state.gradient


def newton_direction(state: OptimisationState, **kwargs) -> VariableData:
    return state.hessian.ldiv(state.gradient)


def newton_abs_direction(state: OptimisationState, d=1e-6, **kwargs) -> VariableData:
    posdef = state.hessian.abs().diagonalupdate(state.parameters.full_like(d))
    return posdef.ldiv(state.gradient)


logger = logging.getLogger(__name__)
logging.captureWarnings(False)
_log_projection_warnings = logger.debug


## Quasi-newton approximations


def sr1_update(
    state1: OptimisationState, state: OptimisationState, mintol=1e-8, **kwargs
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.parameters, state.parameters)
    Bk = state.hessian
    zk = yk + Bk * dk
    zkdk = -zk.dot(dk)

    tol = mintol * dk.norm() * zk.norm()
    if zkdk > tol:
        vk = zk / np.sqrt(zkdk)
        Bk1 = Bk.lowrankupdate(vk)
    elif zkdk < -tol:
        vk = zk / np.sqrt(-zkdk)
        Bk1 = Bk.lowrankdowndate(vk)
    else:
        Bk1 = Bk

    state1.hessian = Bk1
    return state1


def diag_sr1_update(
    state1: OptimisationState, state: OptimisationState, tol=1e-8, **kwargs
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.parameters, state.parameters)
    Bk = state.hessian
    zk = yk + Bk * dk
    dzk = dk * zk
    # alpha = -zk.dot(dk) / dzk.dot(dzk)

    d = dzk.dot(dzk)
    if d > tol * dk.norm() ** 2 * zk.norm() ** 2:
        alpha = -zk.dot(dk) / d
        Bk = Bk.diagonalupdate((zk ** 2) * alpha)

    state1.hessian = Bk
    return state1


# Dead/suspect quasi-Newton variants removed (#1332 F8):
# - `diag_sr1_update_` — never referenced anywhere in the codebase.
# - `diag_sr1_bfgs_update` — computed locals then implicitly returned None
#   (would have nuked the optimiser state if ever wired in).
# - `bfgs1_update` — disagreed with the exported `bfgs_update` on the sign of
#   the y_k difference and carried a stray minus on d_k^T B d_k; the exported
#   `bfgs_update` below is the verified-correct implementation.


def bfgs_update(
    state1: OptimisationState, state: OptimisationState, **kwargs,
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.parameters, state.parameters)
    Bk = state.hessian

    ykTdk = -yk.dot(dk)
    Bdk = Bk.dot(dk)
    dkTBdk = VariableData.dot(Bdk, dk)

    state1.hessian = Bk.update(
        (yk, VariableData(yk).div(ykTdk)), (Bdk, VariableData(Bdk).div(dkTBdk))
    )
    return state1


def quasi_deterministic_update(
    state1: OptimisationState, state: OptimisationState, **kwargs,
) -> OptimisationState:
    dk = VariableData.sub(state1.parameters, state.parameters)
    zk = VariableData.sub(
        state1.value.deterministic_values, state.value.deterministic_values
    )
    Bxk, Bzk = state1.hessian, state.det_hessian
    zkTzk2 = zk.dot(zk) ** 2
    alpha = (dk.dot(Bxk.dot(dk)) - zk.dot(Bzk.dot(zk))) / zkTzk2
    if alpha >= 0:
        Bzk1 = Bzk.lowrankupdate(np.sqrt(alpha) * (zk))
    else:
        Bzk1 = Bzk.lowrankdowndate(np.sqrt(-alpha) * (zk))

    state1.det_hessian = Bzk1
    return state1


def diag_quasi_deterministic_update(
    state1: OptimisationState, state: OptimisationState, **kwargs,
) -> OptimisationState:
    dk = VariableData.sub(state1.parameters, state.parameters)
    zk = VariableData.sub(
        state1.value.deterministic_values, state.value.deterministic_values
    )
    Bxk, Bzk = state1.hessian, state.det_hessian
    zk2 = zk ** 2
    zk4 = (zk2 ** 2).sum()
    alpha = (dk.dot(Bxk.dot(dk)) - zk.dot(Bzk.dot(zk))) / zk4
    state1.det_hessian = Bzk.diagonalupdate(zk2 * alpha)

    return state1


class QuasiNewtonUpdate:
    def __init__(self, quasi_newton_update, det_quasi_newton_update):
        self.quasi_newton_update = quasi_newton_update
        self.det_quasi_newton_update = det_quasi_newton_update

    def __call__(
        self, state1: OptimisationState, state: OptimisationState, **kwargs,
    ) -> OptimisationState:

        # Only update estimate if a step has been taken
        state1 = self.quasi_newton_update(state1, state, **kwargs)
        if state.det_hessian:
            state1 = self.det_quasi_newton_update(state1, state, **kwargs)

        return state1


full_bfgs_update = QuasiNewtonUpdate(bfgs_update, quasi_deterministic_update)
full_sr1_update = QuasiNewtonUpdate(sr1_update, quasi_deterministic_update)
full_diag_update = QuasiNewtonUpdate(diag_sr1_update, diag_quasi_deterministic_update)


## Newton step


def take_step(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    search_direction=newton_abs_direction,
    calc_line_search=line_search,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], OptimisationState]:
    state.search_direction = search_direction(state, **(search_direction_kws or {}))
    return calc_line_search(state, old_state, **(line_search_kws or {}))


def take_quasi_newton_step(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    search_direction=newton_abs_direction,
    calc_line_search=line_search,
    quasi_newton_update=full_bfgs_update,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], OptimisationState]:
    """ """
    state.search_direction = search_direction(state, **(search_direction_kws or {}))
    if state.search_direction.vecnorm(np.inf) == 0:
        # if gradient is zero then at maximum already
        return 0.0, state

    stepsize, state1 = calc_line_search(state, old_state, **(line_search_kws or {}))
    if stepsize:
        # Only update estimate if a step has been taken
        state1 = quasi_newton_update(state1, state, **(quasi_newton_kws or {}))

    return stepsize, state1


def xtol_condition(state, old_state, xtol=1e-6, ord=None, **kwargs):
    if not old_state:
        return

    dx = VariableData.sub(state.parameters, old_state.parameters).vecnorm(ord=ord)
    if dx < xtol:
        return True, f"Minimum parameter change tolerance achieved, {dx} < {xtol}"


def grad_condition(state, old_state, gtol=1e-5, ord=None, **kwargs):
    dg = VariableData.vecnorm(state.gradient, ord=ord)
    if dg < gtol:
        return True, f"Gradient tolerance achieved, {dg} < {gtol}"


def ftol_condition(state, old_state, ftol=1e-6, monotone=True, **kwargs):
    if not old_state:
        return

    df = state.value - old_state.value
    if 0 < df < ftol:
        return True, f"Minimum function change tolerance achieved, {df} < {ftol}"
    elif monotone and df < 0:
        return False, f"factor failed to increase on next step, {df}"


def nfev_condition(state, old_state, maxfev=10000, **kwargs):
    if state.f_count > maxfev:
        return (
            True,
            f"Maximum number of function evaluations (maxfev={maxfev}) has been exceeded.",
        )


def ngev_condition(state, old_state, maxgev=10000, **kwargs):
    if state.g_count > maxgev:
        return (
            True,
            f"Maximum number of gradient evaluations (maxgev={maxgev}) has been exceeded.",
        )


stop_conditions = (
    xtol_condition,
    ftol_condition,
    grad_condition,
)

_OPT_CALLBACK = Callable[[OptimisationState, OptimisationState], None]


def check_stop_conditions(
    stepsize, state, old_state, stop_conditions, **stop_kws
) -> Optional[Tuple[bool, str]]:
    if stepsize is None:
        return False, "abnormal termination of line search"
    elif not np.isfinite(state.value):
        return False, "function is no longer finite"
    else:
        for stop_condition in stop_conditions:
            stop = stop_condition(state, old_state, **(stop_kws or {}))
            if stop:
                return stop


def optimise_quasi_newton(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    max_iter=100,
    search_direction=newton_abs_direction,
    calc_line_search=line_search,
    quasi_newton_update=bfgs_update,
    stop_conditions=stop_conditions,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
    stop_kws: Optional[Dict[str, Any]] = None,
    callback: Optional[_OPT_CALLBACK] = None,
    **kwargs,
) -> Tuple[OptimisationState, Status]:
    success = True
    updated = False
    messages = ()
    message = "max iterations reached"
    stepsize = 0.0
    for i in range(max_iter):
        stop = check_stop_conditions(
            stepsize, state, old_state, stop_conditions, **(stop_kws or {})
        )
        if stop:
            success, message = stop
            break

        with LogWarnings(
            logger=_log_projection_warnings, action="always"
        ) as caught_warnings:
            try:
                stepsize, state1 = take_quasi_newton_step(
                    state,
                    old_state,
                    search_direction=search_direction,
                    calc_line_search=calc_line_search,
                    quasi_newton_update=quasi_newton_update,
                    search_direction_kws=search_direction_kws,
                    line_search_kws=line_search_kws,
                    quasi_newton_kws=quasi_newton_kws,
                )
            except exc.MessageException as e:
                logger.exception(e)
                success = False
                message = f"Line search failed due to {e}"
                break

        for m in caught_warnings.messages:
            messages += (f"optimise_quasi_newton warning: {m}",)

        if stepsize is None:
            success = False
            message = "Line search failed"
            break

        updated = True
        state, old_state = state1, state
        i += 1
        if callback:
            callback(state, old_state)

    message += f", iter={i}"
    messages += (message,)
    status = Status(
        success,
        messages=messages,
        updated=updated,
        flag=StatusFlag.get_flag(success, i),
    )
    return state, status
