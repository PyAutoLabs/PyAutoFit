"""
This module provides a wrapper over the scipy.optimize.linesearch module

To work with the Factor and FactorJacobian interface defined in 
autofit.graphical

Note that this interface assumes that we're performing a maximisation. 
In scipy the interface is defined for minimisations.
"""
import warnings
from typing import Optional, Dict, Tuple

import numpy as np

from autonerves import cached_property
from autofit.graphical.factor_graphs.abstract import (
    FactorValue,
    FactorInterface,
    FactorGradientInterface,
)
from autofit.graphical.utils import FlattenArrays
from autofit.mapper.variable_operator import (
    VariableData,
    VariableLinearOperator,
    MergedVariableOperator,
)


class FlattenedState:
    def __init__(self, state, param_shapes):
        self.state = state
        self.param_shapes = param_shapes

    @classmethod
    def from_state(cls, state):
        param_shapes = FlattenArrays.from_arrays(state.parameters)
        return cls(state, param_shapes)

    def make_state(self, x):
        return self.state.update(parameters=self.param_shapes.unflatten(x))

    def __call__(self, x):
        new_state = self.make_state(x)
        return new_state.value

    def func_gradient(self, x):
        new_state = self.make_state(x)
        val, grad = new_state.value_gradient
        return val, self.param_shapes.flatten(grad)

    def _func(self, x):
        return -self(x)

    def _func_gradient(self, x):
        v, g = self.func_gradient(x)
        return -v, -g

    @property
    def parameters(self):
        return self.param_shapes.flatten(self.state.parameters)


class OptimisationState:
    def __init__(
        self,
        factor: FactorInterface,
        factor_gradient: FactorGradientInterface,
        parameters: VariableData,
        hessian: Optional[VariableLinearOperator] = None,
        det_hessian: Optional[VariableLinearOperator] = None,
        value: Optional[FactorValue] = None,
        gradient: Optional[VariableData] = None,
        search_direction: Optional[VariableData] = None,
        f_count: int = 0,
        g_count: int = 0,
        args=(),
        next_states: Optional[Dict[float, "OptimisationState"]] = None,
        lower_limit=None,
        upper_limit=None,
    ):
        self.factor = factor
        self.factor_gradient = factor_gradient

        self._parameters = None
        self.parameters = parameters
        self.hessian = hessian
        self.det_hessian = det_hessian
        self.f_count = np.asanyarray(f_count)
        self.g_count = np.asanyarray(g_count)
        self.args = args
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        self.next_states = next_states or {}

        if not self.valid:
            value = -FactorValue(np.inf)
            gradient = self.parameters.full_like(np.inf)

        self._value = value
        self._gradient = gradient

        if search_direction is not None:
            self.search_direction = search_direction

    @property
    def valid(self):
        if self.lower_limit and (self.parameters < self.lower_limit).any():
            return False

        if self.upper_limit and (self.parameters > self.upper_limit).any():
            return False

        return True

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        # This forces recalculation of the value and gradient as needed
        self._value = None
        self._gradient = None
        self._parameters = parameters

    @property
    def value(self):
        if self._value is None:
            self.f_count += 1
            self._value = self.factor(self.parameters, *self.args)

        return self._value

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self.value_gradient[1]
        return self._gradient

    @cached_property
    def value_gradient(self):
        self.g_count += 1
        self._value, self._gradient = val = self.factor_gradient(
            self.parameters, *self.args
        )
        return val

    def to_dict(self):
        # don't return value, gradient or search direction as may change
        return {
            "factor": self.factor,
            "factor_gradient": self.factor_gradient,
            "parameters": self.parameters,
            "hessian": self.hessian,
            "det_hessian": self.det_hessian,
            "f_count": self.f_count,
            "g_count": self.g_count,
            "args": self.args,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
        }

    def copy(self):
        return type(self)(**self.to_dict())

    def update(self, **kwargs):
        return type(self)(**{**self.to_dict(), **kwargs})

    def __repr__(self):
        vals = self.to_dict()
        if vals.keys():
            m = max(map(len, list(vals.keys()))) + 1
            attrs = "\n".join(
                [k.rjust(m) + " = " + repr(v) + "," for k, v in vals.items()]
            )
            return self.__class__.__name__ + f"(\n{attrs}\n)"
        else:
            return self.__class__.__name__ + "()"

    def _next_state(self, stepsize):
        next_params = VariableData.add(
            self.parameters, VariableData.mul(self.search_direction, stepsize)
        )
        # memoize stepsizes
        self.next_states[stepsize] = next_state = self.update(parameters=next_params)
        return next_state

    def step(self, stepsize):
        if not stepsize:
            return self

        stepsize = float(stepsize)
        # memoize stepsizes
        next_state = self.next_states.get(stepsize) or self._next_state(stepsize)

        return next_state

    def phi(self, s):
        next_state = self.step(s)
        return -next_state.value

    def derphi(self, s):
        next_state = self.step(s)
        return self.calc_derphi(next_state.gradient)

    def calc_derphi(self, gradient):
        return -VariableData.dot(self.search_direction, gradient)

    @property
    def all_parameters(self):
        return self.parameters.merge(self.value.deterministic_values)

    @property
    def full_hessian(self):
        if self.det_hessian:
            return MergedVariableOperator(self.hessian, self.det_hessian)

        return self.hessian

    def hessian_blocks(self):
        blocks = self.hessian.blocks()
        if self.det_hessian:
            blocks.update(self.det_hessian.blocks())

        return blocks

    def inv_hessian_blocks(self):
        blocks = self.hessian.inv().blocks()
        if self.det_hessian:
            blocks.update(self.det_hessian.inv().blocks())

        return blocks

    def hessian_diagonal(self):
        diagonal = self.hessian.diagonal()
        if self.det_hessian:
            diagonal.update(self.det_hessian.diagonal())
        return diagonal

    def flatten(self):
        return FlattenedState.from_state(self)


def line_search_wolfe1(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    c1=1e-4,
    c2=0.9,
    amax=50,
    amin=1e-8,
    xtol=1e-14,
    extra_condition=None,
    **kwargs,
) -> Tuple[Optional[float], OptimisationState]:
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`
    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    The rest of the parameters are the same as for `scalar_search_wolfe1`.
    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point
    """
    from scipy.optimize import _linesearch as linesearch

    derphi0 = state.derphi(0)
    old_fval = state.value
    stepsize, _, _ = linesearch.scalar_search_wolfe1(
        state.phi,
        state.derphi,
        -old_fval,  # we are actually performing maximisation
        old_state and -old_state.value,
        derphi0,
        c1=c1,
        c2=c2,
        amax=amax,
        amin=amin,
        xtol=xtol,
    )
    next_state = state.step(stepsize)

    if stepsize is not None and extra_condition is not None:
        if not extra_condition(stepsize, next_state):
            stepsize = None

    return stepsize, next_state


def line_search_wolfe2(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    c1=1e-4,
    c2=0.9,
    amax=None,
    extra_condition=None,
    maxiter=10,
    **kwargs,
) -> Tuple[Optional[float], OptimisationState]:
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`
    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    The rest of the parameters are the same as for `scalar_search_wolfe1`.
    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point
    """
    from scipy.optimize import _linesearch as linesearch

    derphi0 = state.derphi(0)
    old_fval = state.value
    stepsize, _, _, _ = linesearch.scalar_search_wolfe2(
        state.phi,
        state.derphi,
        -old_fval,  # we are actually performing maximisation
        old_state and -old_state.value,
        derphi0,
        c1=c1,
        c2=c2,
        amax=amax,
        maxiter=maxiter,
    )

    next_state = state.step(stepsize)
    if stepsize is not None and extra_condition is not None:
        if not extra_condition(stepsize, next_state):
            stepsize = None

    return stepsize, next_state


def line_search(
    state: OptimisationState, old_state: Optional[FactorValue] = None, **kwargs
) -> Tuple[Optional[float], OptimisationState]:
    from scipy.optimize import _linesearch as linesearch

    stepsize, next_state = line_search_wolfe1(state, old_state, **kwargs)

    if stepsize is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", linesearch.LineSearchWarning)
            stepsize, next_state = line_search_wolfe2(state, old_state, **kwargs)

    # if stepsize is None:
    #     raise _LineSearchError()

    return stepsize, next_state
