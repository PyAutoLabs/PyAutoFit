import warnings
from abc import ABC, abstractmethod
from functools import wraps
from typing import Tuple

import numpy as np

from ..mapper.operator import DiagonalMatrix, LinearOperator, ShermanMorrison

epsilon = 1e-14


def ndtri(x):

    from scipy.special import ndtri

    x = np.array(x, dtype=float)
    x[(x <= 0) & (x >= -epsilon)] = epsilon
    x[(x >= 1) & (x <= 1 + epsilon)] = 1 - epsilon
    return ndtri(x)


def numerical_jacobian(x, func, eps=1e-8, args=(), **kwargs):
    x = np.array(x)
    f0 = func(x, *args, **kwargs)
    jac = np.empty(np.shape(f0) + np.shape(x))
    fslice = (slice(None),) * np.ndim(f0)
    with np.nditer(x, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for xi in it:
            xi += eps
            f1 = func(x, *args, **kwargs)
            jac[fslice + it.multi_index] = (f1 - f0) / eps
            xi -= eps

    return jac


class AbstractDensityTransform(ABC):
    """
    This class allows the transformation of a probability density function, p(x)
    whilst preserving the measure of the distribution, i.e.

    int p(x) dx = 1

    p'(f) = p(f(x)) * |df/dx|

    inf p'(f) df = 1

    Methods
    -------
    transform
        calculates f(x)

    inv_transform
        calculates f^{-1}(y)

    jacobian
        calculates df/dx

    log_det
        calculates log |df/dx|

    log_det_grad
        calculates |df/dx|, d log_det/dx

    transform_det
        calculates f(x), |df/dx|

    transform_jac
        calculates f(x), df/dx

    transform_det_jac
        calculates f(x), log_det, d log_det/dx, df/dx

    These final 3 functions are defined so that child classes
    can define custom methods that avoid recalculation of intermediate
    values that are needed to calculate multiple versions of the quantities
    """

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inv_transform(self, x):
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> LinearOperator:
        pass

    @abstractmethod
    def log_det(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def log_det_grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _numerical_jacobian_func(self, x):
        return self.log_det_grad(x)[1].sum(0)

    def log_det_hess(self, x: np.ndarray) -> np.ndarray:
        return numerical_jacobian(x, self._numerical_jacobian_func)

    def transform_det(self, x) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(x), self.log_det(x)

    def transform_jac(self, x) -> Tuple[np.ndarray, LinearOperator]:
        return self.transform(x), self.jacobian(x)

    def transform_det_jac(
        self, x
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LinearOperator]:
        return (self.transform(x), *self.log_det_grad(x), self.jacobian(x))

    def transform_func(self, func):
        @wraps(func)
        def transformed_func(*args, **kwargs):
            x, *args = args
            x = self.transform(x)
            return func(x, *args, **kwargs)

        transformed_func.transform = self

        return transformed_func

    def transform_func_grad(self, func_grad):
        @wraps(func_grad)
        def transformed_func_grad(*args, **kwargs):
            x, *args = args
            x, jac = self.transform_jac(x)
            val, grad = func_grad(x, *args, **kwargs)
            return x, grad * jac

        transformed_func_grad.transform = self
        return transformed_func_grad

    def transform_func_grad_hess(self, func_grad_hess):
        @wraps(func_grad_hess)
        def transformed_func_grad_hess(*args, **kwargs):
            x, *args = args
            x, jac = self.transform_jac(x)
            val, grad, hess = func_grad_hess(x, *args, **kwargs)
            return val, grad * jac, jac.quad(hess)

        transformed_func_grad_hess.transform = self
        return transformed_func_grad_hess


class LinearTransform(AbstractDensityTransform):
    def __init__(self, linear: LinearOperator):
        self.linear = linear

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.linear * x

    def inv_transform(self, x: np.ndarray) -> np.ndarray:
        return self.linear.ldiv(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.linear

    def log_det(self, x: np.ndarray) -> np.ndarray:
        return self.linear.log_det

    def log_det_grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.log_det(x), 0


class LinearShiftTransform(LinearTransform):
    """
    Affine rescaling between a base variable and a physical variable.

    ``shift`` and ``scale`` describe the **physical** space: going base →
    physical (``inv_transform``) is ``x * scale + shift``, which is how e.g.
    ``UniformPrior(lower, upper)`` maps a unit-interval base variable onto
    ``[lower, upper]`` with ``shift=lower, scale=upper - lower``.

    The Jacobian stored on the ``LinearTransform`` parent is therefore
    ``DiagonalMatrix(1 / scale)`` — deliberately the *reciprocal* of the
    intuitive physical scale — because ``transform`` runs in the opposite
    direction, physical → base (``(x - shift) / scale``), and the parent
    stores the Jacobian of ``transform``. Consistently, the change-of-variables
    term for that direction is ``log_det = -log(scale)``.

    Getting this reciprocal backwards was a contributing factor in the
    historical LogUniform sign-convention bug (#1266); see the
    ``composed_transform`` module docstring for the composition-order
    convention this class participates in.
    """

    def __init__(self, shift: float = 0, scale: float = 1):
        self.shift = float(shift)
        self.scale = float(scale)
        super().__init__(DiagonalMatrix(np.reciprocal(self.scale)))

    def inv_transform(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return x * self.scale + self.shift

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.shift) / self.scale

    def log_det(self, x: np.ndarray) -> np.ndarray:
        return -np.log(self.scale) * np.ones_like(x)


class FunctionTransform(AbstractDensityTransform):
    def __init__(self, func, inv_func, grad, hess=None, args=(), func_grad_hess=None):
        self.func = func
        self.inv_func = inv_func
        self.grad = grad
        self.hess = hess
        self.args = args
        self.func_grad_hess = func_grad_hess

    def transform(self, x):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.func(x, *self.args)

    def inv_transform(self, x):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.inv_func(x, *self.args)

    def jacobian(self, x):
        return DiagonalMatrix(self.grad(x, *self.args))

    def log_det(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            gs = self.grad(x, *self.args)
            return np.log(gs)

    def log_det_grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.func_grad_hess:
            x0, gs, hs = self.func_grad_hess(x, *self.args)
        else:
            gs = self.grad(x, *self.args)
            hs = self.hess(x, *self.args)
        return np.log(gs), hs / gs

    def transform_det_jac(
        self, x
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LinearOperator]:
        if self.func_grad_hess:
            x0, gs, hs = self.func_grad_hess(x, *self.args)
        else:
            x0 = self.func(x, *self.args)
            gs = self.grad(x, *self.args)
            hs = self.hess(x, *self.args)
        return x0, np.log(gs), hs / gs, DiagonalMatrix(gs)


def exp3(x):
    expx = np.exp(x)
    return (expx,) * 3


exp_transform = FunctionTransform(np.exp, np.log, np.exp, func_grad_hess=exp3)


def log3(x):
    ix = np.reciprocal(x)
    return np.log(x), ix, -np.square(ix)


log_transform = FunctionTransform(np.log, np.exp, np.reciprocal, func_grad_hess=log3)


def _log_10_inv(x):
    return 10**x

LOG10 = np.log(10)
def _log10_recip(x):
    return np.reciprocal(x) / LOG10

def log10_3(x):
    ix = _log10_recip(x)
    return np.log10(x), ix, -ix/x

# TODO: what should func_grad_hess look like? np.log10, ... ?
log_10_transform = FunctionTransform(np.log10, _log_10_inv, _log10_recip, func_grad_hess=log10_3)


def sigmoid(x, scale=1, shift=0):
    return scale / (1 + np.exp(-x)) + shift


def logit(x, scale=1, shift=0):
    x = (x - shift) / scale
    return np.log(x) - np.log1p(-x)


def sigmoid_grad(x, scale=1, shift=0):
    expx = np.exp(-x)
    return scale * expx / np.square(1 + expx)


def logit_grad(x, scale=1, shift=0):
    x = (x - shift) / scale
    return (np.reciprocal(x) + np.reciprocal(1 - x)) / scale


def logit_hess(x, scale=1, shift=0):
    x = (x - shift) / scale
    return np.reciprocal(1 - x) - np.reciprocal(x)


def logit_grad_hess(x, scale=1, shift=0):
    x = (x - shift) / scale
    ix = np.reciprocal(x)
    ix1 = np.reciprocal(1 - x)
    ix2 = np.square(ix)
    ix12 = np.square(ix1)
    return (np.log(x) - np.log1p(-x), (ix + ix1) / scale, (ix12 - ix2) / scale**2)


logistic_transform = FunctionTransform(
    logit, sigmoid, logit_grad, func_grad_hess=logit_grad_hess
)


def shifted_logistic(shift=0, scale=1):
    return FunctionTransform(
        logit, sigmoid, logit_grad, func_grad_hess=logit_grad_hess, args=(scale, shift)
    )


def ndtri_grad(x):
    from scipy.stats._continuous_distns import _norm_pdf

    return np.reciprocal(_norm_pdf(ndtri(x)))


def ndtri_grad_hess(x):
    from scipy.stats._continuous_distns import _norm_pdf

    f = ndtri(x)
    phi = _norm_pdf(f)
    grad = np.reciprocal(phi)
    hess = grad**2 * f
    return f, grad, hess

def lazy_ndtr(x):
    from scipy.special import ndtr
    return ndtr(x)

def lazy_ndtri(x):
    from scipy.special import ndtri
    return ndtri(x)

phi_transform = FunctionTransform(
    lazy_ndtri, lazy_ndtr, ndtri_grad, func_grad_hess=ndtri_grad_hess
)


class MultinomialLogitTransform(AbstractDensityTransform):
    """
    makes multinomial logististic transform from the p to x, where,

    x_i = log(p_i / (1 - sum(p)))
    p_i = exp(x_i) / (sum(exp(x_j) for x_j in x) + 1)

    When p's n-simplex is defined by,

    all(0 <= p_i <= 1 for p_i in p) and sum(p) < 1
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def _validate(self, p):
        p = np.asanyarray(p)
        keepdims = np.ndim(p) == self.ndim + 1
        if not (keepdims or np.ndim(p) == self.ndim):
            raise ValueError(
                f"dimension of input must be {self.ndim} or {self.ndim + 1}"
            )

        return p, keepdims

    def transform(self, p):
        p = np.asanyarray(p)
        lnp1 = np.log(1 - np.sum(p, axis=self.axis, keepdims=True))
        lnp = np.log(p)
        return lnp - lnp1

    def inv_transform(self, x):
        expx = np.exp(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return expx / (expx.sum(axis=self.axis, keepdims=True) + 1)

    def jacobian(self, p):
        p = np.asanyarray(p)
        pn1 = 1 - np.sum(p, axis=-1, keepdims=True)
        # ln1p = np.log(pn1)
        # lnp = np.log(p)
        return ShermanMorrison(
            DiagonalMatrix(1 / p), 1 / np.sqrt(pn1) * np.ones_like(p)
        )

    def log_det(self, p):
        p = np.asanyarray(p)
        p1 = 1 - np.sum(p, axis=self.axis, keepdims=True)
        # Hack to make sure summation broadcasting works correctly
        log_d = (
            -np.log(p).sum(axis=self.axis, keepdims=True) - np.log(p1)
        ) * np.full_like(p, p1.size / p.size)
        return log_d

    def log_det_grad(self, p):
        p = np.asanyarray(p)
        p1 = 1 - np.sum(p, axis=self.axis, keepdims=True)
        # Hack to make sure summation broadcasting works correctly
        log_d = (
            -np.log(p).sum(axis=self.axis, keepdims=True) - np.log(p1)
        ) * np.full_like(p, p1.size / p.size)
        return log_d, 1 / p1 - 1 / p

    def transform_det_jac(
        self, p
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LinearOperator]:
        p = np.asanyarray(p)
        pn1 = 1 - np.sum(p, axis=self.axis, keepdims=True)
        ln1p = np.log(pn1)
        lnp = np.log(p)
        x = lnp - ln1p
        # Hack to make sure summation broadcasting works correctly
        logd = (-lnp.sum(axis=self.axis, keepdims=True) - ln1p) * np.full_like(
            p, pn1.size / p.size
        )
        logd_grad = 1 / pn1 - 1 / p
        jac = ShermanMorrison(DiagonalMatrix(1 / p), 1 / np.sqrt(pn1) * np.ones_like(p))
        return (x, logd, logd_grad, jac)


multinomial_logit_transform = MultinomialLogitTransform()
