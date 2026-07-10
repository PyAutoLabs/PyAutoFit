"""
Messages composed with deterministic transforms.

A ``TransformedMessage`` represents a distribution over a *physical*
variable as a base exponential-family message over a *base* variable plus
a stack of invertible transforms. For example::

    UniformPrior(0, 2)  =  TransformedMessage(
        UniformNormalMessage,                        # base: N(0, 1) + Φ
        LinearShiftTransform(shift=0, scale=2),      # base → physical
    )

Composition-order convention (the single most important fact in this
module):

- ``transforms`` are stored **innermost-first**: going base → physical
  they apply in tuple order (first element first), so the outermost
  transform — the last function applied — is the *last* tuple element.
- ``_transform`` maps **physical → base** and therefore applies the
  transforms in *reverse* tuple order, unwinding the outermost transform
  (the last element) first.
- ``_inverse_transform`` maps **base → physical** and applies
  ``inv_transform`` in *forward* tuple order, rebuilding the composition
  from the inside out.

Worked example for ``UniformPrior(0, 2).value_for(0.5)`` (base →
physical, forward order):

    1. ``NormalMessage(0, 1).value_for(0.5)``  →  0.0
    2. ``phi_transform.inv_transform(0.0)`` = Φ(0) →  0.5
    3. ``LinearShiftTransform(0, 2).inv_transform(0.5)`` = 0.5·2 + 0 → 1.0

Evaluating ``logpdf(x)`` at a physical ``x`` runs the same three steps in
reverse (physical → base) via ``_transform``, accumulating the
log-Jacobian ``log_det`` of each transform for the change of variables.

A new transform added to the stack must therefore implement
``transform`` (physical → base), ``inv_transform`` (base → physical) and
``log_det`` (the log-Jacobian of ``transform``); a ``LinearTransform``'s
stored operator is the Jacobian of the physical → base direction, which
is why ``LinearShiftTransform(scale=s)`` stores ``1/s``.
"""
import functools
import numpy as np
import warnings
from typing import Tuple, Optional, Union

from autofit.messages.abstract import MessageInterface, AbstractMessage
from autofit.messages.transform import AbstractDensityTransform


def arithmetic(func):
    """
    When arithmetic is performed between a two transformed messages the
    operation is performed between the base messages and the result it
    encapsulated in a transformed message with the same set of transforms.
    """

    @functools.wraps(func)
    def wrapper(self, other):
        if isinstance(other, TransformedMessage):
            other = other.base_message
        return self.with_base(func(self, other))

    return wrapper


def transform(func):
    """
    Decorator to transform the function argument in the space of the
    transformed message to the space of the underlying message.

    For example, a UniformPrior with limits 10 and 20 could be passed
    a value 15. If the underlying message is a NormalMessage with a
    mean of 0 then the result would be 0.
    """

    @functools.wraps(func)
    def wrapper(self, x, xp=np):
        x = self._transform(x)
        return func(self, x, xp)

    return wrapper


def inverse_transform(func):
    """
    Decorator to transform the result of a function in the space of
    the base message to a value in the space of the transformed message.

    Inverts transform (above)
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        return self._inverse_transform(result)

    return wrapper


class TransformedMessage(MessageInterface):
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        base_message: MessageInterface,
        *transforms: AbstractDensityTransform,
        id_: Optional[int] = None,
        lower_limit=float("-inf"),
        upper_limit=float("inf"),
    ):
        """
        Comprises a base message such as a normal message and a list of transforms
        that transform the message into some new distribution, for example the
        shifted uniform distribution which underpins a UniformPrior.

        Parameters
        ----------
        base_message
            A message
        transforms
            A list of transforms applied left to right. For example, a shifted uniform
            normal message is first converted to uniform normal then shifted
        id_
        """
        while isinstance(base_message, TransformedMessage):
            transforms = base_message.transforms + transforms
            base_message = base_message.base_message

        self.transforms = transforms
        self.base_message = base_message
        self.id = id_

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        x0, x1 = zip(*base_message._support)
        z0 = self._inverse_transform(np.array(x0))
        z1 = self._inverse_transform(np.array(x1))
        self._support = tuple(zip(z0, z1))

    log_normalisation = AbstractMessage.log_normalisation

    def from_natural_parameters(self, new_params, **kwargs):
        return self.with_base(
            self.base_message.from_natural_parameters(new_params, **kwargs,)
        )

    @property
    def broadcast(self):
        return self.base_message.broadcast

    def check_support(self) -> np.ndarray:
        return self.base_message.check_support()

    def __call__(self, *args, **kwargs):
        kwargs["id_"] = kwargs.get("id_") or self.id
        return self.with_base(type(self.base_message)(*args, **kwargs))

    def copy(self):
        return TransformedMessage(self.base_message, *self.transforms, id_=self.id)

    def with_base(self, message: MessageInterface) -> "TransformedMessage":
        """
        Creates a new TransformedMessage with the same id and transforms but a new
        underlying base message
        """
        return TransformedMessage(message, *self.transforms, id_=self.id)

    @arithmetic
    def __mul__(self, other):
        return self.base_message * other

    @arithmetic
    def __pow__(self, other):
        return self.base_message ** other

    @arithmetic
    def __rmul__(self, other):
        return self.base_message * other

    @arithmetic
    def __truediv__(self, other):
        return self.base_message / other

    @arithmetic
    def __sub__(self, other):
        return self.base_message - other

    def project(
        self, samples, log_weight_list, **_,
    ):
        return TransformedMessage(
            self.base_message.project(samples, log_weight_list),
            *self.transforms,
            id_=self.id,
        )

    def kl(self, dist):
        """
        KL divergence computed between the base messages.

        Valid because KL is invariant under a common invertible change of
        variables — which requires ``dist`` to share this message's
        transform stack (true inside EP, where both messages describe the
        same variable). For messages with different transforms the result
        is meaningless; no check is performed here.
        """
        return self.base_message.kl(dist.base_message)

    def natural_parameters(self, xp=np) -> np.ndarray:
        return self.base_message.natural_parameters(xp=xp)

    @inverse_transform
    def sample(self, n_samples: Optional[int] = None):
        return self.base_message.sample(n_samples)

    def _transform(self, x: float) -> float:
        """
        Map a value from *physical* space (the transformed message) to
        *base* space (the underlying message).

        Applies the transforms in REVERSE tuple order: the stack is stored
        innermost-first, so going physical -> base must unwind the outermost
        transform (the last tuple element) first (see the module docstring
        for the worked UniformPrior example).

        For example, a UniformPrior with limits 10 and 20 could be passed
        a value 15. If the underlying message is a NormalMessage with a
        mean of 0 then the result would be 0.

        Parameters
        ----------
        x
            A value in the space of the transformed message

        Returns
        -------
        A value in the space of the base message
        """
        for _transform in reversed(self.transforms):
            x = _transform.transform(x)
        return x

    def _inverse_transform(self, x: float) -> float:
        """
        Map a value from *base* space (the underlying message) to *physical*
        space (the transformed message). Inverts ``_transform``.

        Applies each transform's ``inv_transform`` in FORWARD tuple order,
        rebuilding the composition from the inside out — the asymmetry with
        ``_transform`` is deliberate and load-bearing (module docstring).
        """
        for _transform in self.transforms:
            x = _transform.inv_transform(x)
        return x

    def _transform_det(self, x):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logd = 0
            for _transform in reversed(self.transforms):
                x, _logd = _transform.transform_det(x)
                logd += _logd
            return x, logd
    
    def _transform_det_jac(self, x):
        logd = 0
        logd_jacs = []
        for _transform in reversed(self.transforms):
            x, _logd, _logd_grad, _jac = _transform.transform_det_jac(x)
            logd += _logd
            logd_jacs.append((_logd_grad, _jac))

        return x, logd, logd_jacs


    def invert_natural_parameters(
        self, natural_parameters: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        return self.base_message.invert_natural_parameters(natural_parameters)

    @transform
    def cdf(self, x, xp=np):
        return self.base_message.cdf(x, xp=xp)

    def log_partition(self, xp=np) -> np.ndarray:
        return self.base_message.log_partition(xp=xp)

    def invert_sufficient_statistics(self, sufficient_statistics):
        return self.base_message.invert_sufficient_statistics(sufficient_statistics)

    @inverse_transform
    def value_for(self, unit):
        return self.base_message.value_for(unit)

    @transform
    def calc_log_base_measure(self, x, xp=np) -> np.ndarray:
        return self.base_message.calc_log_base_measure(x, xp=xp)

    @transform
    def to_canonical_form(self, x, xp=np) -> np.ndarray:
        return self.base_message.to_canonical_form(x, xp=xp)

    @property
    @inverse_transform
    def mean(self) -> np.ndarray:
        return self.base_message.mean

    @property
    def variance(self) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        variance = self.base_message.variance
        mean = self.base_message.mean
        for _transform in self.transforms:
            mean = _transform.inv_transform(mean)
            jacobian = _transform.jacobian(mean)
            variance = jacobian.invquad(variance)

        return variance

    @inverse_transform
    def _sample(self, n_samples) -> np.ndarray:
        return self.base_message._sample(n_samples)
    
    def exp_factor(self, x):
        return np.exp(np.nan_to_num(self.factor(x), nan=-np.inf))

    def factor(self, x: Union[float, np.ndarray]) -> Union[np.ndarray, float]:
        """
        Call the factor. The closer to the mean a given value is the higher
        the probability returned.

        Parameters
        ----------
        x
            A value in the space of the transformed message.

        Returns
        -------
        The probability this value is correct
        """
        x, logd = self._transform_det(x)
        return self.base_message.logpdf(x) + logd
    
    def factor_gradient(self, x: Union[float, np.ndarray]) -> Tuple[Union[np.ndarray, float],Union[np.ndarray, float]]:
        """
        Call the factor. The closer to the mean a given value is the higher
        the probability returned.

        Parameters
        ----------
        x
            A value in the space of the transformed message.

        Returns
        -------
        The probability this value is correct
        """
        x, logd, logd_grad, jacs = self._transform_det_jac(x)
        logp, grad = self.base_message.logpdf_gradient(x)
        for jac in reversed(jacs):
            grad = grad * jac 
        return logp + logd, grad + logd_grad
    


    

    @property
    def multivariate(self):
        return self.base_message.multivariate

    def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        jacobians = []
        for _transform in reversed(self.transforms):
            x, jacobian = _transform.transform_jac(x)
            jacobians.append(jacobian)

        log_likelihood, gradient = self.base_message.logpdf_gradient(x)

        for jacobian in reversed(jacobians):
            gradient = gradient * jacobian

        return log_likelihood, gradient

    def from_mode(self, mode: np.ndarray, covariance: np.ndarray, **kwargs):
        jac = None
        for _transform in reversed(self.transforms):
            mode, jac = _transform.transform_jac(mode)

        if covariance.shape != ():
            covariance = jac.quad(covariance)

        return self.with_base(self.base_message.from_mode(mode, covariance, **kwargs))

    def update_invalid(self, other: "TransformedMessage") -> "MessageInterface":
        return self.with_base(self.base_message.update_invalid(other.base_message))

    @property
    def log_base_measure(self):
        return self.base_message.log_base_measure

    def zeros_like(self) -> "MessageInterface":
        return self ** 0.