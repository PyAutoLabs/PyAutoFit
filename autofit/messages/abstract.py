import math
from abc import ABC, abstractmethod
from copy import copy
from functools import reduce
from inspect import getfullargspec
from itertools import count
from numbers import Real
from operator import and_
from typing import Dict, Tuple, Iterator
from typing import Optional, Union, Type, List

import numpy as np

from autoconf import cached_property
from ..mapper.variable import Variable

from .interface import MessageInterface

enforce_id_match = True


def update_array(arr1, ind, arr2):
    if np.shape(arr1):
        out = arr1.copy()
        out[ind] = arr2
        return out

    return arr2


class AbstractMessage(MessageInterface, ABC):

    _Base_class: Optional[Type["AbstractMessage"]] = None
    _projection_class: Optional[Type["AbstractMessage"]] = None
    _multivariate: bool = False
    _parameter_support: Optional[Tuple[Tuple[float, float], ...]] = None
    _support: Optional[Tuple[Tuple[float, float], ...]] = None

    ids = count()

    def __init__(
        self,
        *parameters: Union[np.ndarray, float],
        log_norm=0.0,
        lower_limit=-math.inf,
        upper_limit=math.inf,
        id_=None,
        _xp=np
    ):

        xp=_xp

        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)

        self.id = next(self.ids) if id_ is None else id_
        self.log_norm = log_norm


        if xp is np:
            self._broadcast = np.broadcast(*parameters)
        else:
            self._broadcast = _xp.broadcast_arrays(*parameters)

        if self.shape:
            self.parameters = tuple(xp.asarray(p) for p in parameters)
        else:
            self.parameters = tuple(parameters)

    @property
    def broadcast(self):
        return self._broadcast
    
    @property
    def _init_kwargs(self):
        return dict(
            log_norm=self.log_norm,
            id_=self.id,
        )

    def check_support(self) -> np.ndarray:
        if self._parameter_support is not None:
            return reduce(
                and_,
                (
                    (p >= support[0]) & (p <= support[1])
                    for p, support in zip(self.parameters, self._parameter_support)
                ),
            )
        elif self.ndim:
            return np.array(True, dtype=bool, ndmin=self.ndim)
        return np.array([True])

    @property
    def multivariate(self):
        return self._multivariate

    def copy(self):
        cls = self._Base_class or type(self)
        result = cls(
            *(copy(params) for params in self.parameters),
            log_norm=self.log_norm,
        )
        result.id = self.id
        return result

    def __bool__(self):
        return True

    @abstractmethod
    def sample(self, n_samples: Optional[int] = None):
        pass

    @staticmethod
    @abstractmethod
    def invert_natural_parameters(
        natural_parameters: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        pass

    @cached_property
    @abstractmethod
    def variance(self) -> np.ndarray:
        pass

    @cached_property
    def scale(self) -> np.ndarray:
        return self.std

    @cached_property
    def std(self) -> np.ndarray:
        return self.variance ** 0.5

    def __hash__(self):
        return self.id

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.parameters)

    @classmethod
    def _cached_attrs(cls):
        for n in dir(cls):
            attr = getattr(cls, n)
            if isinstance(attr, cached_property):
                yield n

    def _reset_cache(self):
        for attr in self._cached_attrs():
            self.__dict__.pop(attr, None)

    def __getitem__(self, index) -> "AbstractMessage":
        cls = self._Base_class or type(self)
        if index == ():
            return self
        else:
            return cls(*(param[index] for param in self.parameters))

    def __setitem__(self, index, value):
        self._reset_cache()
        for param0, param1 in zip(self.parameters, value.parameters):
            param0[index] = param1

    def merge(self, index, value):
        cls = self._Base_class or type(self)
        return cls(
            *(
                update_array(param0, index, param1)
                for param0, param1 in zip(self.parameters, value.parameters)
            )
        )

    @classmethod
    def from_natural_parameters(
        cls, natural_parameters: np.ndarray, **kwargs
    ) -> "AbstractMessage":
        cls_ = cls._projection_class or cls._Base_class or cls
        args = cls_.invert_natural_parameters(natural_parameters)
        return cls_(*args, **kwargs)
    
    def zeros_like(self) -> "AbstractMessage":
        return self ** 0.

    @classmethod
    @abstractmethod
    def invert_sufficient_statistics(
        cls, sufficient_statistics: np.ndarray
    ) -> np.ndarray:
        pass

    @classmethod
    def from_sufficient_statistics(
        cls, suff_stats: np.ndarray, **kwargs
    ) -> "AbstractMessage":
        natural_params = cls.invert_sufficient_statistics(suff_stats)
        cls_ = cls._projection_class or cls._Base_class or cls
        return cls_.from_natural_parameters(natural_params, **kwargs)

    def __mul__(self, other: Union["AbstractMessage", Real]) -> "AbstractMessage":
        if isinstance(other, MessageInterface):
            return self._multiply(other)
        else:
            cls = self._Base_class or type(self)
            log_norm = self.log_norm + np.log(other)
            return cls(
                *self.parameters,
                log_norm=log_norm,
                id_=self.id,
            )

    def __rmul__(self, other: "AbstractMessage") -> "AbstractMessage":
        return self * other

    def __truediv__(self, other: Union["AbstractMessage", Real]) -> "AbstractMessage":
        if isinstance(other, MessageInterface):
            return self._divide(other)
        else:
            cls = self._Base_class or type(self)
            log_norm = self.log_norm - np.log(other)
            return cls(
                *self.parameters,
                log_norm=log_norm,
                id_=self.id,
            )

    def __pow__(self, other: Real) -> "AbstractMessage":
        natural = self.natural_parameters()
        new_params = other * natural
        log_norm = other * self.log_norm
        new = self.from_natural_parameters(
            new_params,
            log_norm=log_norm,
            id_=self.id,
        )
        return new

    @classmethod
    def parameter_names(cls):
        return getfullargspec(cls.__init__).args[1:-1]

    def __str__(self) -> str:
        param_attrs = [
            (attr, np.asanyarray(getattr(self, attr)))
            for attr in self.parameter_names()
        ]
        if self.shape:
            pad = max(len(attr) for attr, _ in param_attrs)
            attr_str = "    {:<%d}={}" % pad
            param_strs = ",\n".join(
                attr_str.format(attr, np.array2string(val, prefix=" " * (pad + 5)))
                for attr, val in param_attrs
            )
            return f"{type(self).__name__}(\n{param_strs})"
        else:
            param_strs = ", ".join(
                attr + "=" + np.array2string(val, prefix=" " * (len(attr) + 1))
                for attr, val in param_attrs
            )
            return f"{type(self).__name__}({param_strs})"

    __repr__ = __str__

    def factor(self, x):
        return self.logpdf(x)

    @classmethod
    def project(
        cls, samples: np.ndarray, log_weight_list: Optional[np.ndarray] = None, **kwargs
    ) -> "AbstractMessage":
        """
        Moment-matching projection: the exponential-family member closest
        (in inclusive KL) to the weighted sample distribution.

        For an exponential family, argmin_q KL(p || q) is the member whose
        expected sufficient statistics match p's:

            E_q[T(x)] = E_p[T(x)]

        Estimated by importance-weighted sample moments over weighted
        samples {x_s, log w_s} (e.g. from a nested sampler run on the EP
        tilted distribution):

            E[T] ≈ Σ_s w̃_s T(x_s) / Σ_s w̃_s ,  w̃_s = exp(log w_s − max log w)

        The max-log-weight shift is the standard numerical stabilisation;
        the mean unshifted weight supplies the projection's ``log_norm``
        (the tilted distribution's normalisation estimate).
        ``from_sufficient_statistics`` then inverts the moment equations to
        natural parameters for the concrete family.
        """
        # if weight_list aren't passed then equally weight all samples

        # Numerically stable weighting for very small/large weight_list

        # rescale coordinates to 'natural parameter space'
        if log_weight_list is None:
            log_weight_list = np.zeros_like(samples)

        log_w_max = np.max(log_weight_list, axis=0, keepdims=True)
        w = np.exp(log_weight_list - log_w_max)
        norm = w.mean(0)
        log_norm = np.log(norm) + log_w_max[0]

        tx = cls.to_canonical_form(samples)
        w /= norm
        suff_stats = (tx * w[None, ...]).mean(1)

        assert np.isfinite(suff_stats).all()

        cls_ = cls._projection_class or cls._Base_class or cls
        return cls_.from_sufficient_statistics(suff_stats, log_norm=log_norm, **kwargs)

    @classmethod
    def from_mode(
        cls, mode: np.ndarray, covariance: np.ndarray, **kwargs
    ) -> "AbstractMessage":
        pass

    def log_normalisation(self, *elems: Union["AbstractMessage", float]) -> np.ndarray:
        """
        Calculates the log of the integral of the product of a
        set of distributions

        NOTE: ignores log normalisation
        """
        # Remove floats from messages passed

        dists: List[MessageInterface] = [
            dist
            for dist in self._iter_dists(elems)
            if isinstance(dist, MessageInterface,)
        ]

        # Calculate log product of message normalisation
        log_norm = self.log_base_measure - self.log_partition()
        log_norm += sum(dist.log_base_measure - dist.log_partition() for dist in dists)

        # Calculate log normalisation of product of messages
        prod_dist = self.sum_natural_parameters(*dists)
        log_norm -= prod_dist.log_base_measure - prod_dist.log_partition()

        return log_norm

    def instance(self):
        return self

    def update_invalid(self, other: "AbstractMessage") -> "AbstractMessage":
        valid = self.check_valid()
        if self.ndim:
            valid_parameters: Iterator[np.ndarray] = (
                np.where(valid, p, p_safe) for p, p_safe in zip(self, other)
            )
        else:
            # TODO: Fairly certain this would not work
            valid_parameters = iter(self if valid else other)
        cls = self._Base_class or type(self)
        new = cls(
            *valid_parameters,
            log_norm=self.log_norm,
            id_=self.id,
        )
        return new

    @staticmethod
    def _get_mean_variance(
        mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, covariance = np.asanyarray(mean), np.asanyarray(covariance)

        if not covariance.shape:
            # If variance is float simply pass through
            variance = covariance * np.ones_like(mean)
            if not variance.shape:
                variance = variance.item()
        elif mean.shape == covariance.shape:
            variance = np.asanyarray(covariance)
        elif covariance.shape == mean.shape * 2:
            # If 2D covariance matrix passed get diagonal
            inds = tuple(np.indices(mean.shape))
            variance = np.asanyarray(covariance)[inds * 2]
        else:
            raise ValueError(
                f"shape of covariance {covariance.shape} is invalid "
                f"must be (), {mean.shape}, or {mean.shape * 2}"
            )
        return mean, variance

    def __call__(self, x, xp=np):
        return xp.sum(self.logpdf(x, xp=xp))

    def factor_jacobian(
        self, x: np.ndarray, _variables: Optional[Tuple[str]] = ("x",)
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, ...]]]:
        loglike, g = self.logpdf_gradient(x)
        g = np.expand_dims(g, list(range(loglike.ndim)))
        return loglike.sum(), (g,)

    def as_factor(self, variable: "Variable", name: Optional[str] = None):
        from autofit.graphical.factor_graphs import Factor

        if name is None:
            shape = self.shape
            clsname = type(self).__name__
            family = clsname[:-7] if clsname.endswith("Message") else clsname
            name = f"{family}Likelihood" + (str(shape) if shape else "")

        return Factor(
            self,
            variable,
            name=name,
            factor_jacobian=self.factor_jacobian,
            plates=variable.plates,
            arg_names=["x"],
        )

    def calc_exact_update(self, x: "AbstractMessage") -> "AbstractMessage":
        return (self,)

    def has_exact_projection(self, x: "AbstractMessage") -> bool:
        return type(self) is type(x)

    @classmethod
    def _reconstruct(
        cls,
        parameters: Tuple[np.ndarray, ...],
        log_norm: float,
        id_,
        *args,
    ):
        return cls(
            *parameters,
            log_norm=log_norm,
            id_=id_,
        )

    def __reduce__(self):
        # serialises TransformedMessage during pickling
        return (
            self._reconstruct,
            (
                self.parameters,
                self.log_norm,
                self.id,
            ),
        )

    def _sample(self, n_samples):
        # Needed for nested TransformedMessage method resolution
        return self.sample(n_samples)

    @classmethod
    def _logpdf_gradient(cls, self, x):
        # Needed for nested TransformedMessage method resolution
        return cls.logpdf_gradient(self, x)

    @classmethod
    def _logpdf_gradient_hessian(cls, self, x):
        # Needed for nested TransformedMessage method resolution
        return cls.logpdf_gradient_hessian(self, x)


def map_dists(
    dists: Dict[str, AbstractMessage],
    values: Dict[str, np.ndarray],
    _call: str = "logpdf",
) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Calls a method (default: logpdf) for each Message in dists
    on the corresponding value in values
    """
    for v in dists.keys() & values.keys():
        dist = dists[v]
        if isinstance(dist, MessageInterface):
            yield v, getattr(dist, _call)(values[v])
