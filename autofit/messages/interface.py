from abc import ABC, abstractmethod
from typing import Tuple, Iterator
from typing import Union

import numpy as np

from autonerves import cached_property


class MessageInterface(ABC):
    """
    Common base class for base and transformed messages
    """

    log_base_measure: float
    log_norm: float
    id: int

    @property
    @abstractmethod
    def broadcast(self):
        pass

    @property
    def shape(self) -> Tuple[int, ...]:

        # JAX behaviour
        if isinstance(self.broadcast, list):
            return ()

        return self.broadcast.shape

    @property
    def size(self) -> int:
        return self.broadcast.size

    @property
    def ndim(self) -> int:
        return self.broadcast.ndim

    def __eq__(self, other):
        return self.id == other.id

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def logpdf(self, x: Union[np.ndarray, float], xp=np) -> np.ndarray:

        eta = self._broadcast_natural_parameters(x, xp=xp)
        t = self.to_canonical_form(x, xp=xp)
        log_base = self.calc_log_base_measure(x, xp=xp)

        return self.natural_logpdf(eta, t, log_base, self.log_partition(xp=xp), xp=xp)

    def _broadcast_natural_parameters(self, x, xp=np):
        shape = xp.shape(x)
        if shape == self.shape:
            return self.natural_parameters(xp=xp)
        elif shape[1:] == self.shape:
            return self.natural_parameters(xp=xp)[:, None, ...]
        else:
            raise ValueError(
                f"shape of passed value {shape} does not "
                f"match message shape {self.shape}"
            )

    @abstractmethod
    def natural_parameters(self, xp=np) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def to_canonical_form(x: Union[np.ndarray, float], xp=np) -> np.ndarray:
        pass

    @classmethod
    def calc_log_base_measure(cls, x, xp=np):
        return cls.log_base_measure

    @abstractmethod
    def log_partition(self, xp=np) -> np.ndarray:
        pass

    @classmethod
    def natural_logpdf(cls, eta, t, log_base, log_partition, xp=np):
        eta_t = xp.multiply(eta, t).sum(0)
        return xp.nan_to_num(log_base + eta_t - log_partition, nan=-xp.inf)

    def numerical_logpdf_gradient(
        self, x: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        shape = np.shape(x)
        if shape:
            x0 = np.array(x, dtype=np.float64)
            logl0 = self.logpdf(x0)
            if self.multivariate:
                grad_logl = np.empty(logl0.shape + x0.shape)
                sl = tuple(slice(None) for _ in range(logl0.ndim))
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        logl = self.logpdf(x0)
                        grad_logl[sl + it.multi_index] = (logl - logl0) / eps
                        xv -= eps
            else:
                l0 = logl0.sum()
                grad_logl = np.empty_like(x0)
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        logl = self.logpdf(x0).sum()  # type: ignore
                        grad_logl[it.multi_index] = (logl - l0) / eps
                        xv -= eps
        else:
            logl0 = self.logpdf(x)
            grad_logl = (self.logpdf(x + eps) - logl0) / eps

        return logl0, grad_logl

    logpdf_gradient = numerical_logpdf_gradient

    @property
    @abstractmethod
    def multivariate(self):
        pass

    def numerical_logpdf_gradient_hessian(
        self, x: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = np.shape(x)
        if shape:
            x0 = np.array(x, dtype=np.float64)
            if self.multivariate:
                logl0, gradl0 = self.numerical_logpdf_gradient(x0)
                hess_logl = np.empty(gradl0.shape + x0.shape)
                sl = tuple(slice(None) for _ in range(gradl0.ndim))
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        _, gradl = self.numerical_logpdf_gradient(x0)
                        hess_logl[sl + it.multi_index] = (gradl - gradl0) / eps
                        xv -= eps
            else:
                logl0 = self.logpdf(x0)
                l0 = logl0.sum()
                grad_logl = np.empty_like(x0)
                hess_logl = np.empty_like(x0)
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        l1 = self.logpdf(x0).sum()
                        xv -= 2 * eps
                        l2 = self.logpdf(x0).sum()
                        g1 = (l1 - l0) / eps
                        g2 = (l0 - l2) / eps
                        grad_logl[it.multi_index] = g1
                        hess_logl[it.multi_index] = (g1 - g2) / eps
                        xv += eps

                gradl0 = grad_logl
        else:
            logl0 = self.logpdf(x)
            logl1 = self.logpdf(x + eps)
            logl2 = self.logpdf(x - eps)
            gradl0 = (logl1 - logl0) / eps
            gradl1 = (logl0 - logl2) / eps
            hess_logl = (gradl0 - gradl1) / eps

        return logl0, gradl0, hess_logl

    logpdf_gradient_hessian = numerical_logpdf_gradient_hessian

    @staticmethod
    def _iter_dists(dists) -> Iterator[Union["MessageInterface", float]]:
        for elem in dists:
            if isinstance(elem, MessageInterface):
                yield elem
            elif np.isscalar(elem):
                yield elem
            else:
                for dist in elem:
                    yield dist

    @abstractmethod
    def check_support(self) -> np.ndarray:
        pass

    def check_finite(self) -> np.ndarray:
        return np.isfinite(self.natural_parameters()).all(0)

    def check_valid(self) -> np.ndarray:
        return self.check_finite() & self.check_support()

    @cached_property
    def is_valid(self) -> Union[np.ndarray, np.bool_]:
        return np.all(self.check_finite()) and np.all(self.check_support())

    @abstractmethod
    def update_invalid(self, other: "MessageInterface") -> "MessageInterface":
        pass

    def sum_natural_parameters(self, *dists: "MessageInterface") -> "MessageInterface":
        """return the unnormalised result of multiplying the pdf
        of this distribution with another distribution of the same
        type
        """
        new_params = sum(
            (
                dist.natural_parameters()
                for dist in self._iter_dists(dists)
                if isinstance(dist, MessageInterface)
            ),
            self.natural_parameters(),
        )
        return self.from_natural_parameters(
            new_params,
            id_=self.id,
        )

    def sub_natural_parameters(self, other: "MessageInterface") -> "MessageInterface":
        """return the unnormalised result of dividing the pdf
        of this distribution with another distribution of the same
        type"""
        log_norm = self.log_norm - other.log_norm
        new_params = self.natural_parameters() - other.natural_parameters()
        return self.from_natural_parameters(
            new_params,
            log_norm=log_norm,
            id_=self.id,
        )

    @abstractmethod
    def from_natural_parameters(self, new_params, **kwargs):
        pass

    _multiply = sum_natural_parameters
    _divide = sub_natural_parameters
