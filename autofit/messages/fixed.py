from typing import Optional, Tuple

import numpy as np

from autoconf import cached_property
from autofit.messages.abstract import AbstractMessage


class FixedMessage(AbstractMessage):
    log_base_measure = 0

    def __init__(
            self,
            value: np.ndarray,
            log_norm: np.ndarray = 0.,
            id_=None
    ):
        self.value = value
        super().__init__(
            value,
            log_norm=log_norm,
            id_=id_
        )

    def value_for(self, unit: float) -> float:
        raise NotImplemented()

    def natural_parameters(self, xp=np) -> Tuple[np.ndarray, ...]:
        return self.parameters

    @staticmethod
    def invert_natural_parameters(natural_parameters: np.ndarray
                                  ) -> Tuple[np.ndarray]:
        return natural_parameters,

    @staticmethod
    def to_canonical_form(x: np.ndarray, xp=np) -> np.ndarray:
        return x

    def log_partition(self, xp=np) -> np.ndarray:
        return 0.

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats: np.ndarray
                                     ) -> np.ndarray:
        return suff_stats

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Rely on array broadcasting to get fixed values to
        calculate correctly
        """
        if n_samples is None:
            return self.value
        return np.array([self.value])

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        # A fixed message contributes zero log-density everywhere. Return a
        # fresh zero array each call: the previous class-level ``logpdf_cache``
        # was an unbounded dict keyed on shape that also handed back an aliased
        # mutable array, so mutating one result silently corrupted later calls.
        return np.zeros_like(x)

    @cached_property
    def mean(self) -> np.ndarray:
        return self.value

    @cached_property
    def variance(self) -> np.ndarray:
        return np.zeros_like(self.mean)

    def _no_op(self, *other, **kwargs) -> 'FixedMessage':
        """
        'no-op' operation

        In many operations fixed messages should just
        return themselves
        """
        return self

    project = _no_op
    from_mode = _no_op
    __pow__ = _no_op
    __mul__ = _no_op
    __div__ = _no_op
    default = _no_op
    _multiply = _no_op
    _divide = _no_op
    sum_natural_parameters = _no_op
    sub_natural_parameters = _no_op

    def kl(self, dist: "FixedMessage") -> float:
        return 0.
