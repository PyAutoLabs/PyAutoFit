from itertools import chain, count
from typing import Optional, Tuple, Dict, Set, Union, List, TYPE_CHECKING
import operator
from abc import ABC, abstractmethod
from functools import wraps, reduce
from math import sqrt

import numpy as np
from autonerves import cached_property

from autofit.mapper.model_object import ModelObject

if TYPE_CHECKING:
    from autofit.mapper.operator import LinearOperator
    from autofit.mapper.variable_operator import VariableFullOperator, VariableOperator


class Plate:
    _ids = count()

    def __init__(self, name: Optional[str] = None):
        """
        Represents a dimension, such as number of observations, features or dimensions

        Parameters
        ----------
        name
            The name of this dimension
        """
        self.id = next(self._ids)
        self.name = name or f"plate_{self.id}"

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"

    def __eq__(self, other):
        return isinstance(other, Plate) and self.id == other.id

    def __hash__(self):
        return self.id

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def make_index_seq(
        self,
        plates_index: Dict["Plate", Union[List[int], range, slice]],
        plate_sizes: Dict["Plate", int],
    ) -> Union[List[int], range]:
        seq = plates_index.get(self, range(plate_sizes[self]))
        if isinstance(seq, slice):
            seq = range(plate_sizes[self])[seq]

        return seq


def plates(*vals):
    """Helper function for making multiple plate objects

    Example
    -------
    x_, a_, b_, y_, z_ = plates("x, a, b", "y, z")
    """
    for val in vals:
        for v in val.split(","):
            yield Plate(v.strip())


class Variable(ModelObject):
    # __slots__ = ("name", "plates")

    def __init__(self, name: str = None, *plates: Plate, id_=None):
        """
        Represents a variable in the problem. This may be fixed data or some coefficient
        that we are optimising for.

        Parameters
        ----------
        name
            The name of this variable
        plates
            Representation of the dimensions of this variable
        """
        self.plates = plates
        super().__init__(id_=id_)
        self._name = name

    @property
    def name(self):
        return self._name or f"{self.__class__.__name__.lower()}_{self.id}"

    def __repr__(self):
        args = ", ".join(chain([self.name], map(repr, self.plates)))
        return f"{self.__class__.__name__}({args})"

    def __hash__(self):
        return self.id

    def __len__(self):
        return len(self.plates)

    def __str__(self):
        return self.name

    @property
    def ndim(self) -> int:
        """
        How many dimensions does this variable have?
        """
        return len(self.plates)

    def make_indexes(
        self,
        plates_index: Dict["Plate", Union[List[int], range, slice]],
        plate_sizes: Dict["Plate", int],
    ) -> Tuple[np.ndarray, ...]:
        if any(p in plates_index for p in self.plates):
            return np.ix_(
                *(p.make_index_seq(plates_index, plate_sizes) for p in self.plates)
            )
        return ()


def variables(*vals):
    """Helper function for making multiple variable objects

    Example
    -------
    x_, a_, b_, y_, z_ = variables("x, a, b", "y, z")
    """
    for val in vals:
        for v in val.split(","):
            yield Variable(v.strip())


# This allows us to treat the class FactorValue as a variable
# that allows us to keep track of the FactorValue vs deterministic
# values when calculating gradients and jacobians
class VariableMetaClass(type, Variable):
    def __new__(cls, clsname, bases, attrs):
        newcls = super().__new__(cls, clsname, bases, attrs)
        Variable.__init__(newcls, clsname)
        return newcls


class FactorValue(np.ndarray, metaclass=VariableMetaClass):
    def __new__(cls, input_array, deterministic_values=None):
        obj = np.asarray(input_array).view(cls)
        obj.deterministic_values = deterministic_values or {}
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.deterministic_values = getattr(obj, "deterministic_values", None)

    @property
    def log_value(self) -> np.ndarray:
        if self.shape:
            return self.base
        else:
            return self.item()

    def __getitem__(self, index) -> np.ndarray:
        if index is type(self):
            return self
        elif isinstance(index, Variable):
            return self.deterministic_values[index]
        else:
            return super().__getitem__(index)

    def keys(self):
        return self.deterministic_values.keys()

    def values(self):
        return self.deterministic_values.values()

    def items(self):
        return self.deterministic_values.items()

    deterministic_variables = property(keys)

    def __repr__(self):
        r = np.ndarray.__repr__(self)
        return r[:-1] + ", " + repr(self.deterministic_values) + ")"

    def to_dict(self):
        return VariableData({FactorValue: self.base, **self.deterministic_values})


def broadcast_plates(
    value: np.ndarray,
    in_plates: Tuple[Plate, ...],
    out_plates: Tuple[Plate, ...],
    reducer: np.ufunc = np.sum,
) -> np.ndarray:
    """
    Extract the indices of a collection of plates then match
    the shape of the data to that shape.

    Parameters
    ----------
    value
        A value to broadcast
    in_plates
        Plates representing the dimensions of the values
    out_plates
        Plates representing the output dimensions
    reducer
        function to reduce excess plates over, default np.sum
        must take axis as keyword argument


    Returns
    -------
    The value reshaped to match the plates
    """
    n_in = len(in_plates)
    n_out = len(out_plates)
    shift = np.ndim(value) - n_in
    if not (0 <= shift <= 1):
        raise ValueError("dimensions of value incompatible with passed plates")

    in_axes = list(range(shift, n_in + shift))
    out_axes = []
    k = n_out + shift

    for plate in in_plates:
        try:
            out_axes.append(out_plates.index(plate) + shift)
        except ValueError:
            out_axes.append(k)
            k += 1

    moved_value = np.moveaxis(
        np.expand_dims(value, tuple(range(n_in + shift, k))),
        in_axes,
        out_axes,
    )
    return reducer(moved_value, axis=tuple(range(n_out + shift, k)))


def _get_variable_data_class(data):
    # So that these methods work with standard python dictionaries
    return type(data) if isinstance(data, VariableData) else VariableData


def _unary_op(op):
    @wraps(op)
    def __op__(self):
        cls = _get_variable_data_class(self)
        return cls({k: op(val) for k, val in self.items()})

    return __op__


def _binary_op(op, ravel=False):
    if ravel:

        @wraps(op)
        def __op__(self, other):
            cls = _get_variable_data_class(self)
            if isinstance(other, FactorValue):
                other = other.to_dict()

            if isinstance(other, dict):
                return cls(
                    {
                        k: op(np.ravel(self[k]), np.ravel(other[k]))
                        for k in self.keys() & other.keys()
                    }
                )
            elif isinstance(other, VariableLinearOperator):
                return op(dict(self), other)
            else:
                return cls({k: op(val, other) for k, val in self.items()})

    else:

        @wraps(op)
        def __op__(self, other):
            cls = _get_variable_data_class(self)
            if isinstance(other, FactorValue):
                other = other.to_dict()

            if isinstance(other, dict):
                return cls(
                    {k: op(self[k], other[k]) for k in self.keys() & other.keys()}
                )
            elif isinstance(other, VariableLinearOperator):
                return op(dict(self), other)
            else:
                return cls({k: op(val, other) for k, val in self.items()})

    return __op__


def rmul(x, y):
    return y * x


def rtruediv(x, y):
    return y / x


class VariableData(Dict[Variable, np.ndarray]):
    var_norm = _unary_op(np.linalg.norm)
    var_det = _unary_op(np.linalg.det)
    var_max = _unary_op(np.max)
    var_min = _unary_op(np.min)
    var_sum = _unary_op(np.sum)
    var_all = _unary_op(np.all)
    var_any = _unary_op(np.any)
    var_prod = _unary_op(np.prod)
    var_isfinite = _unary_op(np.isfinite)

    __abs__ = _unary_op(operator.abs)
    __pos__ = _unary_op(operator.pos)
    __neg__ = _unary_op(operator.neg)
    __lt__ = _binary_op(operator.lt)
    __le__ = _binary_op(operator.le)
    __eq__ = _binary_op(operator.eq)
    __ne__ = _binary_op(operator.ne)
    __gt__ = _binary_op(operator.gt)
    __ge__ = _binary_op(operator.ge)
    __and__ = _binary_op(operator.and_)
    __and__ = _binary_op(operator.or_)

    __add__ = _binary_op(operator.add)
    __radd__ = _binary_op(operator.add)
    __sub__ = _binary_op(operator.sub)
    __rsub__ = _binary_op(operator.sub)
    __mul__ = _binary_op(operator.mul)
    __rmul__ = _binary_op(rmul)
    __truediv__ = _binary_op(operator.truediv)
    __rtruediv__ = _binary_op(rtruediv)
    __pow__ = _binary_op(operator.pow)

    abs = __abs__
    neg = __neg__
    sub = __sub__
    add = __add__
    mul = __mul__
    div = __truediv__
    var_dot = _binary_op(np.dot, ravel=True)

    @property
    def T(self):
        return type(self)({k: val.T for k, val in self.items()})

    @property
    def shapes(self):
        return {k: np.shape(val) for k, val in self.items()}

    @property
    def sizes(self):
        return {k: np.size(val) for k, val in self.items()}

    @property
    def size(self):
        return sum(np.size(val) for val in self.values())

    def ravel(self):
        cls = _get_variable_data_class(self)
        return cls({k: np.ravel(val) for k, val in self.items()})

    def map(self, func, *args, **kwargs):
        cls = _get_variable_data_class(self)
        return cls(
            (k, func(val, *(arg[k] for arg in args), **kwargs))
            for k, val in self.items()
        )

    def reduce(self, func):
        return reduce(func, self.values())

    def mapreduce(self, func, op):
        return VariableData.map(self, func).reduce(op)

    def subset(self, variables):
        cls = _get_variable_data_class(self)
        return cls((v, self[v]) for v in variables if v in self)

    def sum(self) -> float:
        return sum(VariableData.var_sum(self).values())

    def prod(self) -> float:
        return VariableData.reduce(VariableData.var_prod(self), operator.mul)

    def all(self) -> bool:
        return all(VariableData.var_all(self).values())

    def any(self) -> bool:
        return any(VariableData.var_all(self).values())

    def det(self) -> float:
        return VariableData.var_det(self).reduce(operator.mul)

    def log_det(self) -> float:
        return VariableData.mapreduce(self, np.linalg.logdet, operator.add)

    def max(self) -> float:
        return max(VariableData.var_max(self).values())

    def min(self) -> float:
        return min(VariableData.var_min(self).values())

    def dot(self, other) -> float:
        return VariableData.var_dot(self, other).sum()

    def norm(self) -> float:
        return sqrt(VariableData.dot(self, self))

    def vecnorm(self, ord: Optional[float] = None) -> float:
        if ord:
            absval = VariableData.abs(self)
            if ord == np.inf:
                return absval.max()
            elif ord == -np.inf:
                return absval.min()
            else:
                return (absval**ord).sum() ** (1.0 / ord)
        else:
            return VariableData.norm(self)

    def __repr__(self):
        name = type(self).__name__
        data_repr = dict.__repr__(self)
        return f"{name}({data_repr})"

    def merge(self, other):
        return VariableData({**self, **other})

    def plate_sizes(self):
        sizes = {}
        for v, val in self.items():
            shape = np.shape(val)
            assert len(shape) == len(
                v.plates
            ), f"shape must match the number of plates of {v}"
            for p, s in zip(v.plates, shape):
                assert (
                    sizes.setdefault(p, s) == s
                ), f"plate sizes must be consistent, {sizes[p]} != {s}"
        return sizes

    def full_like(self, fill_value, **kwargs):
        return type(self)(
            {v: np.full_like(val, fill_value, **kwargs) for v, val in self.items()}
        )

    def zeros_like(self, **kwargs):
        return self.full_like(0.0)


class VariableLinearOperator(ABC):
    """Implements the functionality of a linear operator acting
    on a dictionary of values indexed by `Variable` objects
    """

    @abstractmethod
    def __mul__(self, x: VariableData) -> VariableData:
        pass

    @abstractmethod
    def __rtruediv__(self, x: VariableData) -> VariableData:
        pass

    @abstractmethod
    def __rmul__(self, x: VariableData) -> VariableData:
        pass

    @abstractmethod
    def ldiv(self, x: VariableData) -> VariableData:
        pass

    @property
    @abstractmethod
    def variables(self) -> VariableData:
        pass

    def __getitem__(self, variable) -> "LinearOperator":
        raise NotImplementedError()

    def get(self, variable, default=None):
        try:
            return self[variable]
        except KeyError:
            return default

    def __contains__(self, variable):
        return variable in self.variables

    def dot(self, x):
        return self * x

    __matmul__ = dot

    def inv(self) -> "InverseVariableOperator":
        return InverseVariableOperator(self)

    def quad(self, M: VariableData) -> VariableData:
        return (M * self).T * self

    def invquad(self, M: VariableData) -> VariableData:
        return (M / self).T / self

    @abstractmethod
    def update(self, *args: Tuple[VariableData, VariableData]):
        pass

    def lowrankupdate(self, *values: VariableData):
        return self.update(*((value, value) for value in values()))

    def lowrankdowndate(self, *values: VariableData):
        return self.update(*((value, VariableData.neg(value)) for value in values()))

    def blocks(self):
        return self.to_block().blocks()


class InverseVariableOperator(VariableLinearOperator):
    def __init__(self, op):
        self.operator = op

    def __mul__(self, x: VariableData) -> VariableData:
        return self.operator.ldiv(x)

    def __rtruediv__(self, x: VariableData) -> VariableData:
        return x * self.operator

    def __rmul__(self, x: VariableData) -> VariableData:
        return x / self.operator

    def ldiv(self, x: VariableData) -> VariableData:
        return self * x

    def quad(self, M: VariableData) -> VariableData:
        return self.operator.invquad(M)

    def invquad(self, M: VariableData) -> VariableData:
        return self.operator.quad(M)

    def inv(self) -> VariableLinearOperator:
        return self.operator

    @property
    def variables(self) -> Set[Variable]:
        return self.operator.variables

    @property
    def is_diagonal(self):
        return self.operator.is_diagonal

    @cached_property
    def log_det(self):
        return -self.operator.log_det

    def update(self, *args: Tuple[VariableData, VariableData]):
        # apply Sherman-Morrison formulat
        A = self.operator
        for u, v in args:
            A1u = A * u
            A1v = v * A
            vTA1u = -A1u.dot(v)
            A = A.update(A1u, A1v * vTA1u)

        return type(self)(A)

    def diagonalupdate(self, d: VariableData):
        A = self.operator.diagonalupdate(d**-1)
        return type(self)(A)

    def to_full(self) -> "VariableFullOperator":
        full_op = self.operator.to_full()
        M = np.linalg.inv(full_op.operator.to_dense())
        return full_op.from_dense(M, full_op.param_shapes)

    def diagonal(self) -> VariableData:
        full_op = self.to_full()
        diag = full_op.operator.to_dense().diagonal()
        return full_op.param_shapes.unflatten(diag)

    def to_block(self) -> "VariableOperator":
        return self.to_full().to_block()

    def __getitem__(self, variable):
        return self.to_full()[variable]
