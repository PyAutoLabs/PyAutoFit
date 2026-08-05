from __future__ import annotations
import logging
import warnings
from collections import abc
from enum import Enum
from functools import reduce
from operator import mul
from typing import Iterable, Tuple, TypeVar, Dict, NamedTuple, Optional, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

from autofit.mapper.variable import Variable, VariableData
from autofit.non_linear.result import Result


def try_getitem(value, index, default=None):
    try:
        return value[index]
    except TypeError:
        return default


class LogWarnings(warnings.catch_warnings):
    def __init__(self, *, module=None, messages=None, action=None, logger=logging.warning):
        super().__init__(record=True, module=module)
        self.messages = [] if messages is None else messages
        self.log = []
        self.action = action
        self.logger = logger

    def log_warning(self, warn):
        self.log.append(warn)
        warn_message = f"{warn.filename}:{warn.lineno}: {warn.message}"
        self.messages.append(warn_message)
        self.logger(warn_message)

    def __enter__(self):
        self.log = super().__enter__()
        self._module._showwarnmsg_impl = self.log_warning
        if self.action:
            warnings.simplefilter(self.action)

        return self


def is_variable(v, *args):
    return isinstance(v, Variable)


def is_iterable(arg):

    import six

    return isinstance(arg, abc.Iterable) and not isinstance(
        arg, six.string_types
    )

def is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(obj, '_fields')

def nested_getitem(obj, key):
    """
    Example
    -------
    >>> nested_getitem([1, (2, 3), [3, {'a': 1, 'b': 2}]], (2, 1, 'b'))
    2
    """
    for k in key:
        obj = obj[k]
    return obj 


def nested_get(obj, key, default=None):
    """
    Example
    -------
    >>> nested_get([1, (2, 3), [3, {'a': 1, 'b': 2}]], (2, 1, 'b'))
    2
    >>> nested_get([1, (2, 3), [3, {'a': 1, 'b': 2}]], (2, 1, 'c'), 'default')
    'default'
    """
    try:
        return nested_getitem(obj, key)
    except (KeyError, IndexError):
        return default
    

def nested_set(obj, key, val):
    """
    Example
    -------
    >>> obj = [1, (2, 3), [3, {'a': 1, 'b': 2}]]
    >>> nested_set(obj, (2, 1, 'b'), 3)
    >>> obj
    [1, (2, 3), [3, {'a': 1, 'b': 3}]]

    >>> nested_set(obj, (2, 1,), ()))
    >>> obj
    [1, (2, 3), [3, ()]]
    """
    *parents, k = key 
    for p in parents:
        obj = obj[p]
    obj[k] = val 


def nested_items(*args, key=(), key_func=None):
    """
    Example
    -------
    >>> list(nested_items([1, (2, 3), [3, {'a': 1, 'b': 2}]]))
    [((0,), 1), ((1, 0), 2), ((1, 1), 3), ((2, 0), 3), ((2, 1, 'a'), 1), ((2, 1, 'b'), 2)]

    >>> list(nested_items([1, (2, 3)], [4, (5, 6)]))
    [((0,), 1, 4), ((1, 0), 2, 5), ((1, 1), 3, 6)]
    """
    out, *_ = args
    if isinstance(out, dict):
        for k in sorted(out, key=key_func):
            yield from nested_items(*(out[k] for out in args), key=key + (k,), key_func=key_func)
    elif isinstance(out, (tuple, list)):
        for i, elems in enumerate(zip(*args)):
            yield from nested_items(*elems, key=key + (i,), key_func=key_func)
    else:
        yield (key,) + args


def nested_zip(*args, key_func=None):
    """ Iterates through a potentially nested set of list, tuples and dictionaries, 
    recursively looping through the structure and returning the leaves of the tree

    Example
    -------
    >>> list(nested_zip([1, (2, 3), [3, 2, {1, 2}]]))
    [(1,), (2,), (3,), (3,), (2,), (1,), (2,)]

    >>> list(nested_zip(
    ...     [1, (2, 3), [3, 2, {1, 2}]],
    ...     [1, ('a', 3), [3, 'b', {1, 'c'}]]
    ... ))
    [(1, 1), (2, 'a'), (3, 3), (3, 3), (2, 'b'), (1, 1), (2, 'c')]
    """
    out, *_ = args
    if isinstance(out, dict):
        for k in sorted(out, key=key_func):
            yield from nested_zip(*(out[k] for out in args), key_func=key_func)
    elif is_iterable(out):
        for elems in zip(*args):
            yield from nested_zip(*elems, key_func=key_func)
    else:
        yield args


def nested_iter(args, key_func=None):
    """Iterates through a potentially nested set of list, tuples and dictionaries, 
    recursively looping through the structure and returning the leaves of the tree

    Example
    -------
    >>> list(nested_iter([1, (2, 3), [3, 2, {1, 2}]]))    
    [1, 2, 3, 3, 2, 1, 2]
    """
    for elem, in nested_zip(args, key_func=key_func):
        yield elem


def nested_filter(func, *args, key_func=None):
    """ Iterates through a potentially nested set of list, tuples and dictionaries, 
    recursively looping through the structure and returning the arguments
    that func return true on, 

    Example
    -------
    >>> list(nested_filter(
    ...     lambda x, *args: x==2,
    ...     [1, (2, 3), [3, 2, {1, 2}]]
    ... ))
    [(2,), (2,), (2,)]

    >>> list(nested_filter(
    ...     lambda x, *args: x==2,
    ...     [1, (2, 3), [3, 2, {1, 2}]],
    ...     [1, ('a', 3), [3, 'b', {1, 'c'}]]
    ... ))
    [(2, 'a'), (2, 'b'), (2, 'c')]
    """
    for leaves in nested_zip(*args, key_func=key_func):
        if func(*leaves):
            yield leaves


def nested_map(func, *args, key_func=None):
    """
    Given a potentially nested set of list, tuples and dictionaries, recursively loop through the structure and
    replace any values that appear in the dict to_replace
    can set to replace dictionary keys optionally,

    Example
    -------
    >>> nested_map(lambda x: x*2, [1, (2, 3), [3, 2, {1, 2}, {'a': 'b'}]])
    [2, (4, 6), [6, 4, {2, 4}, {'a': 'bb'}]]

    >>> graph.utils.nested_map(
    ...     lambda x, y: x*y, 
    ...     [1, (2, 3), [3, {'a': 1, 'b': 2}]],
    ...     [1, (2, 3), [3, {'a': 1, 'b': 2}]]
    ... )
    [1, (4, 9), [9, {'a': 1, 'b': 4}]]
    """
    out, *_ = args
    if isinstance(out, dict):
        return type(out)(
            {
                k: nested_map(func, *(arg[k] for arg in args), key_func=key_func)
                for k in sorted(out, key=key_func)
            }
        )
    elif is_namedtuple(out):
        return type(out)(*(nested_map(func, *elems) for elems in zip(*args)))
    elif is_iterable(out):
        return type(out)(nested_map(func, *elems) for elems in zip(*args))

    return func(*args)


def nested_update(out, to_replace):
    """
    Given a potentially nested set of list, tuples and dictionaries, recursively loop through the structure and
    replace any values that appear in the dict to_replace
    can set to replace dictionary keys optionally,

    Does not replace values in place, so can 'mutate' tuples

    Example
    -------
    >>> nested_update([1, (2, 3), [3, 2, {1, 2}]], {2: 'a'})
    [1, ('a', 3), [3, 'a', {1, 'a'}]]

    >>> nested_update([{2: 2}], {2: 'a'})
    [{2: 'a'}]
    """

    def replace(val):
        return to_replace.get(val, val)
    
    return nested_map(replace, out)

from_variabledata = nested_update

def to_variabledata(variables, raw_values) -> VariableData:
    return VariableData(nested_filter(is_variable, variables, raw_values))


class StatusFlag(Enum):
    FAILURE = 0
    SUCCESS = 1
    NO_CHANGE = 2
    BAD_PROJECTION = 3
    # The factor's optimiser *raised* rather than returning a failed status.
    # Distinct from FAILURE, which an optimiser returns routinely and which EP
    # is designed to absorb (e.g. "Line search failed" from the Laplace
    # optimiser). Only EXCEPTION counts toward the consecutive-failure abort in
    # `EPOptimiser`.
    EXCEPTION = 4

    @classmethod
    def get_flag(cls, success, n_iter):
        if success:
            if n_iter > 0:
                return cls.SUCCESS
            else:
                return cls.NO_CHANGE

        return cls.FAILURE


class Status:
    success: bool = True
    messages: Tuple[str, ...] = ()
    updated: bool = True
    flag: StatusFlag = StatusFlag.SUCCESS
    result: Optional[Result] = None

    def __init__(
            self,
            success: bool = True,
            messages: Tuple[str, ...] = (),
            updated: bool = True,
            flag: StatusFlag = StatusFlag.SUCCESS,
            result: Optional[Result] = None,
    ):
        self.success = success
        self.messages = messages
        self.updated = updated
        self.flag = flag
        self.result = result

    def __iter__(self):
        return iter((self.success, self.messages, self.updated, self.flag))

    def __bool__(self):
        return self.success

    def __str__(self):
        if self.success:
            return "Optimisation succeeded"
        return f"Optimisation failed: {self.messages}"

    def __repr__(self):
        return str(self)

    def _asdict(self):
        return dict(
            success=self.success,
            messages=self.messages,
            updated=self.updated,
            flag=self.flag,
        )


class FlattenArrays(dict):
    """
    >>> shapes = FlattenArrays(a=(1, 2), b=(2, 3))
    >>> shapes
    FlattenArrays(a=(1, 2), b=(2, 3))
    >>> shapes.flatten(
        a = np.arange(2).reshape(1, 2),
        b = np.arange(6).reshape(2, 3)**2)
    array([ 0,  1,  0,  1,  4,  9, 16, 25])
    >>> shapes.unflatten(
        [ 0,  1,  0,  1,  4,  9, 16, 25])
    {'a': array([[0, 1]]), 'b': array([[ 0,  1,  4],
        [ 9, 16, 25]])}
    """

    def __init__(self, dict_: Dict[Variable, Tuple[int, ...]]):
        super().__init__()

        self.update(dict_)
        self.splits = np.cumsum([np.prod(s) for s in self.values()], dtype=int)
        self.inds = [
            slice(i0, i1)
            for i0, i1 in
            # np.arange(i0, i1, dtype=int) for i0, i1 in
            zip(np.r_[0, self.splits[:-1]], self.splits)
        ]
        self.sizes = {k: np.prod(s, dtype=int) for k, s in self.items()}
        self.k_inds = dict(zip(self, self.inds))

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray]) -> "FlattenArrays":
        return cls({k: np.shape(arr) for k, arr in arrays.items()})

    def flatten(self, arrays_dict: Dict[Variable, np.ndarray]) -> np.ndarray:
        assert all(np.shape(arrays_dict[k]) == shape for k, shape in self.items())
        return np.concatenate([np.ravel(arrays_dict[k]) for k in self.keys()])

    def extract(self, key, flat, ndim=None):
        if ndim is None:
            ndim = len(flat.shape)

        ind = self.k_inds[key]
        return flat[(ind,) * ndim]

    def unflatten(self, arr: np.ndarray, ndim=None) -> Dict[str, np.ndarray]:
        arr = np.asanyarray(arr)
        if ndim is None:
            ndim = arr.ndim
        arrays = [arr[(ind,) * ndim] for ind in self.inds]
        arr_shapes = [arr.shape[ndim:] for arr in arrays]
        return VariableData({
            k: arr.reshape(shape * ndim + arr_shape)
            if shape or arr_shape
            else arr.item()
            for (k, shape), arr_shape, arr in zip(self.items(), arr_shapes, arrays)
        })

    def flatten2d(self, values: Dict[Variable, np.ndarray]) -> np.ndarray:

        from scipy.linalg import block_diag

        assert all(np.shape(values[k]) == shape * 2 for k, shape in self.items())

        return block_diag(
            *(np.reshape(values[k], (n, n)) for k, n in self.sizes.items())
        )

    unflatten2d = unflatten

    def __repr__(self):
        shapes = ", ".join(map("{0[0]}={0[1]}".format, self.items()))
        return f"{type(self).__name__}({shapes})"

    @property
    def size(self):
        return self.splits[-1]



class OptResult(NamedTuple):

    mode: Dict[Variable, np.ndarray]
    hess_inv: Dict[Variable, np.ndarray]
    log_norm: float
    full_hess_inv: np.ndarray
    result: OptimizeResult
    status: Status = Status()


def gen_subsets(n, x, n_iters=None, rng=None):
    """
    Generates random subsets of length n of the array x, if the elements of
    x are unique then each subset will not contain repeated elements. Each 
    element is guaranteed to reappear after at most 2*len(x) new elements. 

    If `x` is a multi-dimensional array, it is only shuffled along its
first index.

    if x is an integer, generate subsets of ``np.arange(x)``.

    generates n_iters subsets before stopping. If n_iters is None then
    generates random subsets for ever

    rng is an optionally passed random number generator

    Examples
    --------
    >>> list(gen_subsets(3, 5, n_iters=3))
    [array([0, 2, 3]), array([1, 4, 0]), array([2, 3, 4])]
    >>> list(gen_subsets(3, [1,10,5,3], n_iters=3))
    [array([ 5, 10,  1]), array([3, 5, 1]), array([10,  3,  5])]
    """
    rng = rng or np.random.default_rng()
    x_shuffled = rng.permutation(x)
    tot = len(x_shuffled)

    i = 0
    stop = tot - n + 1
    iters = iter(int, 1) if n_iters is None else range(n_iters)
    for j in iters:
        if i < stop:
            yield x_shuffled[i: i + n]
            i += n
        else:
            x_shuffled = np.r_[x_shuffled[i:], rng.permutation(x_shuffled[:i])]
            yield x_shuffled[:n]
            i = n


def gen_dict(dict_gen):
    """
    Examples
    --------
    >>> list(gen_dict({1: gen_subsets(3, 4, 3), 2: gen_subsets(2, 5, 3)}))
    [{1: array([2, 1, 3]), 2: array([2, 0])},
     {1: array([0, 3, 1]), 2: array([3, 1])},
     {1: array([2, 0, 1]), 2: array([4, 2])}]
    """
    keys = tuple(dict_gen.keys())
    for val in zip(*dict_gen.values()):
        yield dict(zip(keys, val))


_M = TypeVar("_M")


def prod(iterable: Iterable[_M], *arg: Tuple[_M]) -> _M:
    """calculates the product of the passed iterable,
    much like sum, if a second argument is passed,
    this is the initial value of the calculation

    Examples
    --------
    >>> prod(range(1, 3))
    2

    >>> prod(range(1, 3), 2.)
    4.
    """
    iterable = list(iterable)
    return reduce(mul, iterable, *arg)


def r2_score(y_true, y_pred, axis=None):
    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)

    mse = np.square(y_true - y_pred).mean(axis=axis)
    var = y_true.var(axis=axis)

    return 1 - mse / var


def propagate_uncertainty(cov: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """Propagates the uncertainty of a covariance matrix given the
    passed Jacobian

    If the variable arrays are multidimensional then will output in
    the shape of the arrays

    see https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    """
    cov = np.asanyarray(cov)

    var_ndim = cov.ndim // 2
    det_ndim = jac.ndim - var_ndim
    det_shape, var_shape = jac.shape[:det_ndim], jac.shape[det_ndim:]
    assert var_shape == cov.shape[:var_ndim] == cov.shape[var_ndim:]

    var_size = np.prod(var_shape, dtype=int)
    det_size = np.prod(det_shape, dtype=int)

    cov2d = cov.reshape((var_size, var_size))
    jac2d = jac.reshape((det_size, var_size))

    det_cov2d = np.linalg.multi_dot((jac2d, cov2d, jac2d.T))
    det_cov = det_cov2d.reshape(det_shape + det_shape)
    return det_cov


def rescale_to_artists(artists, ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    while True:
        r = ax.figure.canvas.get_renderer()
        extents = [
            t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            for t in artists
        ]
        min_extent = np.min([e.min for e in extents], axis=0)
        max_extent = np.max([e.max for e in extents], axis=0)
        min_lim, max_lim = zip(ax.get_xlim(), ax.get_ylim())

        # Sometimes the window doesn't always rescale first time around
        if (min_extent < min_lim).any() or (max_extent > max_lim).any():
            extent = max_extent - min_extent
            max_extent += extent * 0.05
            min_extent -= extent * 0.05
            xlim, ylim = zip(
                np.minimum(min_lim, min_extent), np.maximum(max_lim, max_extent)
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        else:
            break

    return xlim, ylim


# These may no longer be needed?
def add_arrays(*arrays: np.ndarray) -> np.ndarray:
    """Sums over broadcasting multidimensional arrays
    whilst preserving the total sum

    a = np.arange(10).reshape(1, 2, 1, 5)
    b = np.arange(8).reshape(2, 2, 2, 1)

    >>> add_arrays(a, b).sum()
    73.0
    >>> add_arrays(a, b).shape
    (2, 2, 2, 5)
    >>> a.sum() + b.sum()
    73
    """
    b = np.broadcast(*arrays)
    return sum(a * np.size(a) / b.size for a in arrays)


Axis = Optional[Union[bool, int, Tuple[int, ...]]]


def aggregate(array: np.ndarray, axis: Axis = None, **kwargs) -> np.ndarray:
    """
    aggregates the values of array

    if axis is False then aggregate returns the unmodified array

    otherwise aggrate returns np.sum(array, axis=axis, **kwargs)
    """
    if axis is False:
        return array

    return np.sum(array, axis=axis, **kwargs)


def diag(array: np.ndarray, *ds: Tuple[int, ...]) -> np.ndarray:
    array = np.asanyarray(array)
    d1 = array.shape
    if ds:
        ds = (d1,) + ds
    else:
        ds = (d1, d1)

    out = np.zeros(sum(ds, ()))
    diag_inds = tuple(map(np.ravel, (i for d in ds for i in np.indices(d))))
    out[diag_inds] = array.ravel()
    return out
