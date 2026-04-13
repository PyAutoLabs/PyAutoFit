from abc import ABC, abstractmethod
from functools import wraps
from typing import Tuple, List

import numpy as np


from autoconf import cached_property


class LinearOperator(ABC):
    r"""Implements the functionality of a linear operator.

    All linear operators can be expressed as a tensor/matrix
    However for some it there may be more efficient representations

    e.g. The Diagonal Matrix (see `DiagonalMatrix`),
    Sherman-Morrison Matrix (see `ShermanMorrison`)
    or the vector-Jacobian product of a function see jax.jvp [0]

    The class also has the attributes, lshape, rshape, lsize, rsize to allow multidimensional tensors to be used,
    see `ShermanMorrison`, `MultiVecOuterProduct`, or
    `autofit.messages.transform.MultinomialLogitTransform`
    for examples of this use case

    If `M` is the dense matrix represenation of the LinearOperator then the
    actions of the methods can be represented by the appropriate matrix
    operation,

    Methods
    -------
    __mul__(x):
        M.dot(x)
        M·x

    __rmul__(x):
        x.dot(M)
        x·M

    __rtruediv__(x):
        x.dot(inv(M))
        x / M

    ldiv(x):
        inv(M).dot(x)
        M \ x

    log_det():
        log(det(M))
        log |M|

    quad(x):
        x.T.dot(M).dot(x)
        xᵀ·M·x

    invquad(x):
        x.T.dot(inv(M)).dot(x)
        xᵀ·M⁻¹·x

    [0] https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-product
    """

    _ldim = 1
    is_diagonal = False

    @abstractmethod
    def __mul__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __rtruediv__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __rmul__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def ldiv(self, x: np.ndarray) -> np.ndarray:
        pass

    def inv(self):
        return InverseOperator(self)

    def from_dense(self, M, shape=None, ldim=None):
        return self.operator.from_dense(M, shape, ldim)

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    def __len__(self) -> int:
        return self.lsize

    @property
    def lsize(self) -> int:
        return np.prod(self.lshape, dtype=int)

    @property
    def rsize(self) -> int:
        return np.prod(self.rshape, dtype=int)

    @property
    def size(self) -> int:
        return np.prod(self.shape, dtype=int)

    @property
    def ldim(self):
        return self._ldim

    @property
    def rdim(self):
        return self.ndim - self.ldim

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def lshape(self):
        return self.shape[: self.ldim]

    @property
    def rshape(self):
        return self.shape[self.ldim :]

    @cached_property
    @abstractmethod
    def log_det(self):
        pass

    def quad(self, M: np.ndarray) -> np.ndarray:
        return (M * self).T * self

    def invquad(self, M: np.ndarray) -> np.ndarray:
        return (M / self).T / self

    def to_operator(self):
        return self

    def transform_bounds(self, bounds: List[Tuple]) -> List[Tuple]:
        """
        Convenience method for transforming the bounds of an
        operation by this matrix.

        Parameters
        ----------
        bounds
            A list of tuples, each describing the lower and upper
            bound for a given dimension.

            There should be N bounds corresponding to N dimensions
            for this NxN matrix.

        Returns
        -------
        The bounds transformed according to this transformation.
        """
        lower, upper = zip(*bounds)
        lower = self * lower
        upper = self * upper
        return list(zip(lower, upper))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in (np.multiply, np.matmul):
            return self.__rmul__(inputs[0])
        elif ufunc is np.divide:
            return self.__rtruediv__(inputs[0])
        else:
            return NotImplemented

    def matrixabs(self):
        raise NotADirectoryError()


class InverseOperator(LinearOperator):
    def __init__(self, operator):
        self.operator = operator

    def __mul__(self, x: np.ndarray) -> np.ndarray:
        return self.operator.ldiv(x)

    def __rtruediv__(self, x: np.ndarray) -> np.ndarray:
        return x * self.operator

    def __rmul__(self, x: np.ndarray) -> np.ndarray:
        return x / self.operator

    def ldiv(self, x: np.ndarray) -> np.ndarray:
        return self * x

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.operator.shape

    def __len__(self) -> int:
        return self.operator.lsize

    @property
    def lsize(self) -> int:
        return np.prod(self.lshape, dtype=int)

    @property
    def rsize(self) -> int:
        return np.prod(self.operator.rshape, dtype=int)

    @cached_property
    def log_det(self):
        return -self.operator.log_det

    def quad(self, M: np.ndarray) -> np.ndarray:
        return self.operator.invquad(M)

    def invquad(self, M: np.ndarray) -> np.ndarray:
        return self.operator.quad(M)

    def to_dense(self):
        M = self.operator.to_dense()
        M2 = np.reshape(M, (self.lsize, self.rsize))
        return np.linalg.inv(M2).reshape(self.operator.shape)

    def to_operator(self):
        return self.from_dense(self.to_dense(), self.shape, self.ldim)

    def diagonal(self):
        return self.to_operator().diagonal()

    def inv(self):
        return self.operator

    @property
    def is_diagonal(self):
        return self.operator.is_diagonal


class IdentityOperator(LinearOperator):
    is_diagonal = True

    def __init__(self):
        pass

    def _identity(self, values: np.ndarray) -> np.ndarray:
        return values

    __mul__ = _identity
    __rtruediv__ = _identity
    __rmul__ = _identity
    ldiv = _identity
    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    quad = _identity
    invquad = _identity

    @cached_property
    def log_det(self):
        return 0.0

    @property
    def shape(self):
        return ()

    def __len__(self):
        return 0


# TODO refactor these for non-square transformations


def _wrap_leftop(method):
    @wraps(method)
    def leftmethod(self, x: np.ndarray) -> np.ndarray:
        x = np.asanyarray(x)
        outshape = self.lshape + x.shape[self.rdim :] if self.ndim else x.shape
        return method(self, x.reshape(self.rsize, -1)).reshape(outshape)

    return leftmethod


def _wrap_rightop(method):
    @wraps(method)
    def rightmethod(self, x: np.ndarray) -> np.ndarray:
        x = np.asanyarray(x)
        outshape = x.shape[: -self.ldim] + self.rshape if self.ndim else x.shape
        return method(self, x.reshape(-1, self.lsize)).reshape(outshape)

    return rightmethod


class MatrixOperator(LinearOperator):
    def __init__(self, M: np.ndarray, shape=None, ldim=None):
        self._shape = shape or np.shape(M)
        self._ldim = len(self.shape) // 2 if ldim is None else ldim
        self._M = np.asanyarray(M).reshape(self.shape)
        self._M2D = self._M.reshape(self.lsize, self.rsize)

    def __getitem__(self, index):
        M = self.operator.to_dense()[index]
        return self.from_dense(M, ldim=self.ldim)

    def reshape(self, shape):
        return self.from_dense(self.operator.to_dense().reshape(shape), ldim=self.ldim)

    @classmethod
    def from_dense(
        cls, M: np.ndarray, shape: Tuple[int, ...] = None, ldim: int = None
    ) -> "MatrixOperator":
        return cls(M, shape, ldim)

    @classmethod
    def from_operator(cls, operator: "MatrixOperator") -> "MatrixOperator":
        return cls.from_dense(operator.to_dense(), operator.shape, operator.ldim)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @cached_property
    def log_det(self):
        return np.linalg.slogdet(self._M2D)[1]

    @_wrap_leftop
    def __mul__(self, x: np.ndarray) -> np.ndarray:
        return self._M2D.dot(x)

    @_wrap_rightop
    def __rtruediv__(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self._M2D.T, x.T)

    @_wrap_rightop
    def __rmul__(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self._M2D)

    @_wrap_leftop
    def ldiv(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self._M2D, x)

    def to_dense(self):
        return self._M.copy()

    def diagonal(self) -> np.ndarray:
        return (
            self.to_dense()
            .reshape(self.rsize, self.lsize)
            .diagonal()
            .reshape(self.rshape)
        )

    def to_diagonal(self):
        return DiagonalMatrix(self.diagonal())

    def __add__(self, other: "MatrixOperator") -> "MatrixOperator":
        return type(self).from_dense(
            self.to_dense() + other.to_dense(), self.shape, self.ldim
        )

    def __sub__(self, other: "MatrixOperator") -> "MatrixOperator":
        return type(self).from_dense(
            self.to_dense() - other.to_dense(), self.shape, self.ldim
        )

    def update(self, *args):
        M = self.to_dense().reshape(self.lsize, self.rsize)
        for (u, v) in args:
            M += u.ravel()[:, None] * v.ravel()[None, :]

        return type(self).from_dense(M, self.shape, self.ldim)

    def lowrankupdate(self, *args):
        return self.update(*((u, u) for u in args))

    def lowrankdowndate(self, *args):
        return self.update(*((u, -u) for u in args))

    def diagonalupdate(self, d):
        M = self._M.copy()
        # set diagonal
        M.flat[:: self.lsize + 1] += np.asanyarray(d).ravel()
        return type(self).from_dense(M, self.shape, self.ldim)


def _mul_triangular(
    c, b, trans=False, lower=True, overwrite_b=False, check_finite=True
):
    """wrapper for BLAS function trmv to perform triangular matrix
    multiplications


    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        vector/matrix being multiplied
    lower, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans, optional
        type of multiplication,

        ========  =========
        trans     system
        ========  =========
        False     a b
        True      a^T b
    overwrite_b, optional
        Allow overwriting data in `b` (may enhance performance)
        not fully tested
    check_finite, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    """
    from scipy._lib._util import _asarray_validated
    from scipy.linalg import (
        get_blas_funcs,
    )

    a1 = _asarray_validated(c, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)

    n = c.shape[1]
    if c.shape[0] != n:
        raise ValueError("Triangular matrix passed must be square")
    if b.shape[0] != n:
        raise ValueError(
            f"shapes {c.shape} and {b.shape} not aligned: "
            f"{n} (dim 1) != {b.shape[0]} (dim 0)"
        )

    (trmv,) = get_blas_funcs(("trmv",), (a1, b1))
    if a1.flags.f_contiguous:

        def _trmv(a1, b1, overwrite_x):
            return trmv(a1, b1, lower=lower, trans=trans, overwrite_x=overwrite_x)

    else:
        # transposed system is solved since trmv expects Fortran ordering
        def _trmv(a1, b1, overwrite_x=overwrite_b):
            return trmv(
                a1.T, b1, lower=not lower, trans=not trans, overwrite_x=overwrite_x
            )

    if b1.ndim == 1:
        return _trmv(a1, b1, overwrite_b)
    elif b1.ndim == 2:
        # trmv only works for vector multiplications
        # set Fortran order so memory contiguous
        b2 = np.array(b1, order="F")
        for i in range(b2.shape[1]):
            # overwrite results
            _trmv(a1, b2[:, i], True)

        if overwrite_b:
            b1[:] = b2
            return b1
        else:
            return b2
    else:
        raise ValueError("b must have 1 or 2 dimensions, has {b.ndim}")


class QROperator(MatrixOperator):
    """This stores the matrix in the QR decomposition form"""

    def __init__(self, Q, R, shape=None, ldim=None):
        self.Q = Q
        self.R = R
        self._shape = shape or self.Q.shape
        self._ldim = ldim or len(self._shape) // 2

    @cached_property
    def M(self) -> np.ndarray:
        return self.Q.dot(self.R)

    @classmethod
    def from_dense(cls, hess, shape=None, ldim=None):

        from scipy.linalg import qr

        shape = np.shape(hess)
        ldim = ldim or np.ndim(hess) // 2
        length = np.prod(shape[:ldim])
        M = np.reshape(hess, (length, length))
        return cls(*qr(M), shape=shape, ldim=ldim)

    @_wrap_leftop
    def __mul__(self, x):
        return self.M.dot(x)

    @_wrap_rightop
    def __rmul__(self, x):
        return x @ self.M

    @_wrap_rightop
    def __rtruediv__(self, x):
        from scipy.linalg import (
            solve_triangular,
        )
        return solve_triangular(self.R.T, x.T @ self.Q.T, lower=True).T

    @_wrap_leftop
    def ldiv(self, x):
        from scipy.linalg import (
            solve_triangular,
        )
        return self.Q.T @ solve_triangular(self.R, x, lower=False)

    @cached_property
    def log_det(self):
        return np.sum(np.log(self.R.diagonal()))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    def to_dense(self):
        return self.M.copy()

    def update(self, *args):
        from scipy.linalg import (
            qr_update,
        )
        Q, R = self.Q, self.R
        for u, v in args:
            Q, R = qr_update(Q, R, np.ravel(u), np.ravel(v))
        return QROperator(Q, R, self.shape, self.ldim)


class CholeskyOperator(MatrixOperator):
    """This performs the whitening transforms for the passed
    cholesky factor of the Hessian/inverse covariance of the system.

    see https://en.wikipedia.org/wiki/Whitening_transformation

    >>> M = CholeskyOperator(linalg.cho_factor(hess))
    >>> y = M * x
    >>> f, df_dx = func_and_gradient(M.ldiv(y))
    >>> df_dy = df_df * M
    >>>
    """

    def __init__(self, cho_factor, shape=None, ldim=None):
        self.c, self.lower = self.cho_factor = cho_factor
        self.L = self.c if self.lower else self.c.T
        self.U = self.c.T if self.lower else self.c
        self._shape = shape or self.L.shape
        self._ldim = ldim or len(self._shape) // 2

    @classmethod
    def from_dense(cls, hess, shape=None, ldim=None):
        from scipy.linalg import (
            cho_factor,
        )
        shape = np.shape(hess)
        ldim = ldim or np.ndim(hess) // 2
        length = np.prod(shape[:ldim])
        return cls(cho_factor(np.reshape(hess, (length, length))), shape, ldim)

    @_wrap_leftop
    def __mul__(self, x):
        return _mul_triangular(self.U, x, lower=False)

    @_wrap_rightop
    def __rmul__(self, x):
        return _mul_triangular(self.L, x.T, lower=True).T

    @_wrap_rightop
    def __rtruediv__(self, x):

        from scipy.linalg import solve_triangular

        return solve_triangular(self.L, x.T, lower=True).T

    @_wrap_leftop
    def ldiv(self, x):

        from scipy.linalg import solve_triangular

        return solve_triangular(self.U, x, lower=False)

    @cached_property
    def log_det(self):
        return np.sum(np.log(self.U.diagonal()))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    def to_dense(self):
        return np.tril(self.L) @ np.triu(self.U)

    def matrixabs(self):
        return self


# This class may no longer be necessary?
class InvCholeskyTransform(CholeskyOperator):
    """In the case where the covariance matrix is passed
    we perform the inverse operations
    """

    __mul__ = CholeskyOperator.__rtruediv__
    __rmul__ = CholeskyOperator.ldiv
    __rtruediv__ = CholeskyOperator.__mul__
    ldiv = CholeskyOperator.__rmul__

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @cached_property
    def log_det(self):
        return -np.sum(np.log(self.U.diagonal()))

    def to_dense(self):

        from scipy.linalg import solve_triangular

        return solve_triangular(self.U, np.triul(self.L), lower=False)


class DiagonalMatrix(MatrixOperator):
    """
    Represents the DiagonalMatrix with diagonal `scale`

    M = np.diag(scale.ravel())

    """

    is_diagonal = True

    def __init__(self, scale, inv_scale=None):
        self.scale = np.asanyarray(scale)
        self._fscale = np.ravel(self.scale)

        # Lazily calculate inverse
        if inv_scale is not None:
            self._finv_scale = np.ravel(inv_scale)

        self._ldim = self.scale.ndim

    def __repr__(self):
        return repr(self.scale)

    def __getitem__(self, index):
        if index[: self.ldim] == index[self.ldim :]:
            return DiagonalMatrix(self.scale[index[: self.ldim]])
        else:
            raise NotImplementedError("Can't get diagonal for non 'square' operators")

    def reshape(self, shape):
        shape = shape[: len(shape) // 2]
        return DiagonalMatrix(self.scale.reshape(shape))

    @cached_property
    def _finv_scale(self):
        return 1 / self._fscale

    def inv(self):
        return DiagonalMatrix(
            np.reshape(self._finv_scale, self.scale.shape), self.scale
        )

    @classmethod
    def from_dense(cls, M: np.ndarray, shape=None, ldim=None) -> "MatrixOperator":
        return MatrixOperator(M, shape, ldim)

    @_wrap_leftop
    def __mul__(self, x):
        return self._fscale[:, None] * x

    @_wrap_rightop
    def __rmul__(self, x):
        return x * self._fscale

    @_wrap_rightop
    def __rtruediv__(self, x):
        return x * self._finv_scale

    @_wrap_leftop
    def ldiv(self, x):
        return self._finv_scale[:, None] * x

    @cached_property
    def log_det(self):
        return np.sum(np.log(self.scale))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @property
    def shape(self):
        return self.scale.shape * 2

    def to_dense(self):
        return np.diag(self._fscale).reshape(self.shape)

    def diagonal(self):
        return self.scale

    def __add__(self, other):
        if isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.scale + other.scale)
        else:
            return MatrixOperator(
                self.to_dense() + other.to_dense(), self.shape, self.ldim
            )

    def __sub__(self, other):
        if isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.scale - other.scale)
        else:
            return MatrixOperator(
                self.to_dense() - other.to_dense(), self.shape, self.ldim
            )

    def diagonalupdate(self, d):
        return type(self)(self.scale + d)

    def matrixabs(self):
        return type(self)(abs(self.scale))


class VecOuterProduct(LinearOperator):
    """
    represents the matrix vector outer product

    outer = vec[:, None] * vecT[None, :]
    """

    def __init__(self, vec, vecT=None):
        self.vec = np.asanyarray(vec)
        self.vecT = np.asanyarray(vec if vecT is None else vecT)
        self._fvec = np.ravel(self.vec)[:, None]
        self._fvecT = np.ravel(self.vecT)[None, :]

    @_wrap_leftop
    def __mul__(self, x):
        return self._fvec @ self._fvecT.dot(x)

    @_wrap_rightop
    def __rmul__(self, x):
        return x.dot(self._fvec) @ self._fvecT

    def __rtruediv__(self, x):
        raise NotImplementedError()

    def ldiv(self, x):
        raise NotImplementedError()

    @cached_property
    def log_det(self):
        return -np.inf

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    @property
    def shape(self):
        return (self.vec.size, self.vecT.size)

    @property
    def ldim(self):
        return len(self.vec.shape)

    def to_dense(self):
        return np.outer(self._fvec, self._fvecT).reshape(self.shape)


class MultiVecOuterProduct(LinearOperator):
    """
    represents the matrix vector outer product for stacked vectors,

    outer -> block_diag(*
        (v[:, None] * u[None, :] for v, u in zip(vec, vecT)
    )
    outer @ x -> np.vstack([
        v[:, None] * u[None, :] @ x for v, u in zip(vec, vecT)
    ])
    """

    def __init__(self, vec, vecT=None):
        self.vec = np.asanyarray(vec)
        self.vecT = np.asanyarray(vec if vecT is None else vecT)
        self.n, self.d = self.vec.shape
        self.nT, self.dT = self.vecT.shape

    @_wrap_leftop
    def __mul__(self, x):
        return np.einsum(
            "ij,ik,ikl -> ijl", self.vec, self.vecT, x.reshape(*self.lshape, -1)
        )

    @_wrap_rightop
    def __rmul__(self, x):
        return np.einsum(
            "ij,ik,lij -> lik", self.vec, self.vecT, x.reshape(-1, *self.rshape)
        )

    def __rtruediv__(self, x):
        raise NotImplementedError()

    def ldiv(self, x):
        raise NotImplementedError()

    @cached_property
    def log_det(self):
        return -np.inf

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    @property
    def shape(self):
        return self.vec.shape + self.vecT.shape

    @property
    def lshape(self):
        return self.shape[:2]

    @property
    def rshape(self):
        return self.shape[2:]

    def to_dense(self):

        from scipy.linalg import block_diag

        return block_diag(*(self.vec[:, :, None] * self.vecT[:, None, :])).reshape(
            self.shape
        )


class ShermanMorrison(LinearOperator):
    """
    Represents the Sherman-Morrison low rank update,
    inv(A + vec @ vec.T) =
        inv(A) - inv(A) @ vec @ vec.T @ inv(A) / (1 + vec @ inv(A) @ vec/T)
    """

    def __init__(self, linear, vec):
        self.linear = linear
        if np.ndim(vec) == 2:
            self.outer = MultiVecOuterProduct(vec)
        elif np.ndim(vec) == 1:
            self.outer = VecOuterProduct(vec)
        else:
            raise ValueError("vec must be 1 or 2 dimensional")

        self.inv_scale = 1 + linear.quad(vec)
        self._ldim = self.linear.ldim

    @_wrap_leftop
    def __mul__(self, x):
        return self.linear * x + self.outer * x

    @_wrap_rightop
    def __rmul__(self, x):
        return x * self.linear + x * self.outer

    @_wrap_rightop
    def __rtruediv__(self, x):
        x1 = x / self.linear
        return x1 - ((x1 / self.inv_scale) * self.outer) / self.linear

    @_wrap_leftop
    def ldiv(self, x):
        x1 = self.linear.ldiv(x)
        return x1 - self.linear.ldiv(self.outer.dot(x1 / self.inv_scale))

    @cached_property
    def log_det(self):
        return self.linear.log_det + np.log(self.inv_scale)

    @property
    def shape(self):
        return self.linear.shape

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    def to_dense(self):
        dense_outer = self.outer.to_dense()
        return self.linear.to_dense().reshape(self.shape) + dense_outer
