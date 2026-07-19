from abc import abstractmethod
from typing import Dict, Tuple, Optional, List

import numpy as np

from autonerves import cached_property
from autofit.graphical.factor_graphs.abstract import AbstractNode, Value, FactorValue
# from ...mapper.operator import
from autofit.mapper.operator import (
    CholeskyOperator,
    InvCholeskyTransform,
    IdentityOperator,
    DiagonalMatrix,
)
from autofit.mapper.variable import Variable, VariableData


class VariableTransform:
    """ """

    def __init__(self, transforms):
        self.transforms = transforms

    def __mul__(self, values: Value) -> Value:
        return {k: M * values[k] for k, M in self.transforms.items()}

    def __rtruediv__(self, values: Value) -> Value:
        return {k: values[k] / M for k, M in self.transforms.items()}

    def __rmul__(self, values: Value) -> Value:
        return {k: values[k] * M for k, M in self.transforms.items()}

    def ldiv(self, values: Value) -> Value:
        return {k: M.ldiv(values[k]) for k, M in self.transforms.items()}

    def __add__(self, other: "VariableTransform") -> "VariableTransform":
        return VariableTransform(
            {k: M + other.transforms[k] for k, M in self.transforms.items()}
        )

    def inv(self) -> "VariableTransform":
        return VariableTransform({k: M.inv() for k, M in self.transforms.items()})

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    def quad(self, values):
        return {v: H.T if np.ndim(H) else H for v, H in (values * self).items()} * self

    def invquad(self, values):
        return {v: H.T if np.ndim(H) else H for v, H in (values / self).items()} / self

    @cached_property
    def log_det(self):
        return sum(M.log_det for M in self.transforms.values())

    @classmethod
    def from_scales(cls, scales):
        return cls({v: DiagonalMatrix(scale) for v, scale in scales.items()})

    @classmethod
    def from_covariances(cls, covs):
        from scipy.linalg import cho_factor

        return cls(
            {v: InvCholeskyTransform(cho_factor(cov)) for v, cov in covs.items()}
        )

    @classmethod
    def from_inv_covariances(cls, inv_covs):
        from scipy.linalg import cho_factor

        return cls(
            {
                v: CholeskyOperator(cho_factor(inv_cov))
                for v, inv_cov in inv_covs.items()
            }
        )


class FullCholeskyTransform(VariableTransform):
    def __init__(self, cholesky, param_shapes):
        self.cholesky = cholesky
        self.param_shapes = param_shapes

    @classmethod
    def from_optresult(cls, opt_result):
        param_shapes = opt_result.param_shapes

        cov = opt_result.result.hess_inv
        if not isinstance(cov, np.ndarray):
            # if optimiser is L-BFGS-B then convert
            # implicit hess_inv into dense matrix
            cov = cov.todense()

        return cls(InvCholeskyTransform.from_dense(cov), param_shapes)

    def __mul__(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(M * x)

    def __rtruediv__(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(x / M)

    def __rmul__(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(x * M)

    @abstractmethod
    def ldiv(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(M.ldiv(x))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @cached_property
    def log_det(self):
        return self.cholesky.log_det


class IdentityVariableTransform(VariableTransform):
    def __init__(self):
        pass

    def _identity(self, values: Value) -> Value:
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

    @property
    def log_det(self):
        return 0.0


identity_transform = IdentityOperator()
identity_variable_transform = IdentityVariableTransform()


class TransformedNode(AbstractNode):
    def __init__(self, node: AbstractNode, transform: VariableTransform):
        self.node = node
        self.transform = transform

    @property
    def variables(self):
        return self.node.variables

    @property
    def deterministic_variables(self):
        return self.node.deterministic_variables

    @property
    def all_variables(self):
        return self.node.all_variables

    @property
    def name(self):
        return f"FactorApproximation({self.node.name})"

    def __call__(
            self,
            values: Dict[Variable, np.ndarray],
    ) -> FactorValue:
        return self.node(self.transform.ldiv(values))

    def func_jacobian(
            self,
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            _calc_deterministic: bool = True,
            **kwargs,
    ) -> Tuple[FactorValue, VariableData]:
        fval, jval = self.node.func_jacobian(
            self.transform.ldiv(values),
            variables=variables,
            _calc_deterministic=_calc_deterministic,
        )

        # TODO this doesn't deal with deterministic jacobians
        grad = jval / self.transform
        return fval, grad

    def func_jacobian_hessian(
            self,
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            _calc_deterministic: bool = True,
            **kwargs,
    ) -> Tuple[FactorValue, VariableData, VariableData]:
        M = self.transform
        fval, jval, hval = self.node.func_jacobian_hessian(
            M.ldiv(values),
            variables=variables,
            _calc_deterministic=_calc_deterministic,
        )

        grad = jval / M
        # hess = {v: H.T for v, H in (hval / M).items()} / M
        hess = M.invquad(hval)
        return fval, grad, hess

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.node, name)
