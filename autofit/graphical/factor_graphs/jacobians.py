import numpy as np

from autonerves import cached_property
from autofit.graphical.utils import (
    nested_filter,
    nested_update,
    is_variable,
    to_variabledata,
)
from autofit.mapper.variable import (
    Variable,
    FactorValue,
    VariableData,
    VariableLinearOperator,
)
from autofit.mapper.variable_operator import (
    RectVariableOperator,
)
from typing import (
    Tuple,
    Dict,
    Union,
    Callable,
    Protocol,
)

Value = Dict[Variable, np.ndarray]
GradientValue = VariableData


class FactorInterface(Protocol):
    def __call__(self, values: Value) -> FactorValue:
        pass


class FactorGradientInterface(Protocol):
    def __call__(self, values: Value) -> Tuple[FactorValue, GradientValue]:
        pass


class AbstractJacobian(VariableLinearOperator):
    """
    Examples
    --------
    def linear(x, a, b):
        z = x.dot(a) + b
        return (z**2).sum(), z

    def full(x, a, b):
        z2, z = linear(x, a, b)
        return z2 + z.sum()

    x_, a_, b_, y_, z_ = variables("x, a, b, y, z")
    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    # values = {x_: x, y_: y, a_: a, b_: b}

    linear_factor_jvp = FactorJVP(
        linear, x_, a_, b_, factor_out=(FactorValue, z_))

    linear_factor_vjp = FactorVJP(
        linear, x_, a_, b_, factor_out=(FactorValue, z_))

    values = {x_: x, a_: a, b_: b}

    jvp_val, jvp_jac = linear_factor_jvp.func_jacobian(values)
    vjp_val, vjp_jac = linear_factor_vjp.func_jacobian(values)


    assert np.allclose(vjp_val, jvp_val)
    assert (vjp_jac(vjp_val) - jvp_jac(vjp_val)).norm() == 0
    """

    def __call__(self, values):
        return self.__rmul__(values)

    def __str__(self) -> str:
        out_var = str(
            nested_update(self.factor_out, {v: v.name for v in self.out_variables})
        ).replace("'", "")

        in_var = ", ".join(v.name for v in self.variables)
        cls_name = type(self).__name__
        return f"{cls_name}({out_var} → ∂({in_var})ᵀ {out_var})"

    __repr__ = __str__

    def _full_repr(self) -> str:
        out_var = str(self.factor_out)
        in_var = str(self.variables)
        cls_name = type(self).__name__
        return f"{cls_name}({out_var} → ∂({in_var})ᵀ {out_var})"

    @property
    def cotangent_variables(self):
        """
        The variables whose cotangents ``self(seed)`` accepts: the factor's
        outputs, i.e. ``FactorValue`` and its deterministic variables.

        ``VectorJacobianProduct`` exposes these as ``out_variables``;
        ``JacobianVectorProduct`` (a ``RectVariableOperator`` mapping its
        ``left_variables`` onto its ``right_variables``) names the same set
        ``left_variables`` and uses ``out_variables`` for the inputs, so it
        overrides this property.
        """
        return self.out_variables

    def grad(self, values=None):
        """
        The gradient of the factor value with respect to its input variables,
        pulled back through this Jacobian.

        Parameters
        ----------
        values
            Optional cotangents. Entries keyed by the factor's output variables
            (``FactorValue`` and its deterministic variables) seed the
            vector-Jacobian product; entries keyed by the input variables
            themselves (e.g. the cavity gradient of the free variables) are not
            part of the seed and are added to the result after the pull-back.
        """
        seed = VariableData({FactorValue: 1.0})
        grad = VariableData(values) if values else VariableData()
        if values:
            for v in self.cotangent_variables:
                if v in values:
                    seed[v] = values[v]

        jac = self(seed)

        for v, g in jac.items():
            if v is not FactorValue:
                grad[v] = grad.get(v, 0) + g

        return grad


class JacobianVectorProduct(AbstractJacobian, RectVariableOperator):
    __init__ = RectVariableOperator.__init__

    @property
    def variables(self):
        return self.left_variables

    @property
    def out_variables(self):
        return self.right_variables

    @property
    def cotangent_variables(self):
        return self.left_variables

    @property
    def factor_out(self):
        return tuple(self.out_variables)


class VectorJacobianProduct(AbstractJacobian):
    def __init__(
        self,
        factor_out,
        vjp: Callable,
        *args: Variable,
        out_shapes=None,
    ):
        self.factor_out = factor_out
        self.vjp = vjp
        self._args = args
        self._variables = tuple(v for v, in nested_filter(is_variable, args))
        self.out_shapes = out_shapes

    @property
    def args(self):
        return self._args

    @property
    def variables(self):
        return self._variables

    @cached_property
    def out_variables(self):
        return set(v[0] for v in nested_filter(is_variable, self.factor_out))

    def _get_cotangent(self, values):
        if isinstance(values, FactorValue):
            values = values.to_dict()

        if isinstance(values, dict):
            if self.out_shapes:
                for v in self.out_shapes.keys() - values.keys():
                    values[v] = np.zeros(self.out_shapes[v])
            out = nested_update(self.factor_out, values)
            return out

        if isinstance(values, int):
            values = float(values)

        return values

    def __call__(self, values: Union[VariableData, FactorValue]) -> VariableData:
        v = self._get_cotangent(values)
        grads = self.vjp(v)
        return to_variabledata(self.args, grads)

    __rmul__ = __call__

    def _not_implemented(self, *args):
        raise NotImplementedError()

    __rtruediv__ = _not_implemented
    ldiv = _not_implemented
    __mul__ = _not_implemented
    update = _not_implemented
