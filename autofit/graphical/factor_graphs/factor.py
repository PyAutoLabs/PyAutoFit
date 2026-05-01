from copy import deepcopy
from inspect import getfullargspec
from typing import Tuple, Dict, Any, Callable, Union, List, Optional, TYPE_CHECKING

import numpy as np

from autofit.graphical.utils import (
    nested_filter,
    to_variabledata,
    nested_zip,
    is_variable,
    try_getitem,
)
from autofit.mapper.variable import Variable, Plate, VariableData

from autofit.graphical.factor_graphs.abstract import FactorValue, AbstractFactor

if TYPE_CHECKING:
    from autofit.graphical.factor_graphs.jacobians import (
        VectorJacobianProduct,
        JacobianVectorProduct,
    )


class Factor(AbstractFactor):
    """Represents factors in Graphical models. The functions passed to this
    object will be called by positional arguments (to allow compatibility)
    with the jax API.

    Parameters
    ----------
    factor
        the function being wrapped, must accept calls
        through positional arguments

    *args: Variables
        Variables for each positional argument for the function

    factor_out, default FactorValue:
        The output of the factor. This can just be `FactorValue`
        or can be a arbitrarily nested structure of lists, tuples and dicts
        e.g.
        >>> foo = lambda x, y: (z, {'a': [a]})
        >>> factor = Factor(foo, x, y, factor_out=(z, {'a': [a]}))
    name: optional, str
        the name of the factor, if not passed then uses the name
        of the function passed

    plates: Tuple[Plate, ...] = ()
        plates that the factor are associated with

    vjp: optional False
        if True uses jax.vjp to calculate the Jacobian of the
        outputs

    factor_vjp: optional
        Must be a function produces functionaly equivalent
        output to jax.vjp(factor, *args) <equiv> factor_vjp(*args)

    factor_jacobian: optional
        function equivalent to calling,
        factor(*args), jax.jacobian(factor, len(range(args)))(*args)

    jacobian=None,
        function equivalent to calling,
        jax.jacobian(factor, len(range(args)))

    numerical_jacobian=True
        if True calculates Jacobian using finite differences
        if False calculates Jacobian using jax

    jacfwd=True
        if calculates jacobian using jax.jacfwd instead of
        jax.jacobian

    eps=1e-8
        the interval overwhich to calculate the finite differences




    Methods
    -------
    __call__({x: x0}) -> FactorValue
        calls the factor, the passed input must be a dictionary with
        where the keys are the Variable objects that the function takes
        as input.

        returns a FactorValue object which behaves like an np.ndarray
        deterministic values are stored in the deterministic_values
        attribute

    func_jacobian({x: x0}) -> Tuple[FactorValue, AbstractJacobianValue]
        calls the factor and returns it value and the Jacobian of its value
        with respect to the `variables` passed. The Jacobian is returned as
        a VariableLinearOperator with the appropriate methods for calculating
        the vector-Jacobian or Jacobian-vector products depending on how
        the Jacobian is calculated internally.

    Examples
    --------
    def linear(x, a, b):
        z = x.dot(a) + b
        return (z**2).sum(), z

    def likelihood(y, z):
        return ((y - z)**2).sum()

    def combined(x, y, a, b):
        like, z = linear(x, a, b)
        return like + likelihood(y, z)

    x_, a_, b_, y_, z_ = variables("x, a, b, y, z")
    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    values = {x_: x, y_: y, a_: a, b_: b}
    linear_factor = FactorVJP(
        linear, x_, a_, b_, factor_out=(FactorValue, z_))
    like_factor = FactorVJP(likelihood, y_, z_)
    full_factor = FactorVJP(combined, x_, y_, a_, b_)

    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    values = {x_: x, y_: y, a_: a, b_: b}

    # Fully working problem
    fval, jac = full_factor.func_jacobian(values)
    grad = jac(1)

    linear_val, linear_jac = linear_factor.func_jacobian(values)
    like_val, like_jac = like_factor.func_jacobian(
        {**values, **linear_val.deterministic_values})
    combined_val = like_val + linear_val

    combined_grads = {FactorValue: 1.}
    for v, g in like_jac(combined_grads).items():
        combined_grads[v] = g + combined_grads.get(v, 0)

    for v, g in linear_jac(combined_grads).items():
        combined_grads[v] = g + combined_grads.get(v, 0)

    assert (fval.log_value - combined_val.log_value) == 0
    assert (grad - combined_grads).norm() == 0
    """

    def __init__(
        self,
        factor,
        *args: Variable,
        name="",
        arg_names=None,
        factor_out=FactorValue,
        plates: Tuple[Plate, ...] = (),
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
        eps=1e-8,
    ):
        if not arg_names:
            arg_names = [arg for arg in getfullargspec(factor).args]
            if arg_names and arg_names[0] == "self":
                arg_names = arg_names[1:]

        # Make sure arg_names matches length of args
        for v in args[len(arg_names) :]:
            arg_name = v.name
            # Make sure arg_name is unique
            while arg_name in arg_names:
                arg_name += "_"
            arg_names.append(arg_name)

        kwargs = dict(zip(arg_names, args))
        name = name or factor.__name__

        AbstractFactor.__init__(
            self,
            name=name,
            plates=plates,
            factor_out=factor_out,
            **kwargs,
        )

        self.eps = eps
        self._set_factor(factor)
        self._set_jacobians(
            vjp=vjp,
            factor_vjp=factor_vjp,
            factor_jacobian=factor_jacobian,
            jacobian=jacobian,
            numerical_jacobian=numerical_jacobian,
            jacfwd=jacfwd,
        )

        self._cache = {}

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_cache":
                v = {}
            setattr(result, k, deepcopy(v, memodict))
        return result

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.plates:
            return self._factor.shape

        return ()

    def __getitem__(
        self, plates_index: Dict[Plate, Union[List[int], range, slice]]
    ) -> "Factor":
        return self.subset(plates_index)

    def subset(
        self,
        plates_index: Dict[Plate, Union[List[int], range, slice]],
        plate_sizes: Optional[Dict[Plate, int]] = None,
    ) -> "Factor":
        if not self.plates:
            return self

        plate_sizes = plate_sizes or dict(zip(self.plates, self.shape))
        index = Variable.make_indexes(self, plates_index, plate_sizes)
        subset_factor = self._factor[index]
        kws = self._subset_jacobian(subset_factor, index)

        subset = ", ".join(map("{0.name}={1}".format, self.plates, map(np.size, index)))
        kws["name"] = f"{self.name}[{subset}]"
        kws["eps"] = self.eps
        kws["factor_out"] = self.factor_out
        kws["plates"] = self.plates
        kws["arg_names"] = self.arg_names

        return Factor(subset_factor, *self.args, **kws)

    def _subset_jacobian(self, subset_factor, index):
        jac_kws = {"vjp": self._vjp}
        if self._vjp:
            factor_vjp = None
            if self._factor_vjp != self._jax_factor_vjp:
                factor_vjp = try_getitem(
                    self._factor_vjp, index, getattr(subset_factor, "factor_vjp", None)
                )

            jac_kws["factor_vjp"] = factor_vjp
        else:
            factor_jacobian = None
            jacobian = None
            if self._factor_jacobian != Factor._factor_jacobian:
                factor_jacobian = try_getitem(
                    self._factor_jacobian,
                    index,
                    getattr(subset_factor, "factor_jacobian", None),
                )
            elif self._jacobian != Factor._jacobian:
                jacobian = try_getitem(
                    self.jacobian, index, getattr(subset_factor, "jacobian", None)
                )
            elif self._factor_jacobian != self._numerical_factor_jacobian:
                jac_kws["jacfwd"] = self._jacfwd

            jac_kws["jacobian"] = jacobian
            jac_kws["factor_jacobian"] = factor_jacobian

        return jac_kws

    def _set_jacobians(
        self,
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
    ):
        self._vjp = vjp
        self._jacfwd = jacfwd
        if vjp or factor_vjp:
            self._factor_vjp = factor_vjp
            self.func_jacobian = self._vjp_func_jacobian
        else:
            # This is set by default
            # self.func_jacobian = self._jvp_func_jacobian

            if factor_jacobian:
                self._factor_jacobian = factor_jacobian
            elif jacobian:
                self._jacobian = jacobian
            elif numerical_jacobian:
                self._factor_jacobian = self._numerical_factor_jacobian
            elif jacfwd:
                import jax
                self._jacobian = jax.jacfwd(self._factor, range(self.n_args))
            else:
                import jax
                self._jacobian = jax.jacobian(self._factor, range(self.n_args))

    def _factor_value(self, raw_fval) -> FactorValue:
        """Converts the raw output of the factor into a `FactorValue`
        where the values of the deterministic values are stored in a dict
        attribute `FactorValue.deterministic_values`
        """
        det_values = to_variabledata(self.factor_out, raw_fval)
        fval = det_values.pop(FactorValue, 0.0)
        return FactorValue(fval, det_values)

    def __call__(self, values: VariableData) -> FactorValue:
        """Calls the factor with the values specified by the dictionary of
        values passed, returns a FactorValue with the value returned by the
        factor, and any deterministic factors"""
        args = self.resolve_args(values)
        key = self._key("__call__", *(val for _, val in nested_zip(self.args, args)))

        if key not in self._cache:
            raw_fval = self._factor_args(*args)
            self._cache[key] = self._factor_value(raw_fval)
        return self._cache[key]

    def _jax_factor_vjp(self, *args) -> Tuple[Any, Callable]:
        import jax
        return jax.vjp(self._factor, *args)

    _factor_vjp = _jax_factor_vjp

    def _vjp_func_jacobian(
        self, values: VariableData
    ) -> Tuple[FactorValue, "VectorJacobianProduct"]:
        """Calls the factor and returns the factor value with deterministic
        values, and a `VectorJacobianProduct` operator that allows the
        calculation of the gradient of the input values to be calculated
        with respect to the gradients of the output values (i.e backprop)
        """
        from autofit.graphical.factor_graphs.jacobians import (
            VectorJacobianProduct,
        )

        raw_fval, fvjp = self._factor_vjp(*self.resolve_args(values))
        fval = self._factor_value(raw_fval)

        fvjp_op = VectorJacobianProduct(
            self.factor_out,
            fvjp,
            *self.args,
            out_shapes=fval.to_dict().shapes,
        )
        return fval, fvjp_op

    @staticmethod
    def _key(*args):
        import xxhash as xxhash

        h = xxhash.xxh64()

        for arg in args:
            try:
                h.update(arg)
            except TypeError:
                h.update(np.array(arg))

        return h.intdigest()

    def _jvp_func_jacobian(
        self, values: VariableData, **kwargs
    ) -> Tuple[FactorValue, "JacobianVectorProduct"]:
        args = self.resolve_args(values)
        key = self._key("_jvp_func_jacobian", *args)

        if key not in self._cache:
            raw_fval, raw_jac = self._factor_jacobian(*args, **kwargs)
            fval = self._factor_value(raw_fval)
            jvp = self._jac_out_to_jvp(raw_jac, values=fval.to_dict().merge(values))
            self._cache[key] = (fval, jvp)
        return self._cache[key]

    func_jacobian = _jvp_func_jacobian

    def _factor_jacobian(self, *args) -> Tuple[Any, Any]:
        return self._factor_args(*args), self._jacobian(*args)

    def _factor_args(self, *args):
        return self._factor(*args)

    def _unpack_jacobian_out(self, raw_jac: Any) -> Dict[Variable, VariableData]:
        jac = {}
        for v0, vjac in nested_filter(is_variable, self.factor_out, raw_jac):
            jac[v0] = VariableData()
            for v1, j in nested_zip(self.args, vjac):
                jac[v0][v1] = j

        return jac

    def _jac_out_to_jvp(
        self, raw_jac: Any, values: VariableData
    ) -> "JacobianVectorProduct":
        from autofit.graphical.factor_graphs.jacobians import (
            JacobianVectorProduct,
        )

        jac = self._unpack_jacobian_out(raw_jac)
        return JacobianVectorProduct.from_dense(jac, values=values)


class FactorKW(Factor):
    """Represents factors in Graphical models. The functions passed to this
    object will be called by keyword arguments

    Parameters
    ----------
    factor
        the function being wrapped, must accept calls
        through positional arguments
    **kwargs: Variables
        Variables for each keyword argument for the function
    factor_out:
        The output of the factor. This can just be `FactorValue`
        or can be a arbitrarily nested structure of lists, tuples and dicts
        e.g.
        >>> foo = lambda x, y: (z, {'a': [a]})
        >>> factor = Factor(foo, x, y, factor_out=(z, {'a': [a]}))
    name: optional, str
        the name of the factor, if not passed then uses the name
        of the function passed

    Methods
    -------
    __call__({x: x0}) -> FactorValue
        calls the factor, the passed input must be a dictionary with
        where the keys are the Variable objects that the function takes
        as input.

        returns a FactorValue object which behaves like an np.ndarray
        deterministic values are stored in the deterministic_values
        attribute

    func_jacobian({x: x0}) -> Tuple[FactorValue, AbstractJacobianValue]
        calls the factor and returns it value and the Jacobian of its value
        with respect to the `variables` passed. The Jacobian is returned as
        a VariableLinearOperator with the appropriate methods for calculating
        the vector-Jacobian or Jacobian-vector products depending on how
        the Jacobian is calculated internally.
    """

    def __init__(
        self,
        factor,
        name="",
        arg_names=None,
        factor_out=FactorValue,
        plates: Tuple[Plate, ...] = (),
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
        eps=1e-8,
        **kwargs: Variable,
    ):
        args = tuple(kwargs.values())
        arg_names = tuple(kwargs.keys())
        super().__init__(
            factor,
            *args,
            name=name,
            arg_names=arg_names,
            factor_out=factor_out,
            plates=plates,
            vjp=vjp,
            factor_vjp=factor_vjp,
            factor_jacobian=factor_jacobian,
            jacobian=jacobian,
            numerical_jacobian=numerical_jacobian,
            jacfwd=jacfwd,
            eps=eps,
        )

    _factor_args = AbstractFactor._factor_args
