from itertools import combinations
import numpy as np
import pytest

# Belt-and-suspenders: the top-level conftest also ignores this file when
# jax isn't installed; importorskip keeps the file self-contained.
jax = pytest.importorskip("jax")

from autofit.mapper.variable import variables
from autofit.graphical.factor_graphs import (
    Factor,
    FactorValue,
)


# def test_jacobian_equiv():
#
#     def linear(x, a, b, c):
#         z = x.dot(a) + b
#         return (z**2).sum(), z
#
#     x_, a_, b_, c_, z_ = variables("x, a, b, c, z")
#     x = np.arange(10.0).reshape(5, 2)
#     a = np.arange(2.0).reshape(2, 1)
#     b = np.ones(1)
#     c = -1.0
#
#     factors = [
#         Factor(
#             linear,
#             x_,
#             a_,
#             b_,
#             c_,
#             factor_out=(FactorValue, z_),
#             numerical_jacobian=False,
#         ),
#         Factor(
#             linear,
#             x_,
#             a_,
#             b_,
#             c_,
#             factor_out=(FactorValue, z_),
#             numerical_jacobian=False,
#             jacfwd=False,
#         ),
#         Factor(
#             linear,
#             x_,
#             a_,
#             b_,
#             c_,
#             factor_out=(FactorValue, z_),
#             numerical_jacobian=False,
#             vjp=True,
#         ),
#         Factor(
#             linear,
#             x_,
#             a_,
#             b_,
#             c_,
#             factor_out=(FactorValue, z_),
#             numerical_jacobian=True,
#         ),
#     ]
#
#     values = {x_: x, a_: a, b_: b, c_: c}
#     outputs = [factor.func_jacobian(values) for factor in factors]
#
#     tol = pytest.approx(0, abs=1e-4)
#     pairs = combinations(outputs, 2)
#     g0 = FactorValue(1.0, {z_: np.ones((5, 1))})
#     for (val1, jac1), (val2, jac2) in pairs:
#         assert val1 == val2
#
#         # test with different ways of calculating gradients
#         grad1, grad2 = jac1.grad(g0), jac2.grad(g0)
#         assert (grad1 - grad2).norm() == tol
#         grad1 = g0.to_dict() * jac1
#         assert (grad1 - grad2).norm() == tol
#         grad2 = g0.to_dict() * jac2
#         assert (grad1 - grad2).norm() == tol
#
#         grad1, grad2 = jac1.grad(val1), jac2.grad(val2)
#         assert (grad1 - grad2).norm() == tol
#
#         # test getting gradient with no args
#         assert (jac1.grad() - jac2.grad()).norm() == tol
#
#
# def test_jac_model():
#
#     def linear(x, a, b):
#         z = x.dot(a) + b
#         return (z**2).sum(), z
#
#     def likelihood(y, z):
#         return ((y - z) ** 2).sum()
#
#     def combined(x, y, a, b):
#         like, z = linear(x, a, b)
#         return like + likelihood(y, z)
#
#     x_, a_, b_, y_, z_ = variables("x, a, b, y, z")
#     x = np.arange(10.0).reshape(5, 2)
#     a = np.arange(2.0).reshape(2, 1)
#     b = np.ones(1)
#     y = np.arange(0.0, 10.0, 2).reshape(5, 1)
#     values = {x_: x, y_: y, a_: a, b_: b}
#     linear_factor = Factor(linear, x_, a_, b_, factor_out=(FactorValue, z_), vjp=True)
#     like_factor = Factor(likelihood, y_, z_, vjp=True)
#     full_factor = Factor(combined, x_, y_, a_, b_, vjp=True)
#     model_factor = like_factor * linear_factor
#
#     x = np.arange(10.0).reshape(5, 2)
#     a = np.arange(2.0).reshape(2, 1)
#     b = np.ones(1)
#     y = np.arange(0.0, 10.0, 2).reshape(5, 1)
#     values = {x_: x, y_: y, a_: a, b_: b}
#
#     # Fully working problem
#     fval, jac = full_factor.func_jacobian(values)
#     grad = jac.grad()
#
#     model_val, model_jac = model_factor.func_jacobian(values)
#     model_grad = model_jac.grad()
#
#     linear_val, linear_jac = linear_factor.func_jacobian(values)
#     like_val, like_jac = like_factor.func_jacobian(
#         {**values, **linear_val.deterministic_values}
#     )
#     combined_val = like_val + linear_val
#
#     # Manually back propagate
#     combined_grads = linear_jac.grad(like_jac.grad())
#
#     vals = (fval, model_val, combined_val)
#     grads = (grad, model_grad, combined_grads)
#     pairs = combinations(zip(vals, grads), 2)
#     for (val1, grad1), (val2, grad2) in pairs:
#         assert val1 == val2
#         assert (grad1 - grad2).norm() == pytest.approx(0, 1e-6)
