"""
The log-likelihood magnitude guard on the JAX paths.

The guard lives inside `Fitness.call`, so `_jit` and `_vmap` inherit it — but only if the ceiling
is a static Python float. A traced ceiling, or a Python `if` on the traced likelihood, would raise
at trace time rather than silently degrade, and these tests are what would catch that.

JAX is an optional dependency and unit tests are the always-green numpy layer, so the file skips
whole rather than importing `jax` unconditionally (the numpy half of the guard is covered in
`test_fitness_assertions.py`).
"""

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness
from test_autofit.non_linear.constant_analysis import ConstantAnalysis

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


#: Above the 1e20 ceiling but comfortably inside float32 range, so the test cannot pass for the
#: wrong reason (a value that overflows to `inf` would be caught by the isinf guard instead).
OVER_CEILING = 1.0e30

#: Below the ceiling, and representable in float32.
UNDER_CEILING = 1.0e19

PARAMETERS = [1.0, 1.0, 1.0]


def _jax_fitness(log_likelihood, **kwargs):
    return Fitness(
        model=af.Model(af.ex.Gaussian),
        analysis=ConstantAnalysis(log_likelihood=log_likelihood, use_jax=True),
        **kwargs,
    )


def test_jit_path_rejects_a_log_likelihood_above_the_ceiling():
    fitness = _jax_fitness(log_likelihood=OVER_CEILING, use_jax_jit=True)

    assert fitness._call is fitness._jit
    assert float(fitness._call(jnp.array(PARAMETERS))) == fitness.resample_figure_of_merit


def test_jit_path_passes_a_log_likelihood_below_the_ceiling():
    fitness = _jax_fitness(log_likelihood=UNDER_CEILING, use_jax_jit=True)

    figure_of_merit = float(fitness._call(jnp.array(PARAMETERS)))

    assert np.isfinite(figure_of_merit)
    assert figure_of_merit == pytest.approx(UNDER_CEILING, rel=1.0e-5)


def test_vmap_path_rejects_a_log_likelihood_above_the_ceiling():
    """
    `vmap` is the stricter check: every parameter is a tracer, so a ceiling that was not a static
    Python float would fail here even where jit-on-concrete passed.
    """
    fitness = _jax_fitness(log_likelihood=OVER_CEILING, use_jax_vmap=True)

    assert fitness._call is fitness._vmap

    figures_of_merit = np.asarray(fitness._call(jnp.array([PARAMETERS, PARAMETERS])))

    assert figures_of_merit.shape == (2,)
    assert np.all(figures_of_merit == fitness.resample_figure_of_merit)


def test_vmap_path_passes_a_log_likelihood_below_the_ceiling():
    fitness = _jax_fitness(log_likelihood=UNDER_CEILING, use_jax_vmap=True)

    figures_of_merit = np.asarray(fitness._call(jnp.array([PARAMETERS, PARAMETERS])))

    assert np.all(np.isfinite(figures_of_merit))
    assert figures_of_merit == pytest.approx(UNDER_CEILING, rel=1.0e-5)


def test_disabled_ceiling_traces_and_passes_any_finite_value():
    """
    With the guard disabled the `where` must still trace (it is not compiled away by a Python
    branch), and must return the value untouched.
    """
    fitness = _jax_fitness(log_likelihood=OVER_CEILING, use_jax_jit=True)
    fitness.log_likelihood_ceiling = float("inf")

    figure_of_merit = float(fitness._call(jnp.array(PARAMETERS)))

    assert figure_of_merit == pytest.approx(OVER_CEILING, rel=1.0e-5)
