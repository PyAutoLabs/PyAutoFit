"""
`Fitness.objective(kind)`, the one objective factory (search-extensibility phase A2).

The compile-count probe follows `test_fitness_vmap_cache.py`: under `jax.jit` the
likelihood's Python body runs only while tracing, so a counting analysis counts traces, and
`objective(kind).__wrapped__._cache_size()` counts compiled executables. Each kind must
compile exactly once for a fixed input shape, however often it is called.
"""
import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.objective import (
    OBJECTIVE_KINDS,
    chunk_slices,
    evaluate_in_chunks,
)


class CountingAnalysis(af.ex.Analysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_count = 0

    def log_likelihood_function(self, instance, **kwargs):
        self.trace_count += 1
        return super().log_likelihood_function(instance=instance, **kwargs)


def _fitness(use_jax, **kwargs):
    model = af.Model(af.ex.Gaussian)
    analysis = CountingAnalysis(
        data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=use_jax
    )
    return Fitness(model=model, analysis=analysis, **kwargs)


def _vector(fitness):
    return np.array(fitness.model.physical_values_from_prior_medians)


def _input_for(kind, fitness):
    vector = _vector(fitness)
    if kind.startswith("batched"):
        return np.tile(vector, (4, 1)) * np.linspace(0.9, 1.1, 4)[:, None]
    return vector


def test_unknown_kind_raises():
    fitness = _fitness(use_jax=False)
    with pytest.raises(ValueError, match="Unknown objective kind"):
        fitness.objective("hessian")


def test_numpy_scalar_is_the_plain_call():
    fitness = _fitness(use_jax=False)
    assert fitness.objective("scalar").__func__ is Fitness.call
    assert fitness._call.__func__ is Fitness.call


def test_numpy_batched_loops_over_call():
    fitness = _fitness(use_jax=False)
    batch = _input_for("batched", fitness)

    values = fitness.objective("batched")(batch)

    assert values.shape == (4,)
    assert np.allclose(values, [fitness.call(row) for row in batch])


@pytest.mark.parametrize("kind", ["value_and_grad", "batched_value_and_grad"])
def test_numpy_grad_kinds_raise(kind):
    fitness = _fitness(use_jax=False)
    with pytest.raises(exc.SearchException, match="needs a JAX analysis"):
        fitness.objective(kind)


@pytest.mark.parametrize("kind", OBJECTIVE_KINDS)
def test_one_compile_per_kind(kind):
    pytest.importorskip("jax")

    fitness = _fitness(use_jax=True)
    objective = fitness.objective(kind)
    parameters = _input_for(kind, fitness)

    # Building the objective compiles nothing: jit is lazy.
    assert fitness.analysis.trace_count == 0
    assert objective.__wrapped__._cache_size() == 0

    first = objective(parameters)
    second = objective(parameters)

    assert fitness.analysis.trace_count == 1
    assert objective.__wrapped__._cache_size() == 1
    # Cached per kind: asking again returns the same compiled callable.
    assert fitness.objective(kind) is objective

    first = first if isinstance(first, tuple) else (first,)
    second = second if isinstance(second, tuple) else (second,)
    for a, b in zip(first, second):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_kinds_agree_with_the_eager_call():
    pytest.importorskip("jax")

    fitness = _fitness(use_jax=True)
    vector = _vector(fitness)
    batch = _input_for("batched", fitness)

    expected = float(fitness.call(vector))

    assert float(fitness.objective("scalar")(vector)) == pytest.approx(expected)

    value, grad = fitness.objective("value_and_grad")(vector)
    assert float(value) == pytest.approx(expected)
    assert np.asarray(grad).shape == vector.shape

    values, grads = fitness.objective("batched_value_and_grad")(batch)
    assert np.allclose(np.asarray(values), np.asarray(fitness.objective("batched")(batch)))
    assert np.asarray(grads).shape == batch.shape


def test_compile_false_is_the_eager_escape_hatch():
    pytest.importorskip("jax")

    fitness = _fitness(use_jax=True)
    vector = _vector(fitness)

    eager = fitness.objective("scalar", compile=False)
    eager(vector)
    eager(vector)

    # Unjitted: the Python body runs on every call, and nothing is cached.
    assert fitness.analysis.trace_count == 2
    assert not hasattr(eager, "__wrapped__")
    assert eager is not fitness._objectives["scalar"]

    hatch = _fitness(use_jax=True, compile=False)
    hatch(vector)
    hatch(vector)
    assert hatch.analysis.trace_count == 2


def test_call_wrap_on_jax_is_jitted_and_compiles_once_across_input_types():
    """
    The scalar path of `call_wrap` (Emcee, Zeus, Drawer, the initializer) is jitted on a JAX
    analysis, and a Python list and an ndarray share one compiled executable.
    """
    pytest.importorskip("jax")

    fitness = _fitness(use_jax=True)
    vector = _vector(fitness)

    a = fitness(vector)
    b = fitness(list(vector))

    assert isinstance(a, float) and a == b
    assert fitness.analysis.trace_count == 1
    assert fitness.objective("scalar").__wrapped__._cache_size() == 1


def test_call_wrap_value_and_grad_books_history():
    pytest.importorskip("jax")

    fitness = _fitness(use_jax=True, store_history=True)
    vector = _vector(fitness)

    value, grad = fitness.call_wrap_value_and_grad(vector)

    assert isinstance(value, float)
    assert grad.dtype == np.float64 and grad.shape == vector.shape
    assert len(fitness.parameters_history_list) == 1
    assert len(fitness.log_likelihood_history_list) == 1


def test_objective_cache_is_stripped_on_pickle():
    pytest.importorskip("jax")
    import pickle

    fitness = _fitness(use_jax=True)
    vector = _vector(fitness)
    fitness.objective("scalar")(vector)

    assert "_objectives" not in fitness.__getstate__()

    restored = pickle.loads(pickle.dumps(fitness))

    # Rebuilt lazily: nothing compiled until the restored objective is first called.
    assert restored.objective("scalar").__wrapped__._cache_size() == 0
    assert float(restored.objective("scalar")(vector)) == pytest.approx(
        float(fitness.objective("scalar")(vector))
    )


def test_deprecated_kwargs_warn_and_alias():
    with pytest.warns(FutureWarning, match="use_jax_vmap"):
        fitness = _fitness(use_jax=False, use_jax_vmap=True)
    assert fitness.batched is True
    assert fitness.use_jax_vmap is True

    with pytest.warns(FutureWarning, match="use_jax_jit"):
        fitness = _fitness(use_jax=False, use_jax_jit=False)
    assert fitness.compile is False


def test_chunk_slices_pad_only_the_final_chunk():
    assert chunk_slices(5, 2) == [(0, 2, 0), (2, 4, 0), (4, 5, 1)]
    assert chunk_slices(4, 2) == [(0, 2, 0), (2, 4, 0)]


def test_evaluate_in_chunks_matches_one_call_and_compiles_one_shape():
    pytest.importorskip("jax")

    fitness = _fitness(use_jax=True)
    batch = np.tile(_vector(fitness), (5, 1)) * np.linspace(0.9, 1.1, 5)[:, None]

    batched = fitness.objective("batched")

    chunked = evaluate_in_chunks(batched, batch, batch_size=2)
    whole = batched(batch)

    assert np.allclose(np.asarray(chunked), np.asarray(whole))
    # Shapes (2, d) and (5, d): the padded ragged chunk reuses the (2, d) executable.
    assert batched.__wrapped__._cache_size() == 2
