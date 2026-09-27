"""
Analysis-declared gradient mode (PyAutoFit#1648).

An ``Analysis`` declares how its likelihood is best differentiated through the class attribute
``gradient_mode`` ("reverse" by default, "forward" for likelihoods like the source-plane point-source
chi-squared, which carry an inner forward-mode derivative). ``autofit.jax.gradient`` resolves that
declaration against an optional search-level override and builds the matching JAX transform;
``Fitness.grad`` and ``MultiStartGradient`` both go through it.

The two modes are the same mathematical gradient, so every test below pins forward against reverse
rather than against a hand-written derivative: what this change may alter is speed, never numbers.
"""

import pickle
import shutil
import uuid

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness
from autonerves.dictable import from_dict, to_dict

jax = pytest.importorskip("jax")

from autofit.jax.gradient import (  # noqa: E402
    GRADIENT_MODES,
    grad_from,
    resolve_gradient_mode,
    value_and_grad_from,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _gaussian_data():
    xvalues = np.arange(40, dtype=float)
    truth = af.ex.Gaussian(centre=20.0, normalization=5.0, sigma=6.0)
    data = truth.model_data_from(xvalues=xvalues)
    return data, np.full_like(data, 0.5)


class ForwardAnalysis(af.ex.Analysis):
    gradient_mode = "forward"


def _fitness(analysis_cls=af.ex.Analysis, **kwargs):
    model = af.Model(af.ex.Gaussian)
    data, noise_map = _gaussian_data()
    analysis = analysis_cls(data=data, noise_map=noise_map, use_jax=True)
    return Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=False,
        convert_to_chi_squared=True,
        **kwargs,
    )


def _vectors(model, n=5, seed=3):
    rng = np.random.default_rng(seed)
    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for _ in range(n - 1):
        unit = rng.uniform(0.2, 0.8, size=model.prior_count)
        vectors.append(np.asarray(model.vector_from_unit_vector(list(unit)), dtype=float))
    return vectors


# --------------------------------------------------------------------------
# resolve_gradient_mode
# --------------------------------------------------------------------------


def test__modes_are_reverse_and_forward():
    assert GRADIENT_MODES == ("reverse", "forward")


def test__analysis_default_is_reverse():
    assert af.Analysis.gradient_mode == "reverse"
    assert resolve_gradient_mode(af.ex.Analysis(data=np.ones(3), noise_map=np.ones(3))) == "reverse"


def test__class_level_declaration_is_resolved():
    analysis = ForwardAnalysis(data=np.ones(3), noise_map=np.ones(3))

    assert resolve_gradient_mode(analysis) == "forward"


def test__override_wins_over_the_analysis_declaration():
    analysis = ForwardAnalysis(data=np.ones(3), noise_map=np.ones(3))

    assert resolve_gradient_mode(analysis, override="reverse") == "reverse"
    assert (
        resolve_gradient_mode(
            af.ex.Analysis(data=np.ones(3), noise_map=np.ones(3)), override="forward"
        )
        == "forward"
    )


def test__an_object_without_the_attribute_resolves_to_reverse():
    assert resolve_gradient_mode(object()) == "reverse"


@pytest.mark.parametrize("where", ["override", "analysis"])
def test__an_unknown_mode_raises_naming_the_valid_ones(where):
    class Bad(af.ex.Analysis):
        gradient_mode = "sideways"

    if where == "override":
        analysis, override = af.ex.Analysis(data=np.ones(3), noise_map=np.ones(3)), "sideways"
    else:
        analysis, override = Bad(data=np.ones(3), noise_map=np.ones(3)), None

    with pytest.raises(ValueError) as exc_info:
        resolve_gradient_mode(analysis, override=override)

    message = str(exc_info.value)
    assert "sideways" in message
    assert "reverse" in message and "forward" in message


def test__an_unknown_mode_raises_in_the_builders():
    with pytest.raises(ValueError):
        value_and_grad_from(lambda v: v.sum(), "sideways")
    with pytest.raises(ValueError):
        grad_from(lambda v: v.sum(), "sideways")


# --------------------------------------------------------------------------
# The transforms: forward is the reverse gradient, to round-off
# --------------------------------------------------------------------------


def test__value_and_grad_from__forward_matches_reverse_on_the_fitness_objective():
    fitness = _fitness()

    reverse = jax.jit(value_and_grad_from(fitness.call, "reverse"))
    forward = jax.jit(value_and_grad_from(fitness.call, "forward"))

    for vector in _vectors(fitness.model):
        value_r, grad_r = reverse(vector)
        value_f, grad_f = forward(vector)

        assert np.shape(value_f) == ()
        assert np.shape(grad_f) == np.shape(vector)
        assert np.all(np.asarray(grad_r) != 0.0)
        np.testing.assert_allclose(value_f, value_r, rtol=1e-10)
        np.testing.assert_allclose(grad_f, grad_r, rtol=1e-10)


def test__value_and_grad_from__reverse_is_jax_value_and_grad():
    fitness = _fitness()
    vector = _vectors(fitness.model, n=1)[0]

    value, grad = value_and_grad_from(fitness.call, "reverse")(vector)
    value_ref, grad_ref = jax.value_and_grad(fitness.call)(vector)

    np.testing.assert_array_equal(value, value_ref)
    np.testing.assert_array_equal(grad, grad_ref)


def test__grad_from__forward_matches_reverse():
    fitness = _fitness()

    for vector in _vectors(fitness.model):
        np.testing.assert_allclose(
            grad_from(fitness.call, "forward")(vector),
            grad_from(fitness.call, "reverse")(vector),
            rtol=1e-10,
        )


def test__forward_mode_rejects_a_non_scalar_objective():
    with pytest.raises(ValueError, match="scalar"):
        value_and_grad_from(lambda v: v * 2.0, "forward")(np.ones(3))


# --------------------------------------------------------------------------
# Fitness.grad honours the mode
# --------------------------------------------------------------------------


def test__fitness_gradient_mode_defaults_to_the_analysis_declaration():
    assert _fitness().gradient_mode is None
    assert _fitness(gradient_mode="forward").gradient_mode == "forward"


def test__fitness_grad_forward_matches_reverse():
    reverse = _fitness()
    forward_by_override = _fitness(gradient_mode="forward")
    forward_by_analysis = _fitness(analysis_cls=ForwardAnalysis)

    for vector in _vectors(reverse.model):
        grad_r = np.asarray(reverse.grad(vector))
        assert np.all(grad_r != 0.0)
        np.testing.assert_allclose(forward_by_override.grad(vector), grad_r, rtol=1e-10)
        np.testing.assert_allclose(forward_by_analysis.grad(vector), grad_r, rtol=1e-10)


def test__fitness_grad_builds_the_resolved_mode(monkeypatch):
    """The parity above cannot tell WHICH transform ran; this can."""
    import autofit.jax.gradient as gradient

    seen = []
    real = gradient.grad_from

    def spy(func, mode):
        seen.append(mode)
        return real(func, mode)

    monkeypatch.setattr(gradient, "grad_from", spy)

    fitness = _fitness(analysis_cls=ForwardAnalysis)
    fitness.grad(_vectors(fitness.model, n=1)[0])
    overridden = _fitness(analysis_cls=ForwardAnalysis, gradient_mode="reverse")
    overridden.grad(_vectors(overridden.model, n=1)[0])

    assert seen == ["forward", "reverse"]


def test__fitness_invalid_mode_raises_at_construction():
    with pytest.raises(ValueError):
        _fitness(gradient_mode="sideways")


def test__fitness_pickle_round_trip_keeps_the_mode():
    fitness = _fitness(gradient_mode="forward")
    vector = _vectors(fitness.model, n=1)[0]
    before = np.asarray(fitness.grad(vector))

    restored = pickle.loads(pickle.dumps(fitness))

    assert restored.gradient_mode == "forward"
    np.testing.assert_allclose(restored.grad(vector), before, rtol=1e-12)


def test__fitness_pickled_before_the_mode_existed_still_loads():
    fitness = _fitness()
    state = fitness.__getstate__()
    state.pop("gradient_mode")

    restored = Fitness.__new__(Fitness)
    restored.__setstate__(state)

    assert restored.gradient_mode is None
    vector = _vectors(restored.model, n=1)[0]
    np.testing.assert_allclose(restored.grad(vector), fitness.grad(vector), rtol=1e-12)


# --------------------------------------------------------------------------
# MultiStartGradient
# --------------------------------------------------------------------------


def test__multi_start_gradient_mode_defaults_to_none_and_is_carried():
    assert af.MultiStartAdam().gradient_mode is None
    for cls in (
        af.MultiStartAdam,
        af.MultiStartADABelief,
        af.MultiStartLion,
        af.MultiStartProdigy,
    ):
        assert cls(gradient_mode="forward").gradient_mode == "forward"


def test__multi_start_invalid_gradient_mode_raises_at_construction():
    with pytest.raises(ValueError) as exc_info:
        af.MultiStartAdam(gradient_mode="sideways")

    assert "reverse" in str(exc_info.value)


def test__multi_start_dict_round_trip_keeps_the_mode():
    restored = from_dict(to_dict(af.MultiStartAdam(gradient_mode="forward", n_starts=5)))

    assert isinstance(restored, af.MultiStartAdam)
    assert restored.gradient_mode == "forward"


def test__multi_start_pickle_round_trip_keeps_the_mode():
    restored = pickle.loads(pickle.dumps(af.MultiStartAdam(gradient_mode="forward")))

    assert restored.gradient_mode == "forward"


def test__multi_start_pickled_before_the_mode_existed_resolves_the_analysis():
    """A search unpickled from before this attribute existed carries no
    ``gradient_mode``; the fit must fall back to the analysis declaration."""
    search = af.MultiStartAdam()
    del search.__dict__["gradient_mode"]

    assert search._resolved_gradient_mode(ForwardAnalysis(data=np.ones(3), noise_map=np.ones(3))) == "forward"


def _multi_start_fit(gradient_mode, analysis_cls=af.ex.Analysis, **search_kwargs):
    pytest.importorskip("optax")

    model = af.Model(af.ex.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=40.0)
    model.normalization = af.UniformPrior(lower_limit=0.1, upper_limit=20.0)
    model.sigma = af.UniformPrior(lower_limit=1.0, upper_limit=15.0)

    data, noise_map = _gaussian_data()
    analysis = analysis_cls(data=data, noise_map=noise_map, use_jax=True)

    search = af.MultiStartAdam(
        name=f"gradient_mode_{gradient_mode}_{uuid.uuid4().hex}",
        n_starts=4,
        n_steps=25,
        seed=11,
        convergence=af.MultiStartGradientConvergence(check_for_convergence=False),
        gradient_mode=gradient_mode,
        **search_kwargs,
    )

    try:
        result = search.fit(model=model, analysis=analysis)
    finally:
        shutil.rmtree(search.paths.output_path, ignore_errors=True)

    return result


@pytest.mark.parametrize(
    "search_kwargs",
    [
        {},
        {"scaler": af.ScalerPriorWidth()},
        {"bijector": af.BijectorAuto()},
    ],
    ids=["physical", "scaler", "bijector"],
)
def test__multi_start_forward_and_reverse_give_the_same_fit(search_kwargs):
    reverse = _multi_start_fit("reverse", **search_kwargs)
    forward = _multi_start_fit("forward", **search_kwargs)

    np.testing.assert_allclose(
        forward.samples.max_log_likelihood(as_instance=False),
        reverse.samples.max_log_likelihood(as_instance=False),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        forward.log_likelihood, reverse.log_likelihood, rtol=1e-8
    )

    assert reverse.samples.samples_info["gradient_mode"] == "reverse"
    assert forward.samples.samples_info["gradient_mode"] == "forward"


def test__multi_start_resolves_the_analysis_declaration_when_unset():
    result = _multi_start_fit(None, analysis_cls=ForwardAnalysis)

    assert result.samples.samples_info["gradient_mode"] == "forward"


def test__multi_start_keyword_overrides_the_analysis_declaration():
    result = _multi_start_fit("reverse", analysis_cls=ForwardAnalysis)

    assert result.samples.samples_info["gradient_mode"] == "reverse"
