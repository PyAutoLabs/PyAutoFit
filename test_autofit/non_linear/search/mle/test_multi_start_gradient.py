import numpy as np
import pytest

import autofit as af
from autofit import example
from autoconf.dictable import from_dict, to_dict

# The MultiStart gradient searches are JAX-native at fit time, but their
# plumbing (config knobs, dict round-trip, and the internal-results -> Samples
# mapping) is pure NumPy and is tested here. The end-to-end JAX/optax fit that
# recovers a truth basin is validated in autofit_workspace_test, keeping JAX out
# of the library unit suite.

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__explicit_params():
    search = af.MultiStartAdam(
        n_starts=16,
        n_steps=100,
        learning_rate=0.05,
        start_lower_limit=0.2,
        start_upper_limit=0.8,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
    )

    assert search.n_starts == 16
    assert search.n_steps == 100
    assert search.learning_rate == 0.05
    assert search.start_lower_limit == 0.2
    assert search.start_upper_limit == 0.8
    assert search.optax_method == "adam"
    assert isinstance(search.initializer, af.InitializerBall)


def test__per_rule_defaults():
    # Each concrete search fixes its optax rule and a sensible default lr; Lion is
    # sign-based so it defaults to a ~10x smaller learning rate.
    assert af.MultiStartAdam().optax_method == "adam"
    assert af.MultiStartAdam().learning_rate == pytest.approx(1.0e-2)

    assert af.MultiStartADABelief().optax_method == "adabelief"
    assert af.MultiStartADABelief().learning_rate == pytest.approx(1.0e-2)

    assert af.MultiStartLion().optax_method == "lion"
    assert af.MultiStartLion().learning_rate == pytest.approx(1.0e-3)

    # defaults for the shared knobs
    default = af.MultiStartAdam()
    assert default.n_starts == 48
    assert default.n_steps == 300
    assert default.start_lower_limit == 0.15
    assert default.start_upper_limit == 0.85


def test__dict_round_trip():
    dictionary = to_dict(af.MultiStartLion(n_starts=7, n_steps=33))
    restored = from_dict(dictionary)

    assert isinstance(restored, af.MultiStartLion)
    assert restored.n_starts == 7
    assert restored.n_steps == 33
    assert restored.optax_method == "lion"
    assert restored.learning_rate == pytest.approx(1.0e-3)


def test__samples_via_internal_from():
    model = af.Model(example.Gaussian)

    # Valid physical parameter vectors via the (NumPy) unit transform.
    best_params = np.asarray(model.vector_from_unit_vector([0.5] * model.prior_count))
    start_a = np.asarray(model.vector_from_unit_vector([0.4] * model.prior_count))
    start_b = np.asarray(model.vector_from_unit_vector([0.6] * model.prior_count))
    per_start_params = np.stack([start_a, start_b])

    # Fitness.call returns -2 * log_posterior; pick a known log_posterior.
    best_log_posterior = 5.0
    best_fom = -2.0 * best_log_posterior

    search_internal = {
        "params": per_start_params,
        "best_params": best_params,
        "best_fom": best_fom,
        "fom_history": np.asarray([-4.0, -8.0, best_fom]),
        "total_steps": 42,
    }

    search = af.MultiStartAdam(n_starts=2, n_steps=42)
    samples = search.samples_via_internal_from(
        model=model, search_internal=search_internal
    )

    # First sample is the best-basin MAP point; the two starts follow as
    # zero-weight diagnostics.
    assert samples.parameter_lists[0] == pytest.approx(list(best_params))
    assert len(samples.parameter_lists) == 3
    assert samples.weight_list[0] == 1.0
    assert all(w == 0.0 for w in samples.weight_list[1:])

    # Sign convention: log_likelihood[0] = log_posterior - log_prior.
    log_prior_0 = model.log_prior_list_from(parameter_lists=[list(best_params)])[0]
    assert samples.log_likelihood_list[0] == pytest.approx(
        best_log_posterior - log_prior_0
    )

    # The MAP point is recovered as the maximum-likelihood instance.
    assert samples.max_log_likelihood(as_instance=False) == pytest.approx(
        list(best_params)
    )

    assert samples.samples_info["optax_method"] == "adam"
    assert samples.samples_info["n_starts"] == 2
    assert samples.samples_info["total_steps"] == 42
