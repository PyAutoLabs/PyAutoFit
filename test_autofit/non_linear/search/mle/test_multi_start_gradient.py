import numpy as np
import pytest

import autofit as af
from autofit import example
from autonerves.dictable import from_dict, to_dict

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
        batch_size=4,
        start_lower_limit=0.2,
        start_upper_limit=0.8,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
    )

    assert search.n_starts == 16
    assert search.n_steps == 100
    assert search.learning_rate == 0.05
    assert search.batch_size == 4
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

    # Prodigy is learning-rate-free: it resolves from optax.contrib and takes no
    # learning rate (None -> built from the rule's own default at fit time).
    assert af.MultiStartProdigy().optax_method == "prodigy"
    assert af.MultiStartProdigy().learning_rate is None
    assert af.MultiStartProdigy._default_learning_rate is None

    # defaults for the shared knobs
    default = af.MultiStartAdam()
    assert default.n_starts == 48
    assert default.n_steps == 300
    assert default.start_lower_limit == 0.15
    assert default.start_upper_limit == 0.85
    # batch_size defaults to None = evaluate all starts in one vmapped call
    # (the pre-batch_size behaviour, so no regression).
    assert default.batch_size is None
    # apply_if_finite per-start rejected-step budget (the in-step NaN guard).
    assert default.max_consecutive_nan == 8
    # restart-on-death is opt-in — default off keeps the MGE-cell behaviour.
    assert default.resurrect is False


def test__resurrect_defaults_off_and_is_a_carried_knob():
    """``resurrect`` (restart-on-death) is a shared base-class knob, off by
    default and dict-serialised so a resumed search keeps it. The redraw /
    per-start state reinit itself is JAX and is validated in
    autofit_workspace_test (the library suite is NumPy-only)."""
    for cls in (
        af.MultiStartAdam,
        af.MultiStartADABelief,
        af.MultiStartLion,
        af.MultiStartProdigy,
    ):
        assert cls().resurrect is False
        assert cls(resurrect=True).resurrect is True


def test__max_consecutive_nan_is_a_carried_knob():
    """The apply_if_finite budget is a shared base-class knob, dict-serialised
    so a resumed search rebuilds the same guard."""
    for cls in (
        af.MultiStartAdam,
        af.MultiStartADABelief,
        af.MultiStartLion,
        af.MultiStartProdigy,
    ):
        assert cls().max_consecutive_nan == 8
        assert cls(max_consecutive_nan=3).max_consecutive_nan == 3


def test__batch_size_is_carried_to_every_rule():
    """``batch_size`` is a shared knob on the abstract base, not per-rule.

    The numerical guarantee it must honour — chunked evaluation is identical to
    the unchunked vmap — is a JAX property of ``jax.lax.map(..., batch_size=)``
    and is asserted in autofit_workspace_test, since the library suite is
    NumPy-only.
    """
    for cls in (
        af.MultiStartAdam,
        af.MultiStartADABelief,
        af.MultiStartLion,
        af.MultiStartProdigy,
    ):
        assert cls().batch_size is None
        assert cls(batch_size=8).batch_size == 8


def test__convergence_default_is_on_and_carried():
    """Auto-convergence is a shared base-class setting, on by default (so users do
    not hand-tune ``n_steps``), and a custom settings object is carried through."""
    for cls in (
        af.MultiStartAdam,
        af.MultiStartADABelief,
        af.MultiStartLion,
        af.MultiStartProdigy,
    ):
        default = cls().convergence
        assert isinstance(default, af.MultiStartGradientConvergence)
        assert default.check_for_convergence is True
        # documented defaults for the plateau check.
        assert default.window == 50
        assert default.min_steps == 100
        assert default.rtol == pytest.approx(1.0e-4)
        assert default.atol == pytest.approx(1.0e-3)

        custom = af.MultiStartGradientConvergence(check_for_convergence=False)
        assert cls(convergence=custom).convergence is custom


def test__check_if_converged__plateau_stops_climbing_does_not():
    # rtol/atol both zero: converges only on an exactly-flat window.
    convergence = af.MultiStartGradientConvergence(
        window=3, min_steps=3, rtol=0.0, atol=0.5
    )

    # Flat global-best over the window -> converged.
    assert convergence.check_if_converged([10.0, 10.0, 10.0]) is True

    # Still descending over the window -> not converged.
    assert convergence.check_if_converged([10.0, 5.0, 1.0]) is False


def test__check_if_converged__respects_min_steps_and_window_length():
    convergence = af.MultiStartGradientConvergence(
        window=3, min_steps=5, rtol=0.0, atol=1.0
    )

    # Fewer than max(min_steps, window) entries -> never terminates early.
    assert convergence.check_if_converged([10.0, 10.0, 10.0]) is False
    assert convergence.check_if_converged([10.0, 10.0, 10.0, 10.0]) is False

    # Once min_steps entries exist and the window is flat -> converged.
    assert convergence.check_if_converged([10.0] * 5) is True


def test__check_if_converged__relative_tolerance():
    convergence = af.MultiStartGradientConvergence(
        window=3, min_steps=3, rtol=1.0e-4, atol=0.0
    )

    # Improvement 0.1 over a best-fom ~1000 is within rtol * 1000 = 0.1 -> converged.
    assert convergence.check_if_converged([1000.1, 1000.05, 1000.0]) is True

    # A large improvement over the same window is not within tolerance.
    assert convergence.check_if_converged([1002.0, 1001.0, 1000.0]) is False


def test__check_if_converged__disabled_and_non_finite():
    # check_for_convergence=False never terminates, even on a flat window.
    disabled = af.MultiStartGradientConvergence(
        check_for_convergence=False, window=3, min_steps=3
    )
    assert disabled.check_if_converged([10.0, 10.0, 10.0]) is False

    # A non-finite window (no finite basin found yet) is not a real plateau.
    convergence = af.MultiStartGradientConvergence(window=3, min_steps=3, atol=1.0)
    assert convergence.check_if_converged([np.inf, np.inf, np.inf]) is False


def test__dict_round_trip__convergence():
    # The convergence settings must survive serialisation so a resumed search
    # keeps the same early-stopping behaviour.
    search = af.MultiStartAdam(
        n_starts=4,
        convergence=af.MultiStartGradientConvergence(
            check_for_convergence=False, window=7, min_steps=9
        ),
    )
    restored = from_dict(to_dict(search))

    assert isinstance(restored, af.MultiStartAdam)
    assert isinstance(restored.convergence, af.MultiStartGradientConvergence)
    assert restored.convergence.check_for_convergence is False
    assert restored.convergence.window == 7
    assert restored.convergence.min_steps == 9


def test__dict_round_trip():
    dictionary = to_dict(af.MultiStartLion(n_starts=7, n_steps=33))
    restored = from_dict(dictionary)

    assert isinstance(restored, af.MultiStartLion)
    assert restored.n_starts == 7
    assert restored.n_steps == 33
    assert restored.optax_method == "lion"
    assert restored.learning_rate == pytest.approx(1.0e-3)


def test__dict_round_trip__prodigy_is_learning_rate_free():
    # The learning-rate-free rule must survive serialisation with lr None so a
    # resumed Prodigy search rebuilds from the rule's own default, not lr=None
    # being coerced to a number.
    dictionary = to_dict(af.MultiStartProdigy(n_starts=5, max_consecutive_nan=4))
    restored = from_dict(dictionary)

    assert isinstance(restored, af.MultiStartProdigy)
    assert restored.n_starts == 5
    assert restored.optax_method == "prodigy"
    assert restored.learning_rate is None


def test__dict_round_trip__resurrect():
    # The restart-on-death flag must survive serialisation so a resumed search
    # keeps redrawing dead starts.
    restored = from_dict(to_dict(af.MultiStartAdam(resurrect=True, n_starts=6)))

    assert isinstance(restored, af.MultiStartAdam)
    assert restored.resurrect is True
    assert restored.n_starts == 6


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
        "n_resurrections": 7,
        "stop_reason": "converged",
    }

    search = af.MultiStartAdam(n_starts=2, n_steps=42, resurrect=True)
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
    # restart-on-death diagnostics flow through to samples_info.
    assert samples.samples_info["resurrect"] is True
    assert samples.samples_info["n_resurrections"] == 7

    # Auto-convergence outcome (phase-2 results contract) flows through to
    # samples_info: the stop reason, the converged flag, the settings, and the
    # global-best figure-of-merit trace (plain floats, JSON-round-trippable).
    assert samples.samples_info["stop_reason"] == "converged"
    assert samples.samples_info["converged"] is True
    assert samples.samples_info["convergence"]["check_for_convergence"] is True
    assert samples.samples_info["convergence"]["window"] == search.convergence.window
    assert (
        samples.samples_info["convergence"]["min_steps"] == search.convergence.min_steps
    )
    assert samples.samples_info["fom_history"] == pytest.approx([-4.0, -8.0, best_fom])
    assert all(isinstance(x, float) for x in samples.samples_info["fom_history"])


def test__samples_info__stop_reason_max_steps_and_legacy_search_internal():
    model = af.Model(example.Gaussian)

    best_params = np.asarray(model.vector_from_unit_vector([0.5] * model.prior_count))
    per_start_params = np.stack([best_params])

    base_internal = {
        "params": per_start_params,
        "best_params": best_params,
        "best_fom": -2.0,
        "total_steps": 300,
        "n_resurrections": 0,
    }

    search = af.MultiStartAdam(n_starts=1, n_steps=300)

    # Ceiling reached: converged is False, stop_reason is "max_steps".
    samples = search.samples_via_internal_from(
        model=model,
        search_internal={
            **base_internal,
            "fom_history": np.asarray([-2.0]),
            "stop_reason": "max_steps",
        },
    )
    assert samples.samples_info["stop_reason"] == "max_steps"
    assert samples.samples_info["converged"] is False

    # Legacy search_internal (pre-auto-convergence: no stop_reason / fom_history)
    # degrades gracefully rather than KeyError-ing.
    legacy = search.samples_via_internal_from(
        model=model, search_internal=base_internal
    )
    assert legacy.samples_info["stop_reason"] is None
    assert legacy.samples_info["converged"] is False
    assert legacy.samples_info["fom_history"] is None


def test__variable_length_zero_weight_nan_rows_are_robust():
    """The multi-start searches write zero-weight, NaN-log-likelihood diagnostic
    rows for the non-best starts, and auto-convergence makes runs variable-length.
    The max-likelihood/posterior accessors must never pick a NaN diagnostic row
    (plain ``np.argmax`` would), and the aggregator summary must not raise for
    runs of differing length."""
    model = af.Model(example.Gaussian)
    best_params = np.asarray(model.vector_from_unit_vector([0.5] * model.prior_count))

    def samples_for(n_starts, total_steps):
        per_start = np.stack(
            [np.asarray(model.vector_from_unit_vector([0.4] * model.prior_count))]
            * n_starts
        )
        search_internal = {
            "params": per_start,
            "best_params": best_params,
            "best_fom": -10.0,
            "fom_history": np.asarray([-4.0] * total_steps),
            "total_steps": total_steps,
            "n_resurrections": 0,
            "stop_reason": "converged",
        }
        return af.MultiStartAdam(
            n_starts=n_starts, n_steps=300
        ).samples_via_internal_from(model=model, search_internal=search_internal)

    for n_starts, total_steps in [(2, 40), (16, 175)]:
        samples = samples_for(n_starts, total_steps)

        # The NaN diagnostic rows are never selected as the best (argmax would).
        assert samples.max_log_likelihood_index == 0
        assert samples.max_log_posterior_index == 0
        assert samples.max_log_likelihood(as_instance=False) == pytest.approx(
            list(best_params)
        )

        # The aggregator summary is computed without an IndexError.
        assert samples.summary() is not None
