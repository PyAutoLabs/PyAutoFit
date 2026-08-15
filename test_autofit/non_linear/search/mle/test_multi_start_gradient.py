import inspect

import numpy as np
import pytest

import autofit as af
from autofit import example
from autofit.non_linear.search import abstract_search
from autofit.non_linear.search.mle.multi_start_gradient.search import (
    _chunk_slices,
    batch_memory_model,
    batch_size_within_budget,
)
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
    the unchunked vmap — is asserted in autofit_workspace_test, since the
    library suite is NumPy-only. The chunk bookkeeping the sweep is built on
    (``_chunk_slices``) is pure Python and tested below.
    """
    for cls in (
        af.MultiStartAdam,
        af.MultiStartADABelief,
        af.MultiStartLion,
        af.MultiStartProdigy,
    ):
        assert cls().batch_size is None
        assert cls(batch_size=8).batch_size == 8


@pytest.mark.parametrize(
    "n_rows, batch_size, expected",
    [
        (8, 4, [(0, 4, 0), (4, 8, 0)]),  # exact division
        (10, 4, [(0, 4, 0), (4, 8, 0), (8, 10, 2)]),  # ragged final chunk
        (3, 8, [(0, 3, 5)]),  # fewer rows than one chunk (short broad-start draw)
        (3, 1, [(0, 1, 0), (1, 2, 0), (2, 3, 0)]),  # batch_size=1
        (4, 4, [(0, 4, 0)]),  # single exact chunk
    ],
)
def test__chunk_slices(n_rows, batch_size, expected):
    assert _chunk_slices(n_rows, batch_size) == expected


@pytest.mark.parametrize("n_rows", [1, 3, 4, 7, 16, 47, 48])
@pytest.mark.parametrize("batch_size", [1, 3, 4, 16])
def test__chunk_slices__covers_every_row_at_constant_shape(n_rows, batch_size):
    """Every chunk presents the same ``batch_size`` shape to the compiled
    function (rows + pad), chunks tile ``0..n_rows`` in order, and only the
    final chunk may be padded — the invariants the one-compile sweep rests on.
    """
    slices = _chunk_slices(n_rows, batch_size)

    assert slices[0][0] == 0
    assert slices[-1][1] == n_rows
    for i, (lo, hi, pad) in enumerate(slices):
        assert (hi - lo) + pad == batch_size
        assert pad == 0 or i == len(slices) - 1
        if i:
            assert lo == slices[i - 1][1]


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


@pytest.mark.parametrize(
    "foms, grad_finite, n_value_nan, n_grad_nan",
    [
        # Nothing wrong: every lane finite in value and gradient.
        ([1.0, 2.0, 3.0], [True, True, True], 0, 0),
        # Value-NaN only -- the lane death resurrect already triggers on.
        ([1.0, np.nan, 3.0], [True, True, True], 1, 0),
        ([np.inf, -np.inf, 3.0], [True, True, True], 2, 0),
        # Gradient-NaN only: likelihood defined, but not differentiable. This
        # is the frozen-zombie lane that nothing detected before.
        ([1.0, 2.0, 3.0], [True, False, True], 0, 1),
        # Both failure modes on *different* lanes: counted once each.
        ([1.0, np.nan, 3.0], [True, True, False], 1, 1),
        # Both on the SAME lane: one value-NaN, NOT one of each. A dead lane's
        # gradient is non-finite too, so double-counting here would inflate the
        # gradient counter on every ordinary death and bury the real signal.
        ([1.0, np.nan, 3.0], [True, False, True], 1, 0),
        ([np.nan, np.nan], [False, False], 2, 0),
    ],
)
def test__nan_lane_counts__splits_value_and_gradient_failures(
    foms, grad_finite, n_value_nan, n_grad_nan
):
    alive, value_nan, grad_nan = af.MultiStartAdam._nan_lane_counts(
        foms=np.asarray(foms), grad_finite=np.asarray(grad_finite)
    )

    assert value_nan == n_value_nan
    assert grad_nan == n_grad_nan

    # The alive mask is returned so the caller does not re-test finiteness.
    assert list(alive) == list(np.isfinite(np.asarray(foms)))

    # The two counts partition the lanes they describe: neither can exceed the
    # lane count, and together they never over-count it.
    assert value_nan + grad_nan <= len(foms)


def test__nan_lane_counts__accepts_jax_style_array_outputs():
    """
    ``grad_finite`` arrives from a jitted call as a device array of bools; the
    helper must not assume a NumPy bool dtype (a 0/1 integer array is the
    common surprise).
    """
    alive, value_nan, grad_nan = af.MultiStartAdam._nan_lane_counts(
        foms=np.asarray([1.0, 2.0, np.nan]),
        grad_finite=np.asarray([1, 0, 1]),
    )

    assert value_nan == 1
    assert grad_nan == 1
    assert list(alive) == [True, True, False]


def test__samples_info__nan_lane_step_counters():
    model = af.Model(example.Gaussian)

    best_params = np.asarray(model.vector_from_unit_vector([0.5] * model.prior_count))
    per_start_params = np.stack([best_params])

    search = af.MultiStartAdam(n_starts=1, n_steps=300)

    samples = search.samples_via_internal_from(
        model=model,
        search_internal={
            "params": per_start_params,
            "best_params": best_params,
            "best_fom": -2.0,
            "fom_history": np.asarray([-2.0]),
            "total_steps": 300,
            "n_resurrections": 7,
            "n_value_nan_lane_steps": 7,
            "n_grad_nan_lane_steps": 3,
            "stop_reason": "max_steps",
        },
    )

    assert samples.samples_info["n_value_nan_lane_steps"] == 7
    assert samples.samples_info["n_grad_nan_lane_steps"] == 3

    # A search_internal written before the NaN accounting existed has neither
    # key. Resuming or re-reading one must degrade to zero, not KeyError.
    legacy = search.samples_via_internal_from(
        model=model,
        search_internal={
            "params": per_start_params,
            "best_params": best_params,
            "best_fom": -2.0,
            "fom_history": np.asarray([-2.0]),
            "total_steps": 300,
            "n_resurrections": 0,
        },
    )

    assert legacy.samples_info["n_value_nan_lane_steps"] == 0
    assert legacy.samples_info["n_grad_nan_lane_steps"] == 0


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


@pytest.mark.parametrize(
    "converged, total_steps, is_final",
    [
        (False, 50, False),   # ordinary intermediate boundary -> update here
        (False, 299, False),  # still short of the ceiling
        (False, 300, True),   # ceiling reached
        (False, 301, True),   # overshoot is still terminal
        (True, 50, True),     # early convergence, well short of the ceiling
        (True, 300, True),
    ],
)
def test__is_final_boundary(converged, total_steps, is_final):
    """``_fit`` must not emit a ``perform_update`` at a terminal boundary:
    ``start_resume_fit`` performs the final update unconditionally the moment
    ``_fit`` returns, so one here is duplicated work — and flipping
    ``during_analysis`` does not avoid it, because ``SearchUpdater.update``
    rebuilds the samples, recomputes the summary and re-runs likelihood
    profiling on every call regardless of the flag.

    The rule is tested directly; ``_fit`` needs jax + optax + a JAX-traceable
    ``Analysis`` and cannot be driven from this NumPy-only suite.
    """
    search = af.MultiStartAdam(n_steps=300)

    assert (
        search._is_final_boundary(converged=converged, total_steps=total_steps)
        is is_final
    )


@pytest.mark.parametrize(
    "restored, expected",
    [
        ("converged", "converged"),  # must survive: the loop guard reads it
        ("max_steps", None),         # stale — the run it described is over
        (None, None),
        ("some_future_reason", None),
    ],
)
def test__stop_reason_on_resume(restored, expected):
    """A resumed run inherits the previous run's ``stop_reason``. Keeping a
    ``"max_steps"`` stamps every intermediate checkpoint of a search that is
    still running as finished — which is what raising ``n_steps`` (the
    documented way to extend a budget) produces. ``"converged"`` must survive,
    because the step loop's guard uses it to refuse resuming a converged search
    into further steps."""
    assert af.MultiStartAdam._stop_reason_on_resume(restored) == expected


@pytest.mark.parametrize(
    "total_steps, converged, silence, should_log",
    [
        (1, False, False, True),  # first step: the compile is over, we are moving
        (2, False, False, False),  # off-cadence
        (10, False, False, True),  # on-cadence
        (20, False, False, True),
        (23, False, False, False),
        (23, True, False, True),  # converged off-cadence still reports
        (10, False, True, False),  # silence suppresses a cadence hit
        (1, False, True, False),  # ...and the first step
        (23, True, True, False),  # ...and convergence
    ],
)
def test__should_log_progress(total_steps, converged, silence, should_log):
    """The cadence rule. The first step always logs because that line is what
    tells a user the XLA compile finished and stepping has begun — the longest
    unexplained wait in a real run. The converged step always logs because it
    lands off-cadence more often than not."""
    search = af.MultiStartAdam(n_steps=300, iterations_per_log=10, silence=silence)

    assert (
        search._should_log_progress(total_steps=total_steps, converged=converged)
        is should_log
    )


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test__iterations_per_log__rejects_a_cadence_that_cannot_schedule(bad):
    """Rejected at construction, not clamped: ``total_steps % 0`` raises and a
    negative cadence is never 0, so it would log every step. A user who mistyped
    the knob is better served by an error than by a schedule they never asked
    for."""
    with pytest.raises(ValueError, match="iterations_per_log"):
        af.MultiStartAdam(iterations_per_log=bad)


def test__progress_message__reports_log_posterior_not_the_raw_fom():
    """``best_fom`` is ``-2 * log_posterior`` (what the search minimises), so the
    line must apply the same sign-and-halve conversion as
    ``samples_via_internal_from`` — otherwise the number in the log cannot be
    compared with the number in the results."""
    search = af.MultiStartAdam(n_starts=48, n_steps=300)

    message = search._progress_message(
        total_steps=30,
        best_fom=-63575.6876,
        previous_fom=-63571.4676,
        n_alive=46,
    )

    assert "adam step 30/300" in message
    assert "best log_post 31787.8438" in message
    # a fom drop of 4.22 is a log-posterior gain of 2.11
    assert "gained 2.1100" in message
    assert "alive 46/48" in message
    # no learning-rate-free state was supplied, so the field is omitted entirely
    assert " d " not in message


def test__progress_message__includes_the_prodigy_step_scale_when_present():
    """Prodigy's per-start ``estim_lr`` (its ``d``) is the one genuinely
    learning-rate-free diagnostic, and is otherwise invisible: min/median/max
    across starts says whether it has finished ramping or is still climbing."""
    search = af.MultiStartProdigy(n_starts=4, n_steps=300)

    message = search._progress_message(
        total_steps=10,
        best_fom=-100.0,
        previous_fom=-90.0,
        n_alive=4,
        estim_lr=np.array([1.0e-2, 2.0e-2, 4.0e-2, 8.0e-2]),
    )

    assert "prodigy step 10/300" in message
    assert "d 1.00e-02 / 3.00e-02 / 8.00e-02" in message


@pytest.mark.parametrize(
    "best_fom, previous_fom, expect_gain",
    [
        (np.inf, None, False),  # nothing has found a finite basin yet
        (-100.0, None, False),  # first line: no previous to compare against
        (-100.0, np.inf, False),  # arrival at a first basin is not a "gain"
        (-100.0, -90.0, True),
    ],
)
def test__progress_message__non_finite_and_first_line_edges(
    best_fom, previous_fom, expect_gain
):
    """``inf - x`` would print as ``inf`` and a bare ``inf`` best-fom would print
    as a figure of merit the user cannot act on, so both are reported plainly
    instead."""
    search = af.MultiStartAdam(n_starts=8, n_steps=300)

    message = search._progress_message(
        total_steps=10,
        best_fom=best_fom,
        previous_fom=previous_fom,
        n_alive=8,
    )

    assert ("gained" in message) is expect_gain
    assert "inf" not in message

    if not np.isfinite(best_fom):
        assert "no finite basin yet" in message


@pytest.mark.parametrize(
    "batch_size, expected",
    [
        (None, "all 16 starts at once"),
        (4, "batches of 4 starts"),
    ],
)
def test__compile_message__names_what_is_being_compiled(batch_size, expected):
    """Two distinct compiles block a fresh run — the single-point objective
    ``_broad_starts`` calls per draw, and the vmapped/chunked one the step loop
    calls. Neither is reachable by the ``Fitness._jit`` / ``_vmap`` / ``_grad``
    notices, because this search builds its transforms straight off
    ``fitness.call``."""
    search = af.MultiStartProdigy(n_starts=16, batch_size=batch_size)

    single = search._compile_message(batched=False)
    assert "single-point objective" in single
    assert "16 broad starts" in single
    assert "could take seconds or minutes" in single

    batched = search._compile_message(batched=True)
    assert "vmapped objective" in batched
    assert expected in batched
    assert "could take seconds or minutes" in batched


def test__dict_round_trip__iterations_per_log():
    """The cadence must survive serialisation so a resumed search keeps
    reporting at the rate the user chose."""
    restored = from_dict(to_dict(af.MultiStartProdigy(iterations_per_log=25)))

    assert isinstance(restored, af.MultiStartProdigy)
    assert restored.iterations_per_log == 25


def test__fit_wires_the_progress_and_compile_seams():
    """Wiring guard for the three progress seams, which are otherwise tested in
    isolation and would keep passing if ``_fit`` stopped calling them.

    Necessarily a source-level check — ``_fit`` cannot run from this suite.
    """
    source = " ".join(inspect.getsource(af.MultiStartAdam._fit).split())

    # the cadence gate and the line it guards
    assert (
        "if self._should_log_progress( total_steps=total_steps, converged=converged )"
        in source
    )
    assert "self._progress_message(" in source

    # both compile notices, each emitted before the call that triggers its compile
    assert "self._compile_message(batched=False)" in source
    assert "self._compile_message(batched=True)" in source

    # the step scale is read from the optimizer state, not recomputed
    assert 'optax.tree_utils.tree_get(opt_state, "estim_lr")' in source

    # the converged step must still be able to report before the loop breaks:
    # setting `converged` and breaking must not be fused back together.
    assert "converged = True break" not in source


def test__fit_uses_both_seams_and_keeps_the_converged_loop_guard():
    """Wiring guard for the two seams above, which are otherwise tested in
    isolation and would keep passing if ``_fit`` stopped calling them.

    Necessarily a source-level check — ``_fit`` cannot run from this suite. It
    pins the call sites and the loop condition; the *behaviour* of each seam is
    pinned by the parametrised tests above, so a semantically-equivalent rewrite
    of a seam is caught there rather than here.
    """
    source = " ".join(inspect.getsource(af.MultiStartAdam._fit).split())

    assert "if not self._is_final_boundary(" in source
    assert "stop_reason = self._stop_reason_on_resume(stop_reason)" in source
    assert 'while total_steps < self.n_steps and stop_reason != "converged":' in source
    # the form that ran the whole final pass twice must not come back
    assert "during_analysis=not is_final" not in source

    # The framework's single final update is still there and unconditional —
    # without it this change would remove the final update rather than
    # de-duplicate it. ``start_resume_fit`` is decorated without
    # ``functools.wraps``, so its body is sliced out of the module source.
    module = inspect.getsource(abstract_search)
    body = module[module.index("    def start_resume_fit(") :]
    body = body[: body.index("\n    def ", 1)]
    assert "during_analysis=False" in body


def test__samples_info__reports_a_cleared_stop_reason_as_unfinished():
    """The observable half of the fix: a checkpoint written mid-run carries no
    stop reason, so nothing downstream reads it as a finished search."""
    model = af.Model(example.Gaussian)
    best_params = np.asarray(model.vector_from_unit_vector([0.5] * model.prior_count))

    search = af.MultiStartAdam(n_starts=1, n_steps=600)
    samples = search.samples_via_internal_from(
        model=model,
        search_internal={
            "params": np.stack([best_params]),
            "best_params": best_params,
            "best_fom": -2.0,
            "fom_history": np.asarray([-2.0] * 300),
            "total_steps": 300,
            "n_resurrections": 0,
            # what an intermediate checkpoint of a resumed, still-running search
            # now writes — previously this said "max_steps".
            "stop_reason": None,
        },
    )

    assert samples.samples_info["stop_reason"] is None
    assert samples.samples_info["converged"] is False


# --- Batched-jvp memory projection (PyAutoFit#1452) ---------------------------
#
# Same contract as the _chunk_slices tests above: the projection arithmetic is
# pure Python and lives here, while the JAX measurement it consumes
# (`Analysis.batched_memory_bytes`) is exercised in autofit_workspace_test.

GB = 1024 ** 3

# The 2026-07-30 release failure: XLA reported 85,898,814,480 bytes for a
# 48-start interferometer jvp, i.e. 1,789,558,635 bytes per start.
INCIDENT_PER_START = 1_789_558_635


def test__batch_memory_model__recovers_fixed_and_per_start():
    fixed = 2 * GB
    at_1 = fixed + INCIDENT_PER_START
    at_2 = fixed + 2 * INCIDENT_PER_START

    assert batch_memory_model(at_1, at_2) == (fixed, INCIDENT_PER_START)


@pytest.mark.parametrize(
    "at_1, at_2",
    [
        (10, 4),  # non-monotonic measurements
        (10, 10),  # identical measurements
    ],
)
def test__batch_memory_model__never_yields_a_negative_slope(at_1, at_2):
    # A negative slope would invert the budget arithmetic and suggest an
    # absurdly large batch, which is worse than saying nothing.
    fixed, per_start = batch_memory_model(at_1, at_2)
    assert per_start == 0
    assert fixed >= 0


def test__batch_size_within_budget__suggests_the_largest_batch_that_fits():
    fixed, per_start = 2 * GB, INCIDENT_PER_START

    suggested = batch_size_within_budget(fixed, per_start, 48, 16 * GB)

    assert 1 <= suggested < 48
    assert fixed + suggested * per_start <= 16 * GB
    assert fixed + (suggested + 1) * per_start > 16 * GB


def test__batch_size_within_budget__none_when_the_full_batch_fits():
    # None means "leave batch_size=None" — the single-vmap fast path is kept.
    assert batch_size_within_budget(2 * GB, INCIDENT_PER_START, 48, 400 * GB) is None


@pytest.mark.parametrize(
    "fixed, per_start, n_starts, budget, expected",
    [
        (100 * GB, 0, 48, 1 * GB, 1),  # fixed cost alone busts it; tiling cannot help
        (1, 0, 48, 1 * GB, None),  # no per-start cost and it fits
        (0, 1, 48, 47, 47),  # never returns n_starts itself
        (0, 100 * GB, 48, 1 * GB, 1),  # not even one start fits -> still 1
        (2 * GB, INCIDENT_PER_START, 48, 0, None),  # unknown budget
        (2 * GB, INCIDENT_PER_START, 1, 1, None),  # nothing to tile
    ],
)
def test__batch_size_within_budget__degenerate_inputs(
    fixed, per_start, n_starts, budget, expected
):
    assert batch_size_within_budget(fixed, per_start, n_starts, budget) == expected
