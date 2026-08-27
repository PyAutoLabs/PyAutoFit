import logging
import threading
import time
from unittest import mock

from autofit.non_linear import jax_compile
from autofit.non_linear.jax_compile import log_on_first_compile

# ``log_on_first_compile`` wraps the jax.jit / jax.vmap / jax.grad callables so the
# "this is compiling" line is emitted where the user actually waits -- on the first
# call -- rather than at wrapper-construction time, where the old logging sat and
# reported ~0 seconds immediately before an unexplained multi-minute pause.
#
# Tested against plain callables: the contract is about *when* it logs, which does
# not need JAX installed (the library suite is numpy-only).

LOGGER = "autofit.non_linear.jax_compile"


def test_compile_message_is_logged_on_the_first_call_not_at_wrap_time(caplog):
    calls = []
    wrapped = log_on_first_compile(
        lambda x: calls.append(x) or x * 2, "likelihood function"
    )

    # Wrapping alone must say nothing -- this is the bug being fixed.
    assert caplog.records == []
    assert calls == []

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert wrapped(3) == 6

    messages = [record.getMessage() for record in caplog.records]

    assert (
        "JAX jit compiling likelihood function, could take seconds or minutes..."
        in messages
    )
    assert any("compilation of likelihood function complete in" in m for m in messages)


def test_compile_message_is_logged_only_once(caplog):
    wrapped = log_on_first_compile(lambda x: x * 2, "likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert wrapped(1) == 2
        assert wrapped(2) == 4
        assert wrapped(3) == 6

    compiling = [
        record
        for record in caplog.records
        if "could take seconds or minutes" in record.getMessage()
    ]
    # Every later evaluation reuses the compiled trace, so announcing a compile
    # again would be false -- and there are millions of evaluations per fit.
    assert len(compiling) == 1


def test_wrapped_function_passes_through_args_kwargs_and_return_value():
    wrapped = log_on_first_compile(
        lambda a, b, scale=1: (a + b) * scale, "likelihood function"
    )

    assert wrapped(2, 3, scale=10) == 50
    assert wrapped(2, 3) == 5


def test_a_raising_first_call_propagates_and_does_not_claim_completion(caplog):
    def boom(_):
        raise ValueError("compile failed")

    wrapped = log_on_first_compile(boom, "likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        try:
            wrapped(1)
        except ValueError:
            pass

    messages = [record.getMessage() for record in caplog.records]

    # The announcement is fine -- the compile really was attempted. Claiming it
    # "completed in N seconds" after it raised would not be.
    assert any("could take seconds or minutes" in m for m in messages)
    assert not any("complete in" in m for m in messages)


# The stall this instrumentation exists for: a first compile that emits the
# "compiling..." line and then nothing at all until a CI cap kills it. The tests
# below pin the three things that turn that silence into evidence -- a
# heartbeat, a faulthandler dump, and a compile-vs-execute timing split -- and,
# as above, none of them needs JAX installed.


def test_a_slow_first_call_reports_that_it_is_still_alive(caplog):
    wrapped = log_on_first_compile(lambda: time.sleep(0.25), "likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        with mock.patch.object(jax_compile, "heartbeat_seconds", return_value=0.05):
            wrapped()

    beats = [
        record
        for record in caplog.records
        if "still compiling likelihood function" in record.getMessage()
    ]

    # Without these the log cannot distinguish a compile that is slow from one
    # that has stopped, which is the whole diagnostic gap.
    assert beats


def test_the_heartbeat_stops_when_the_compile_does(caplog):
    wrapped = log_on_first_compile(lambda: None, "likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        with mock.patch.object(jax_compile, "heartbeat_seconds", return_value=0.05):
            wrapped()

        before = len(caplog.records)
        time.sleep(0.2)

        # A heartbeat thread that outlived its compile would log against every
        # later evaluation, and there are millions of those per fit.
        assert len(caplog.records) == before

    assert not any(
        thread.name == "jax-compile-heartbeat" for thread in threading.enumerate()
    )


def test_the_heartbeat_can_be_disabled(caplog):
    wrapped = log_on_first_compile(lambda: time.sleep(0.15), "likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        with mock.patch.object(jax_compile, "heartbeat_seconds", return_value=0.0):
            wrapped()

    assert not any(
        "still compiling" in record.getMessage() for record in caplog.records
    )


def test_the_traceback_dump_is_armed_for_the_compile_and_cancelled_after():
    wrapped = log_on_first_compile(lambda: None, "likelihood function")

    with mock.patch.object(jax_compile, "dump_traceback_seconds", return_value=120.0):
        with mock.patch.object(jax_compile.faulthandler, "dump_traceback_later") as arm:
            with mock.patch.object(
                jax_compile.faulthandler, "cancel_dump_traceback_later"
            ) as cancel:
                wrapped()

    arm.assert_called_once_with(120.0, repeat=True, exit=False)
    cancel.assert_called_once()


def test_the_traceback_dump_is_cancelled_even_when_the_compile_raises():
    def boom():
        raise ValueError("compile failed")

    wrapped = log_on_first_compile(boom, "likelihood function")

    with mock.patch.object(jax_compile, "dump_traceback_seconds", return_value=120.0):
        with mock.patch.object(jax_compile.faulthandler, "dump_traceback_later"):
            with mock.patch.object(
                jax_compile.faulthandler, "cancel_dump_traceback_later"
            ) as cancel:
                try:
                    wrapped()
                except ValueError:
                    pass

    # A timer left armed would fire against whatever ran next and report an
    # unrelated stack as the stalled one.
    cancel.assert_called_once()


def test_a_disabled_traceback_dump_arms_nothing_and_cancels_nothing():
    wrapped = log_on_first_compile(lambda: None, "likelihood function")

    with mock.patch.object(jax_compile, "dump_traceback_seconds", return_value=0.0):
        with mock.patch.object(jax_compile.faulthandler, "dump_traceback_later") as arm:
            with mock.patch.object(
                jax_compile.faulthandler, "cancel_dump_traceback_later"
            ) as cancel:
                wrapped()

    arm.assert_not_called()
    # Cancelling unconditionally would disarm a faulthandler timer the caller
    # set for their own reasons.
    cancel.assert_not_called()


def test_a_dump_that_cannot_be_armed_does_not_break_the_compile(caplog):
    wrapped = log_on_first_compile(lambda: "fit", "likelihood function")

    with mock.patch.object(jax_compile, "dump_traceback_seconds", return_value=120.0):
        with mock.patch.object(
            jax_compile.faulthandler,
            "dump_traceback_later",
            side_effect=RuntimeError("sys.stderr has no file descriptor"),
        ):
            with caplog.at_level(logging.INFO, logger=LOGGER):
                assert wrapped() == "fit"

    # A capturing harness can leave stderr without a real fd. Losing the dump is
    # a lost diagnostic, never a failed fit.
    assert any("complete in" in record.getMessage() for record in caplog.records)


def test_the_compile_wait_and_the_execution_wait_are_reported_separately(caplog):
    wrapped = log_on_first_compile(lambda: None, "vectorized (vmap) likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        wrapped()

    messages = [record.getMessage() for record in caplog.records]

    compiled = [i for i, m in enumerate(messages) if "traced, lowered and compiled in" in m]
    materialized = [i for i, m in enumerate(messages) if "result materialized in" in m]

    assert compiled and materialized

    # Two lines, in order -- not one line carrying both. The order is the
    # assertion that matters: it is what lets a run that is killed between them
    # be read as "compiled, then stuck in execution".
    assert compiled[0] < materialized[0]


def test_the_compile_half_is_reported_before_the_execution_half_is_waited_on(caplog):
    """
    The defect this pins: the split used to be logged once BOTH halves finished,
    so a stalled run -- which never finishes the second -- reported neither, and
    twenty of them could not say that compilation had in fact completed.
    """
    blocked = threading.Event()
    reached = threading.Event()

    def never_returns(_result):
        reached.set()
        blocked.wait(timeout=10)

    wrapped = log_on_first_compile(lambda: None, "vectorized (vmap) likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        with mock.patch.object(jax_compile, "_block_until_ready", never_returns):
            worker = threading.Thread(target=wrapped, daemon=True)
            worker.start()
            assert reached.wait(timeout=5)

            # Still inside the execution half, exactly where a stalled CI run
            # sits when the runner kills it.
            messages = [record.getMessage() for record in caplog.records]
            assert any("traced, lowered and compiled in" in m for m in messages)
            assert not any("result materialized in" in m for m in messages)

            blocked.set()
            worker.join(timeout=5)

    assert not worker.is_alive()


def test_the_heartbeat_names_the_half_it_is_actually_in(caplog):
    """
    A phase-blind heartbeat logged `still compiling ... 1770s elapsed` while the
    process sat in `jax.block_until_ready`. Every marker written off that log
    calls this an "XLA compile stall"; it is not one.
    """
    blocked = threading.Event()
    reached = threading.Event()

    def never_returns(_result):
        reached.set()
        blocked.wait(timeout=10)

    wrapped = log_on_first_compile(lambda: None, "vectorized (vmap) likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        with mock.patch.object(jax_compile, "heartbeat_seconds", return_value=0.05):
            with mock.patch.object(jax_compile, "_block_until_ready", never_returns):
                worker = threading.Thread(target=wrapped, daemon=True)
                worker.start()
                assert reached.wait(timeout=5)
                time.sleep(0.25)

                beats = [
                    m
                    for m in (r.getMessage() for r in caplog.records)
                    if "elapsed..." in m
                ]

                blocked.set()
                worker.join(timeout=5)

    assert beats

    # Every beat emitted while parked in the execution half must say so, and
    # none may still claim to be compiling.
    stuck = [m for m in beats if "waiting for the result to materialize" in m]
    assert stuck
    assert not any("still compiling" in m for m in beats[-len(stuck):])


def test_a_heartbeat_in_the_execution_half_carries_the_compile_time(caplog):
    """
    A stalled run is SIGKILLed and reaches no summary line, so the compile/execute
    split has to ride on the beats themselves to survive to the captured tail.
    """
    blocked = threading.Event()
    reached = threading.Event()

    def never_returns(_result):
        reached.set()
        blocked.wait(timeout=10)

    def slow_compile():
        time.sleep(0.2)

    wrapped = log_on_first_compile(slow_compile, "vectorized (vmap) likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        with mock.patch.object(jax_compile, "heartbeat_seconds", return_value=0.05):
            with mock.patch.object(jax_compile, "_block_until_ready", never_returns):
                worker = threading.Thread(target=wrapped, daemon=True)
                worker.start()
                assert reached.wait(timeout=5)
                time.sleep(0.2)

                blocked.set()
                worker.join(timeout=5)

    # "is still waiting" is the beat; the wrapper's own hand-off line says
    # "waiting" without the "is still", and carries no elapsed counter.
    stuck = [
        m
        for m in (r.getMessage() for r in caplog.records)
        if "is still waiting for the result to materialize" in m
    ]

    assert stuck
    assert all("compiled vectorized (vmap) likelihood function in" in m for m in stuck)


def test_the_execution_half_is_skipped_without_jax_and_does_not_raise():
    """
    `_block_until_ready` exists to be substitutable, but its real body must stay
    a no-op when JAX is absent -- the library suite is numpy-only, and losing the
    wait is a lost diagnostic, never a failed fit.
    """
    with mock.patch.dict("sys.modules", {"jax": None}):
        jax_compile._block_until_ready(object())


def test_the_split_is_not_reported_when_the_first_call_raises(caplog):
    def boom():
        raise ValueError("compile failed")

    wrapped = log_on_first_compile(boom, "likelihood function")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        try:
            wrapped()
        except ValueError:
            pass

    messages = [record.getMessage() for record in caplog.records]

    assert not any("traced, lowered and compiled in" in m for m in messages)


def test_the_heartbeat_interval_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", raising=False)
    assert jax_compile.heartbeat_seconds() == jax_compile.DEFAULT_HEARTBEAT_SECONDS

    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", "5")
    assert jax_compile.heartbeat_seconds() == 5.0

    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", "0")
    assert jax_compile.heartbeat_seconds() == 0.0


def test_the_traceback_dump_defaults_on_under_ci_and_off_elsewhere(monkeypatch):
    monkeypatch.delenv("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", raising=False)
    monkeypatch.delenv("BUILD_SCRIPT_TIMEOUT", raising=False)

    monkeypatch.delenv("CI", raising=False)
    assert jax_compile.dump_traceback_seconds() == 0.0

    # The point of the default: the next CI stall dumps a stack with no
    # workspace script and no CI runner edited.
    monkeypatch.setenv("CI", "true")
    assert jax_compile.dump_traceback_seconds() == jax_compile.DEFAULT_CI_DUMP_SECONDS

    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", "60")
    assert jax_compile.dump_traceback_seconds() == 60.0

    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", "0")
    assert jax_compile.dump_traceback_seconds() == 0.0


def test_the_dump_lands_strictly_before_the_runner_kills_the_process(monkeypatch):
    monkeypatch.delenv("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", raising=False)
    monkeypatch.setenv("CI", "true")

    # The defect this pins (autolens_workspace_test#271): a flat 300s default
    # against a 300s smoke cap meant the runner's SIGKILL always beat the dump,
    # so 20 stalled CI runs produced heartbeats and not one traceback. A killed
    # process writes no stack, so the threshold MUST be under the cap.
    for cap in ("300", "1800", "60"):
        monkeypatch.setenv("BUILD_SCRIPT_TIMEOUT", cap)
        assert jax_compile.dump_traceback_seconds() < float(cap)

    monkeypatch.setenv("BUILD_SCRIPT_TIMEOUT", "300")
    assert jax_compile.dump_traceback_seconds() == 240.0

    monkeypatch.setenv("BUILD_SCRIPT_TIMEOUT", "1800")
    assert jax_compile.dump_traceback_seconds() == 1440.0


def test_an_unusable_runner_cap_falls_back_to_the_flat_ci_default(monkeypatch):
    monkeypatch.delenv("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", raising=False)
    monkeypatch.setenv("CI", "true")

    # No cap advertised, or a meaningless one: there is nothing to derive from,
    # so use the flat default rather than computing a fraction of zero (which
    # would silently disable the dump).
    for cap in ("", "0", "not-a-number"):
        monkeypatch.setenv("BUILD_SCRIPT_TIMEOUT", cap)
        assert jax_compile.dump_traceback_seconds() == jax_compile.DEFAULT_CI_DUMP_SECONDS


def test_an_explicit_dump_threshold_still_wins_over_the_derived_one(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("BUILD_SCRIPT_TIMEOUT", "1800")
    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", "90")

    assert jax_compile.dump_traceback_seconds() == 90.0


def test_a_malformed_interval_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", "soon")
    assert jax_compile.heartbeat_seconds() == jax_compile.DEFAULT_HEARTBEAT_SECONDS

    # A negative interval would busy-loop the heartbeat thread.
    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", "-5")
    assert jax_compile.heartbeat_seconds() == 0.0
