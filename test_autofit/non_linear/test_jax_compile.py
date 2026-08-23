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

    # One summary line covering both waits cannot say which half a stalled run
    # is stuck in -- the ambiguity this whole module exists to remove.
    assert any(
        "traced, lowered and compiled in" in m and "result materialized in" in m
        for m in messages
    )


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


def test_a_malformed_interval_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", "soon")
    assert jax_compile.heartbeat_seconds() == jax_compile.DEFAULT_HEARTBEAT_SECONDS

    # A negative interval would busy-loop the heartbeat thread.
    monkeypatch.setenv("PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", "-5")
    assert jax_compile.heartbeat_seconds() == 0.0
