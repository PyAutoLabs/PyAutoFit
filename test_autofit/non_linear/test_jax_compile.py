import logging

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
