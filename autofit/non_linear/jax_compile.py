import logging
import time

logger = logging.getLogger(__name__)


def log_on_first_compile(func, description):
    """
    Wrap `func` so that its first invocation reports the JAX compilation it
    triggers.

    `jax.jit`, `jax.vmap` and `jax.grad` all return immediately -- they only
    build a wrapper. Tracing, lowering and XLA compilation happen on the *first
    call* to the function they return, and that call is where the user waits:
    seconds for a small model, minutes for a large one. Logging at
    wrapper-construction time therefore announced a wait that had not started
    yet and reported a duration of roughly zero, leaving the real wait that
    followed unexplained.

    The one-shot flag lives in a closure rather than on an object because the
    callers cache these wrappers on attributes that are stripped for pickling; a
    fresh process rebuilds the wrapper and genuinely does recompile, so it
    should log again.

    Parameters
    ----------
    func
        The JAX-transformed callable whose first call triggers compilation.
    description
        What is being compiled, as it should read mid-sentence in the log.

    Returns
    -------
    A callable with the same signature as `func`.
    """
    state = {"compiled": False}

    def wrapper(*args, **kwargs):
        if state["compiled"]:
            return func(*args, **kwargs)

        logger.info(
            f"JAX jit compiling {description}, could take seconds or minutes..."
        )
        start = time.time()

        try:
            result = func(*args, **kwargs)

            # JAX dispatches asynchronously, so `result` may be a future that is
            # not yet materialized. Block once, on this first call only, so the
            # duration logged below is the wait the user actually sat through
            # rather than the dispatch latency.
            try:
                import jax

                jax.block_until_ready(result)
            except Exception:
                pass
        finally:
            state["compiled"] = True

        logger.info(
            f"JAX jit compilation of {description} complete in "
            f"{time.time() - start:.1f} seconds."
        )

        return result

    return wrapper
