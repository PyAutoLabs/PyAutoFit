import faulthandler
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# How often a compile that is still running reports that it is still alive.
DEFAULT_HEARTBEAT_SECONDS = 30.0

# How long a first compile may run under CI before it dumps its own traceback.
# Off by default off-CI: an interactive user watching a slow compile does not
# want a traceback on stderr, they want the heartbeat above.
DEFAULT_CI_DUMP_SECONDS = 300.0


def _env_seconds(name, default):
    """
    Read a non-negative number of seconds from the environment.

    A malformed value warns and falls back rather than raising: every consumer
    of this module is diagnostics wrapped around a model-fit, and diagnostics
    must never be the reason a fit dies.
    """
    value = os.environ.get(name)

    if value is None:
        return default

    try:
        seconds = float(value)
    except ValueError:
        logger.warning(
            f"Ignoring {name}={value!r}: expected a number of seconds. "
            f"Using {default} instead."
        )
        return default

    return max(seconds, 0.0)


def heartbeat_seconds():
    """
    The interval between "still compiling" log lines, from
    `PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS`. `0` disables the heartbeat.
    """
    return _env_seconds(
        "PYAUTOFIT_JAX_COMPILE_HEARTBEAT_SECS", DEFAULT_HEARTBEAT_SECONDS
    )


def dump_traceback_seconds():
    """
    How long a first compile may run before dumping a traceback, from
    `PYAUTOFIT_JAX_COMPILE_DUMP_SECS`. `0` disables the dump.

    The default is on under CI and off elsewhere. That is what makes a stalled
    CI run self-diagnosing without editing a single workspace script or CI
    runner -- the alternative, threading an environment variable through each
    workspace's `config/build/env_vars_*.yaml`, is more repos touched for the
    same effect, and the workspace scripts are user-facing documentation.
    """
    default = DEFAULT_CI_DUMP_SECONDS if os.environ.get("CI") else 0.0
    return _env_seconds("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", default)


class _Heartbeat:
    """
    Log that a compile is still running, on an interval, until stopped.

    Without this a long compile is indistinguishable from a stopped one: the
    process emits the "compiling..." line and then nothing at all until it
    either finishes or is killed. Several CI stalls were quarantined without
    diagnosis for exactly that reason -- there was no evidence to diagnose from.

    The thread is a daemon and waits on an `Event`, so it neither holds the
    interpreter open at exit nor sleeps past `stop`.
    """

    def __init__(self, description, start, interval=None):
        self.description = description
        self.start_time = start
        self.interval = heartbeat_seconds() if interval is None else interval
        self._stop = threading.Event()
        self._thread = None

    def _beat(self):
        while not self._stop.wait(self.interval):
            logger.info(
                f"JAX jit still compiling {self.description}, "
                f"{time.time() - self.start_time:.0f}s elapsed..."
            )

    def start(self):
        if self.interval <= 0:
            return

        thread = threading.Thread(
            target=self._beat,
            name="jax-compile-heartbeat",
            daemon=True,
        )

        try:
            thread.start()
        except RuntimeError as e:
            # Out of threads. Losing the heartbeat is a lost diagnostic; letting
            # it raise here would make this instrumentation the reason a fit
            # died, which is the opposite of the point.
            logger.debug(f"Could not start the JAX compile heartbeat: {e}")
            return

        self._thread = thread

    def stop(self):
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=self.interval)
            self._thread = None


class _TracebackWatchdog:
    """
    Arm `faulthandler` so a compile that overruns dumps its own traceback.

    A stalled run used to be killed by the runner's cap having emitted nothing
    since the "compiling..." line, so there was no way to tell whether it was
    inside XLA, inside execution, or blocked on a lock. This makes the process
    say so itself, before anything kills it.

    Note the limitation: a Python traceback taken during XLA compilation parks
    at the pybind boundary and will not show XLA internals. It still separates
    *in compile* from *in execution* from *blocked on a Python-level lock* --
    for instance the persistent compilation cache -- which is the distinction
    that matters here.

    `faulthandler`'s timer is process-global, so nested or concurrent first
    compiles share one. That is benign for the callers here, which compile on
    the main thread one at a time.
    """

    def __init__(self, seconds=None):
        self.seconds = dump_traceback_seconds() if seconds is None else seconds
        self._armed = False

    def arm(self):
        if self.seconds <= 0:
            return

        try:
            faulthandler.dump_traceback_later(
                self.seconds, repeat=True, exit=False
            )
        except Exception as e:
            # `dump_traceback_later` needs a stderr with a real file descriptor,
            # which a capturing harness may not provide. Losing the dump is a
            # lost diagnostic, never a failed fit.
            logger.debug(f"Could not arm the JAX compile traceback dump: {e}")
            return

        self._armed = True

    def cancel(self):
        # Only cancel a timer this object armed -- cancelling unconditionally
        # would silently disarm a `faulthandler` timer set by the caller.
        if not self._armed:
            return

        faulthandler.cancel_dump_traceback_later()
        self._armed = False


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

    That first call contains two different waits, and this reports them
    separately:

    1. `func(*args, **kwargs)` -- tracing, lowering and XLA compilation;
    2. `jax.block_until_ready(result)` -- execution, since JAX dispatches
       asynchronously.

    One summary line covering both cannot say which half a stalled run is stuck
    in, and that ambiguity is why the same intermittent stall has been
    quarantined three times across the test workspaces without a diagnosis. A
    heartbeat reports liveness while either half is in flight, and a
    `faulthandler` watchdog dumps a traceback if the whole thing overruns.

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

        heartbeat = _Heartbeat(description, start)
        watchdog = _TracebackWatchdog()

        heartbeat.start()
        watchdog.arm()

        try:
            result = func(*args, **kwargs)
            compiled_at = time.time()

            # JAX dispatches asynchronously, so `result` may be a future that is
            # not yet materialized. Block once, on this first call only, so the
            # duration logged below is the wait the user actually sat through
            # rather than the dispatch latency.
            try:
                import jax

                jax.block_until_ready(result)
            except Exception:
                pass

            materialized_at = time.time()
        finally:
            watchdog.cancel()
            heartbeat.stop()
            state["compiled"] = True

        logger.info(
            f"JAX jit compilation of {description}: traced, lowered and "
            f"compiled in {compiled_at - start:.1f} seconds, result "
            f"materialized in {materialized_at - compiled_at:.1f} seconds."
        )

        logger.info(
            f"JAX jit compilation of {description} complete in "
            f"{time.time() - start:.1f} seconds."
        )

        return result

    return wrapper
