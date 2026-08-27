import faulthandler
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# How often a compile that is still running reports that it is still alive.
DEFAULT_HEARTBEAT_SECONDS = 30.0

# How long a first compile may run under CI before it dumps its own traceback,
# when the runner has not told us its own cap. Off by default off-CI: an
# interactive user watching a slow compile does not want a traceback on stderr,
# they want the heartbeat above.
DEFAULT_CI_DUMP_SECONDS = 240.0

# Fraction of the runner's per-script cap at which to dump. The dump is only
# ever useful STRICTLY BEFORE the kill -- see `dump_traceback_seconds`.
DUMP_FRACTION_OF_CAP = 0.8


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

    Under CI the default is derived from `BUILD_SCRIPT_TIMEOUT`, the per-script
    cap the workspace runners and PyAutoHands both enforce. **A dump scheduled
    at or after that cap never happens**: the runner SIGKILLs the process group
    on expiry, and a killed process writes no traceback. The first CI use of
    this watchdog hit exactly that -- a flat 300s default against a 300s smoke
    cap produced heartbeats from 20 stalled runs and not one stack
    (autolens_workspace_test#271). Dumping at a fraction of the cap leaves the
    traceback time to reach stderr before the kill lands.
    """
    if not os.environ.get("CI"):
        default = 0.0
    else:
        cap = _env_seconds("BUILD_SCRIPT_TIMEOUT", 0.0)
        default = cap * DUMP_FRACTION_OF_CAP if cap > 0 else DEFAULT_CI_DUMP_SECONDS

    return _env_seconds("PYAUTOFIT_JAX_COMPILE_DUMP_SECS", default)


# The two halves of a first call. They are separately nameable, and a heartbeat
# that cannot tell them apart reports the wrong one: before this existed a
# stalled run logged `still compiling ... 1770s elapsed` while its stack sat in
# `jax.block_until_ready`, and every marker written off that log calls this an
# "XLA compile stall" (autolens_workspace_test#271).
COMPILING = "compiling"
MATERIALIZING = "materializing"


class _Heartbeat:
    """
    Log that a first call is still running, on an interval, until stopped.

    The message names which half it is in. A phase-blind heartbeat is worse
    than none for the case it exists to diagnose: it is *positive evidence*
    for the wrong diagnosis, and three quarantines were written against
    exactly that.

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
        self.phase = COMPILING
        self.compiled_in = None
        self._stop = threading.Event()
        self._thread = None

    def materializing(self, compiled_in):
        """
        Move to the execution half, having compiled in `compiled_in` seconds.

        `compiled_in` is written BEFORE `phase` so the beat thread can never
        observe `MATERIALIZING` with no compile time to report. Both are plain
        attribute assignments, which are atomic under the GIL; the ordering is
        the whole synchronisation and it is deliberate.
        """
        self.compiled_in = compiled_in
        self.phase = MATERIALIZING

    def _beat(self):
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self.start_time

            if self.phase == COMPILING:
                logger.info(
                    f"JAX jit still compiling {self.description}, "
                    f"{elapsed:.0f}s elapsed..."
                )
            else:
                # The compile time rides on every beat rather than only on the
                # summary line, because a stalled run never reaches a summary
                # line -- the runner SIGKILLs it. Whatever the last beat to
                # reach stderr was, it carries the split.
                logger.info(
                    f"JAX jit compiled {self.description} in "
                    f"{self.compiled_in:.1f}s and is still waiting for the "
                    f"result to materialize, {elapsed:.0f}s elapsed..."
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


def _block_until_ready(result):
    """
    Wait for `result` to materialize, if JAX is installed.

    Separated from the wrapper for two reasons: the execution half is the one
    that hangs, so it has to be substitutable in a test that must not depend on
    JAX being installed; and keeping the broad `except` around the import alone
    stops it swallowing failures from the wait itself.
    """
    try:
        import jax
    except Exception:
        return

    jax.block_until_ready(result)


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

    Both are reported AS THEY FINISH, never in one line at the end. A stalled
    run is killed by the runner's cap and reaches no end, so a summary emitted
    there describes only the runs that did not stall -- which is how the same
    intermittent stall was quarantined five times across the test workspaces
    under the name "XLA compile stall" while its stack sat in
    `jax.block_until_ready`, the *execution* half, with compilation long since
    finished (autolens_workspace_test#271, PyAutoFit#1528).

    So: the compile half is logged the instant `func` returns; the heartbeat
    then names the materialize half it has moved into and repeats the compile
    time on every beat; and a `faulthandler` watchdog dumps a traceback if the
    whole thing overruns. A run killed at any point after tracing has already
    said on stderr which half it was in, and how long the other one took.

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

            # Reported HERE, not once both halves are done. A stalled run never
            # reaches the end of this function, so a split emitted after the
            # wait characterises only the healthy case -- which is why the logs
            # from 20 stalled runs could not say that compilation had in fact
            # finished in ~16s. Emitted before the wait, it says so.
            heartbeat.materializing(compiled_at - start)
            logger.info(
                f"JAX jit compilation of {description}: traced, lowered and "
                f"compiled in {compiled_at - start:.1f} seconds; waiting for "
                f"the result to materialize..."
            )

            # JAX dispatches asynchronously, so `result` may be a future that is
            # not yet materialized. Block once, on this first call only, so the
            # duration logged below is the wait the user actually sat through
            # rather than the dispatch latency.
            _block_until_ready(result)

            materialized_at = time.time()
        finally:
            watchdog.cancel()
            heartbeat.stop()
            state["compiled"] = True

        logger.info(
            f"JAX jit compilation of {description}: result materialized in "
            f"{materialized_at - compiled_at:.1f} seconds."
        )

        logger.info(
            f"JAX jit compilation of {description} complete in "
            f"{time.time() - start:.1f} seconds."
        )

        return result

    return wrapper
