"""
SIGUSR1 / SIGTERM handling for a fit's status record.

A batch scheduler warns a job before its time limit with a signal (Slurm
``--signal=B:USR1@<seconds>``) and stops it with SIGTERM. On either the handler flushes
the search's checkpoint (``FitStatus.flush_checkpoint``) and writes
``timeout_pending``; on SIGTERM it then exits with status 143 (128 + SIGTERM), raised as
``SystemExit`` so the fit's ``finally`` blocks (pool shutdown, handler restore) run.

Handlers are installed only in the main thread of the top-level process (never in a
pool worker or a thread), and the previous handlers are restored when the fit ends. A
forked worker inherits the handlers; there they hand the signal to the previous
handler (or the default action) unchanged, so workers behave exactly as they did
without a status record.
"""
import logging
import multiprocessing
import os
import signal
import sys
import threading
from typing import Callable

logger = logging.getLogger(__name__)

EXIT_SIGTERM = 128 + 15


def _signals():
    names = ("SIGUSR1", "SIGTERM")
    return [getattr(signal, name) for name in names if hasattr(signal, name)]


def can_install() -> bool:
    """
    Whether this is the main thread of a top-level process on a platform with the
    signals (``SIGUSR1`` does not exist on Windows).
    """
    return (
        hasattr(signal, "SIGUSR1")
        and threading.current_thread() is threading.main_thread()
        and multiprocessing.parent_process() is None
    )


def _pass_on(signum, frame, previous):
    """
    Hand ``signum`` to ``previous``: call it if it is a handler, else restore it and
    re-raise the signal so its default action (or ignore) applies.
    """
    if callable(previous):
        previous(signum, frame)
        return
    signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def install(status) -> Callable[[], None]:
    """
    Install the SIGUSR1 / SIGTERM handlers of ``status`` (a ``FitStatus``) and return
    the function that restores the previous ones; a no-op when ``can_install`` is
    false.
    """
    if not can_install():
        return lambda: None

    owner = os.getpid()
    previous = {}

    def handler(signum, frame):
        if os.getpid() != owner:
            _pass_on(signum, frame, previous.get(signum))
            return

        name = signal.Signals(signum).name
        logger.info(f"{name} received: flushing the checkpoint (timeout_pending).")
        status.timeout_pending(name)

        if signum == signal.SIGTERM:
            sys.exit(EXIT_SIGTERM)

        if callable(previous.get(signum)):
            previous[signum](signum, frame)

    for signum in _signals():
        try:
            previous[signum] = signal.signal(signum, handler)
        except (ValueError, OSError):
            pass

    def restore():
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, handler_before in previous.items():
            try:
                signal.signal(
                    signum,
                    handler_before if handler_before is not None else signal.SIG_DFL,
                )
            except (ValueError, OSError):
                pass

    return restore
