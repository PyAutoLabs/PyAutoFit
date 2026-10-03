import multiprocessing
import sys


def fork_context():
    """
    The multiprocessing context used for every pool and process PyAutoFit
    creates.

    Python 3.14 changed the default start method on Linux (and other POSIX
    platforms except macOS) from "fork" to "forkserver". PyAutoFit's
    parallelism relies on fork semantics: fitness functions, models and the
    state of user scripts (which run at module level without an
    ``if __name__ == "__main__"`` guard) are inherited by worker processes
    rather than pickled or re-imported. Under "forkserver" workers receive
    corrupted model instances
    (https://github.com/PyAutoLabs/PyAutoFit/issues/1437), so the "fork"
    context is pinned explicitly.

    macOS is deliberately excluded: its default has been "spawn" since Python
    3.8 and forking a process whose threads hold ObjC/CoreFoundation state can
    abort, so pinning "fork" there would introduce new behaviour rather than
    restore old behaviour. This helper reproduces the pre-3.14 default on
    every platform.

    On macOS and Windows the ``multiprocessing`` module itself is returned
    rather than ``multiprocessing.get_context()``. Both expose the same
    ``Process`` / ``Queue`` / ``Pool`` API, but calling ``get_context()`` with
    no argument *fixes* the interpreter's default start method, and this
    function runs at import time (``class Process(fork_context().Process)``).
    That made ``multiprocessing.set_start_method("fork")`` raise
    ``RuntimeError: context has already been set`` in any script or test
    suite that imports autofit before choosing its start method (the autofit
    test suite does exactly this on macOS in ``test_autofit/conftest.py``).
    The module-level API defers the choice until a process actually starts,
    which is the pre-3.14 behaviour this helper exists to preserve.

    Returns
    -------
    The "fork" multiprocessing context on POSIX platforms other than macOS,
    else the ``multiprocessing`` module, which dispatches to whichever start
    method is current when a process is created ("spawn" by default on
    Windows and macOS).
    """
    if sys.platform != "darwin" and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing
