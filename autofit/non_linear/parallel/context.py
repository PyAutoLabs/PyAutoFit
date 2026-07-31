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

    Returns
    -------
    The "fork" multiprocessing context on POSIX platforms other than macOS,
    else the platform default ("spawn" on Windows and macOS).
    """
    if sys.platform != "darwin" and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()
