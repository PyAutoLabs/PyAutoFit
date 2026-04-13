import copy
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


def _convert_jax_to_numpy(instance):
    """
    Return a deep copy of *instance* with every JAX array replaced by a
    NumPy array.  Plain NumPy values and non-array attributes are left
    unchanged.

    This is used so that the background visualisation thread never
    touches JAX / GPU state, which is not thread-safe.
    """
    instance = copy.deepcopy(instance)

    for attr in vars(instance):
        value = getattr(instance, attr)
        if hasattr(value, "device"):
            setattr(instance, attr, np.asarray(value))

    return instance


class BackgroundQuickUpdate:
    """
    Runs ``analysis.perform_quick_update`` on a background daemon thread so
    that the sampler is not blocked while matplotlib renders and saves plots.

    Uses a **latest-only** pattern: if a new best-fit arrives before the
    previous visualisation finishes, the stale request is silently replaced.

    Parameters
    ----------
    convert_jax
        If ``True``, JAX arrays on the model instance are converted to
        NumPy before handing them to the worker thread.
    """

    def __init__(self, convert_jax: bool = False):
        self._convert_jax = convert_jax

        self._lock = threading.Lock()
        self._pending = None
        self._has_work = threading.Event()
        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="quick-update-worker",
        )
        self._thread.start()

    def submit(self, analysis, paths, instance):
        """
        Enqueue a quick-update request.  If a previous request is still
        pending (not yet picked up by the worker), it is replaced.
        """

        if self._convert_jax:
            instance = _convert_jax_to_numpy(instance)

        with self._lock:
            self._pending = (analysis, paths, instance)

        self._has_work.set()

    def shutdown(self, timeout: float = 10.0):
        """Signal the worker to stop after draining pending work."""
        self._stop.set()
        self._has_work.set()
        self._thread.join(timeout=timeout)

    def _process_pending(self):
        with self._lock:
            work = self._pending
            self._pending = None

        if work is None:
            return

        analysis, paths, instance = work

        try:
            analysis.perform_quick_update(paths, instance)
        except NotImplementedError:
            pass
        except Exception:
            logger.exception(
                "Background quick-update raised an exception (ignored)."
            )

    def _worker(self):
        while True:
            self._has_work.wait()
            self._has_work.clear()

            self._process_pending()

            if self._stop.is_set():
                self._process_pending()
                break
