import copy
import logging
import os
import threading
from pathlib import Path

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


# Filenames the worker will look for under `paths.image_path`, in priority
# order, when pushing a quick-update frame to a Jupyter / Colab cell. The
# first existing file wins. Imaging analyses write `subplot_fit.png`;
# interferometer / dataset-model variants may write `fit.png` or
# `subplot_tracer.png`.
_DISPLAY_CANDIDATES = ("subplot_fit.png", "fit.png", "subplot_tracer.png")


class BackgroundQuickUpdate:
    """
    Runs ``analysis.perform_quick_update`` on a background daemon thread so
    that the sampler is not blocked while matplotlib renders and saves plots.

    Uses a **latest-only** pattern: if a new best-fit arrives before the
    previous visualisation finishes, the stale request is silently replaced.

    When the search is running inside a Jupyter / Colab kernel, the worker
    additionally pushes the freshly-written subplot PNG to the active cell
    via :func:`IPython.display.update_display` with a stable ``display_id``,
    so the cell that ran ``search.fit(...)`` shows a single self-updating
    image rather than just writing PNGs to disk. Outside a kernel (plain
    ``python my_fit.py``) this layer is silently skipped — PNGs still land
    on disk and no IPython side effects fire. Users can opt out inside a
    kernel by setting ``PYAUTO_DISABLE_IPYTHON_DISPLAY=1`` (e.g. for
    papermill / automated nbconvert pipelines).

    Parameters
    ----------
    convert_jax
        If ``True``, JAX arrays on the model instance are converted to
        NumPy before handing them to the worker thread.
    display_id
        Stable identifier passed to ``IPython.display.update_display`` so
        every quick-update frame refreshes the same cell output element
        rather than appending a new one. Default
        ``"pyauto_fit_progress"`` is fine for almost all uses; override
        only when running two concurrent searches in the same kernel and
        wanting each in its own cell output.
    """

    def __init__(
        self,
        convert_jax: bool = False,
        display_id: str = "pyauto_fit_progress",
    ):
        self._convert_jax = convert_jax
        self._display_id = display_id
        self._display_initialised = False

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

    @staticmethod
    def _is_ipython_kernel() -> bool:
        """
        Return ``True`` if running inside a Jupyter / Colab IPython kernel,
        ``False`` otherwise (script mode, REPL, IPython terminal).

        Detection works by importing ``IPython.get_ipython`` and checking
        that the returned shell has ``IPKernelApp`` registered in its
        config — only kernel-based shells (Jupyter, Colab, JupyterLab) do.
        A plain IPython terminal returns a shell without the kernel app,
        so it does not match. ``ImportError`` (IPython not installed) and
        any other exception are swallowed: the worker stays decoupled from
        the optional IPython dependency.
        """
        try:
            from IPython import get_ipython
        except ImportError:
            return False
        try:
            ipy = get_ipython()
            if ipy is None:
                return False
            return "IPKernelApp" in getattr(ipy, "config", {})
        except Exception:
            return False

    def _resolve_display_image_path(self, paths):
        """
        Return the first existing PNG under ``paths.image_path`` that
        matches one of the canonical quick-update filenames, or ``None``
        if none of them exist yet (e.g. the analysis ran with
        ``PYAUTO_FAST_PLOTS=1`` and wrote no figures this iteration).
        """
        image_path = getattr(paths, "image_path", None)
        if image_path is None:
            return None
        base = Path(image_path)
        for name in _DISPLAY_CANDIDATES:
            candidate = base / name
            if candidate.exists():
                return candidate
        return None

    def _push_to_ipython(self, paths):
        """
        Display or refresh the latest quick-update subplot in the active
        IPython cell using a stable ``display_id``.

        The first call publishes the image via ``display(... display_id=...)``;
        subsequent calls update it in place via
        ``update_display(... display_id=...)``. The image is read from disk
        as PNG bytes — we deliberately do **not** touch matplotlib Figure
        objects here because this method runs on the daemon worker thread
        and matplotlib Figures are not thread-safe.

        Skipped entirely when ``PYAUTO_DISABLE_IPYTHON_DISPLAY=1`` is set
        — useful for papermill / nbconvert pipelines that want PNGs on
        disk but no inline display side effects.
        """
        if os.environ.get("PYAUTO_DISABLE_IPYTHON_DISPLAY") == "1":
            return

        png_path = self._resolve_display_image_path(paths)
        if png_path is None:
            return

        try:
            from IPython.display import Image, display, update_display
        except ImportError:
            return

        img = Image(filename=str(png_path))
        if self._display_initialised:
            update_display(img, display_id=self._display_id)
        else:
            display(img, display_id=self._display_id)
            self._display_initialised = True

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
            return
        except Exception:
            logger.exception(
                "Background quick-update raised an exception (ignored)."
            )
            return

        # If running inside a Jupyter / Colab kernel, push the freshly-
        # written subplot PNG to the active cell so the user sees the fit
        # update in place. Any failure here is logged-and-swallowed so a
        # display problem never takes the search down.
        if self._is_ipython_kernel():
            try:
                self._push_to_ipython(paths)
            except Exception:
                logger.exception(
                    "IPython display update raised an exception (ignored)."
                )

    def _worker(self):
        while True:
            self._has_work.wait()
            self._has_work.clear()

            self._process_pending()

            if self._stop.is_set():
                self._process_pending()
                break
