import copy
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

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


def _is_ipython_kernel() -> bool:
    """
    Return ``True`` if running inside a Jupyter / Colab IPython kernel,
    ``False`` otherwise (script mode, REPL, IPython terminal).

    Detection works by importing ``IPython.get_ipython`` and checking
    that the returned shell has ``IPKernelApp`` registered in its
    config — only kernel-based shells (Jupyter, Colab, JupyterLab) do.
    A plain IPython terminal returns a shell without the kernel app,
    so it does not match. ``ImportError`` (IPython not installed) and
    any other exception are swallowed so detection stays decoupled
    from the optional IPython dependency.
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


def _resolve_display_image_path(paths) -> Optional[Path]:
    """
    Return the first existing PNG under ``paths.image_path`` that matches
    one of the canonical quick-update filenames, or ``None`` if none of
    them exist yet (e.g. the analysis ran with ``PYAUTO_FAST_PLOTS=1``
    and wrote no figures this iteration).
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


class LiveDisplay:
    """
    Manages live display surfaces for quick-update PNGs.

    When ``live_visual_update=True`` this object dispatches each refresh
    to one of two surfaces:

    - **Jupyter / Colab kernel** — pushes the freshly-written PNG into a
      single cell output element using a stable ``display_id`` so the
      image updates in place.
    - **Script mode** — lazily spawns a small subprocess
      (``python -m autofit.non_linear.live_viewer``) that owns its own
      matplotlib window and polls the PNG on disk. Spawning is deferred
      until the first call because we need ``paths.image_path`` from the
      first work item.

    When ``live_visual_update=False`` every method is a no-op — disk PNG
    writes still happen elsewhere, but no live surface is opened. This
    is the default; users must opt in explicitly per-search.

    The opt-out env var ``PYAUTO_DISABLE_IPYTHON_DISPLAY=1`` continues to
    suppress kernel-side display side-effects (papermill / nbconvert
    pipelines that want PNGs on disk only).
    """

    def __init__(
        self,
        live_visual_update: bool = False,
        display_id: str = "pyauto_fit_progress",
    ):
        self.live_visual_update = live_visual_update
        self.display_id = display_id

        self._display_initialised = False
        self._viewer_proc: Optional[subprocess.Popen] = None
        self._viewer_attempted = False

        self._is_kernel = _is_ipython_kernel() if live_visual_update else False

    def update(self, paths) -> None:
        """
        Refresh the live display surface using whatever PNG has just been
        written under ``paths.image_path``. Called by the quick-update
        path immediately after :py:meth:`Analysis.perform_quick_update`
        returns.

        No-op when ``live_visual_update=False``.
        """
        if not self.live_visual_update:
            return

        if self._is_kernel:
            self._push_to_ipython(paths)
        else:
            self._ensure_viewer(paths)

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Tear down any spawned viewer subprocess. Safe to call when no
        viewer is running.
        """
        proc = self._viewer_proc
        self._viewer_proc = None
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
        except Exception:
            pass

    def _push_to_ipython(self, paths) -> None:
        """
        Display or refresh the latest quick-update subplot in the active
        IPython cell using a stable ``display_id``.

        The image is read from disk as PNG bytes — we deliberately do
        **not** touch matplotlib Figure objects here because callers may
        invoke this from a daemon thread, and matplotlib Figures are not
        thread-safe.

        Skipped entirely when ``PYAUTO_DISABLE_IPYTHON_DISPLAY=1`` is set.
        """
        if os.environ.get("PYAUTO_DISABLE_IPYTHON_DISPLAY") == "1":
            return

        png_path = _resolve_display_image_path(paths)
        if png_path is None:
            return

        try:
            from IPython.display import Image, display, update_display
        except ImportError:
            return

        img = Image(filename=str(png_path))
        if self._display_initialised:
            update_display(img, display_id=self.display_id)
        else:
            display(img, display_id=self.display_id)
            self._display_initialised = True

    def _ensure_viewer(self, paths) -> None:
        """
        Lazily spawn the matplotlib viewer subprocess on first call. If
        the viewer is already running, this is a no-op. If a spawn was
        already attempted (successful or not), we do not retry — closing
        the window is treated as opt-out for the rest of the run.
        """
        if self._viewer_proc is not None:
            return
        if self._viewer_attempted:
            return

        png_path = _resolve_display_image_path(paths)
        if png_path is None:
            return

        self._viewer_attempted = True

        try:
            self._viewer_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "autofit.non_linear.live_viewer",
                    str(png_path),
                ],
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception(
                "Failed to launch live_viewer subprocess; live "
                "visualization disabled for this run."
            )
            self._viewer_proc = None


class BackgroundQuickUpdate:
    """
    Runs ``analysis.perform_quick_update`` on a background daemon thread so
    that the sampler is not blocked while matplotlib renders and saves plots.

    Uses a **latest-only** pattern: if a new best-fit arrives before the
    previous visualisation finishes, the stale request is silently replaced.

    After each ``perform_quick_update`` completes, the worker delegates to
    a :class:`LiveDisplay` to push the freshly-written PNG to either a
    Jupyter cell or a script-mode matplotlib viewer subprocess (only when
    ``live_visual_update=True``).

    Parameters
    ----------
    convert_jax
        If ``True``, JAX arrays on the model instance are converted to
        NumPy before handing them to the worker thread.
    live_visual_update
        If ``True``, enable live display surfaces (Jupyter cell update
        when running inside a kernel, matplotlib viewer subprocess
        otherwise). Defaults to ``False`` — disk PNG writes always
        happen regardless of this flag.
    display_id
        Stable identifier passed to ``IPython.display.update_display`` so
        every quick-update frame refreshes the same cell output element
        rather than appending a new one.
    """

    def __init__(
        self,
        convert_jax: bool = False,
        live_visual_update: bool = False,
        display_id: str = "pyauto_fit_progress",
    ):
        self._convert_jax = convert_jax
        self._live_display = LiveDisplay(
            live_visual_update=live_visual_update,
            display_id=display_id,
        )

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
        self._live_display.shutdown()

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

        try:
            self._live_display.update(paths)
        except Exception:
            logger.exception(
                "Live display update raised an exception (ignored)."
            )

    def _worker(self):
        while True:
            self._has_work.wait()
            self._has_work.clear()

            self._process_pending()

            if self._stop.is_set():
                self._process_pending()
                break
