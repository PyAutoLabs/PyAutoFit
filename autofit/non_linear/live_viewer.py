"""
Standalone matplotlib viewer process for live quick-update visualization.

Invoked by :class:`BackgroundQuickUpdate` (via ``subprocess.Popen``) when
``live_visual_update=True`` and the search is running in plain script mode
(i.e. not inside a Jupyter / Colab kernel — in a kernel the cell update
path is used instead).

The viewer owns its own matplotlib window in its own process. Running in
a subprocess sidesteps two issues:

- matplotlib Figures are not thread-safe, so the existing
  ``BackgroundQuickUpdate`` daemon-thread can't open windows itself.
- A GUI event loop in the search process would compete with the sampler
  for the main thread.

Invocation::

    python -m autofit.non_linear.live_viewer <image_path> [--title TITLE]

The viewer polls ``<image_path>`` (typically ``<output>/image/fit.png``,
the canonical filename written by ``subplot_fit`` across all dataset
types) every 0.5s. On ``st_mtime`` change it reloads the PNG and redraws. Exits cleanly on SIGINT / SIGTERM / window
close. Logs a single warning and exits 0 if the configured matplotlib
backend cannot display a window (e.g. headless WSL2 with MPLBACKEND=Agg).
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger("autofit.live_viewer")


POLL_SECONDS = 0.5
HEADLESS_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}


def _backend_can_show() -> bool:
    """Return True if the active matplotlib backend can display a window."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    return not any(backend.startswith(name) for name in HEADLESS_BACKENDS)


def _install_signal_handlers(stop_event):
    def _handle(signum, frame):
        stop_event["stop"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def run(image_path: Path, title: str) -> int:
    import matplotlib

    if not _backend_can_show():
        logger.warning(
            "live_viewer: matplotlib backend %r cannot display a window; "
            "live visualization disabled (PNG writes to %s continue normally).",
            matplotlib.get_backend(),
            image_path,
        )
        return 0

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure(num=title)
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    fig.tight_layout()

    im = None
    last_mtime: float | None = None
    stop = {"stop": False}
    _install_signal_handlers(stop)

    def _window_closed(_event):
        stop["stop"] = True

    fig.canvas.mpl_connect("close_event", _window_closed)

    while not stop["stop"]:
        try:
            mtime = image_path.stat().st_mtime if image_path.exists() else None
        except OSError:
            mtime = None

        if mtime is not None and mtime != last_mtime:
            try:
                img = mpimg.imread(str(image_path))
            except Exception:
                img = None

            if img is not None:
                if im is None:
                    im = ax.imshow(img)
                else:
                    im.set_data(img)
                fig.canvas.draw_idle()
                last_mtime = mtime

        try:
            plt.pause(POLL_SECONDS)
        except Exception:
            break

    try:
        plt.close(fig)
    except Exception:
        pass
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="autofit.live_viewer")
    parser.add_argument("image_path", help="Path to the PNG to display.")
    parser.add_argument(
        "--title",
        default="PyAutoFit — quick update",
        help="Window title.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("PYAUTO_LIVE_VIEWER_LOG", "WARNING"),
        format="%(levelname)s %(name)s: %(message)s",
    )

    return run(Path(args.image_path), args.title)


if __name__ == "__main__":
    sys.exit(main())
