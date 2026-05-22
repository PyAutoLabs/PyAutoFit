import threading
import time

import numpy as np
import pytest

from autofit.non_linear.quick_update import (
    BackgroundQuickUpdate,
    LiveDisplay,
    _convert_jax_to_numpy,
    _is_ipython_kernel,
)


class MockPaths:
    pass


class MockAnalysis:
    """Records calls to perform_quick_update for assertions."""

    def __init__(self, delay=0.0):
        self.calls = []
        self._lock = threading.Lock()
        self._delay = delay

    def perform_quick_update(self, paths, instance):
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.calls.append(instance)


class ErrorAnalysis:
    """Always raises from perform_quick_update."""

    def perform_quick_update(self, paths, instance):
        raise RuntimeError("Visualization failed")


class TestBackgroundQuickUpdate:
    def test_single_submit(self):
        analysis = MockAnalysis()
        worker = BackgroundQuickUpdate()

        worker.submit(analysis, MockPaths(), "instance_1")
        worker.shutdown()

        assert analysis.calls == ["instance_1"]

    def test_latest_only(self):
        """When multiple submits happen before the worker picks up, only the
        last one should be processed."""
        analysis = MockAnalysis(delay=0.2)
        worker = BackgroundQuickUpdate()

        # First submit — will be picked up and start processing (with delay)
        worker.submit(analysis, MockPaths(), "instance_1")
        time.sleep(0.05)  # let the worker pick it up and start processing

        # These two land while the worker is busy with instance_1
        worker.submit(analysis, MockPaths(), "instance_2")
        worker.submit(analysis, MockPaths(), "instance_3")

        worker.shutdown()

        # instance_1 was already being processed, instance_3 replaces instance_2
        assert analysis.calls == ["instance_1", "instance_3"]

    def test_shutdown_joins_cleanly(self):
        worker = BackgroundQuickUpdate()
        worker.shutdown(timeout=2.0)
        assert not worker._thread.is_alive()

    def test_exception_does_not_crash(self):
        analysis = ErrorAnalysis()
        worker = BackgroundQuickUpdate()

        worker.submit(analysis, MockPaths(), "instance")
        time.sleep(0.1)  # let the worker process it

        # Worker should still be alive and functional
        assert worker._thread.is_alive()
        worker.shutdown()

    def test_exception_followed_by_valid(self):
        """After an exception, subsequent submits should still work."""
        error_analysis = ErrorAnalysis()
        good_analysis = MockAnalysis()
        worker = BackgroundQuickUpdate()

        worker.submit(error_analysis, MockPaths(), "bad")
        time.sleep(0.1)

        worker.submit(good_analysis, MockPaths(), "good")
        worker.shutdown()

        assert good_analysis.calls == ["good"]


class TestConvertJaxToNumpy:
    def test_converts_array_with_device_attr(self):
        class FakeJaxArray:
            def __init__(self, data):
                self.data = data
                self.device = "gpu:0"

            def __array__(self, dtype=None):
                return np.array(self.data, dtype=dtype)

        class Instance:
            def __init__(self):
                self.param = FakeJaxArray([1.0, 2.0, 3.0])
                self.name = "test"

        instance = Instance()
        converted = _convert_jax_to_numpy(instance)

        assert isinstance(converted.param, np.ndarray)
        np.testing.assert_array_equal(converted.param, [1.0, 2.0, 3.0])
        assert converted.name == "test"
        # Original should be unchanged
        assert hasattr(instance.param, "device")

    def test_leaves_plain_values_alone(self):
        class Instance:
            def __init__(self):
                self.x = 1.0
                self.arr = np.array([1, 2])

        instance = Instance()
        converted = _convert_jax_to_numpy(instance)

        assert converted.x == 1.0
        np.testing.assert_array_equal(converted.arr, [1, 2])


class TestConvertJaxFlag:
    def test_convert_jax_false(self):
        analysis = MockAnalysis()
        worker = BackgroundQuickUpdate(convert_jax=False)

        obj = {"key": "value"}  # not a real instance, just checking pass-through
        worker.submit(analysis, MockPaths(), obj)
        worker.shutdown()

        assert analysis.calls[0] is obj

    def test_convert_jax_true(self):
        class FakeJaxArray:
            def __init__(self, data):
                self.data = data
                self.device = "gpu:0"

            def __array__(self, dtype=None):
                return np.array(self.data, dtype=dtype)

        class Instance:
            def __init__(self):
                self.param = FakeJaxArray([1.0])

        analysis = MockAnalysis()
        worker = BackgroundQuickUpdate(convert_jax=True)

        worker.submit(analysis, MockPaths(), Instance())
        worker.shutdown()

        assert isinstance(analysis.calls[0].param, np.ndarray)


class TestIPythonDisplayLayer:
    """
    Covers the Jupyter / Colab cell live-update wiring added on top of the
    existing background-thread / latest-only / log-and-swallow behaviour.

    The display layer is gated on `_is_ipython_kernel()` so script-mode
    behaviour (PNG-on-disk, no IPython side effects) is preserved.
    """

    def test_is_ipython_kernel_false_in_pytest(self):
        """
        Plain pytest does not run inside a Jupyter / Colab kernel, so the
        detection helper must return False. This is the script-mode
        fallback path users rely on when running `python my_fit.py`.
        """
        assert _is_ipython_kernel() is False

    def test_push_to_ipython_no_op_when_png_missing(self, tmp_path):
        """
        If `perform_quick_update` produced no PNGs (e.g. `PYAUTO_FAST_PLOTS=1`
        suppressed them, or an early-iteration scenario where the visualizer
        wrote nothing yet), `_push_to_ipython` must silently no-op rather
        than raising — the search must not be taken down by a missing file.
        """
        class Paths:
            image_path = tmp_path  # exists but contains no PNGs

        live = LiveDisplay(live_visual_update=True)
        # No exception expected, no display side effects.
        live._push_to_ipython(Paths())
        assert live._display_initialised is False

    def test_push_to_ipython_display_then_update_sequence(
        self, tmp_path, monkeypatch
    ):
        """
        With a fake IPython.display module installed in `sys.modules`, the
        first `_push_to_ipython` call must invoke `display(..., display_id=...)`
        and the second must invoke `update_display(..., display_id=...)` with
        the same id. This is the contract that lets the notebook cell
        refresh in place rather than appending stacked frames.
        """
        import sys
        import types

        png_path = tmp_path / "subplot_fit.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)  # fake PNG header

        calls = []

        def fake_display(obj, display_id=None):
            calls.append(("display", display_id))

        def fake_update_display(obj, display_id=None):
            calls.append(("update_display", display_id))

        class FakeImage:
            def __init__(self, filename=None):
                self.filename = filename

        fake_display_module = types.ModuleType("IPython.display")
        fake_display_module.display = fake_display
        fake_display_module.update_display = fake_update_display
        fake_display_module.Image = FakeImage
        fake_ipython_module = types.ModuleType("IPython")
        fake_ipython_module.display = fake_display_module

        monkeypatch.setitem(sys.modules, "IPython", fake_ipython_module)
        monkeypatch.setitem(sys.modules, "IPython.display", fake_display_module)
        monkeypatch.delenv("PYAUTO_DISABLE_IPYTHON_DISPLAY", raising=False)

        class Paths:
            image_path = tmp_path

        live = LiveDisplay(
            live_visual_update=True, display_id="test-display-id"
        )
        live._push_to_ipython(Paths())
        live._push_to_ipython(Paths())
        live._push_to_ipython(Paths())

        # First call uses `display`, subsequent calls use `update_display`,
        # and the display_id is stable across all of them so the cell
        # output is replaced rather than appended.
        assert calls == [
            ("display", "test-display-id"),
            ("update_display", "test-display-id"),
            ("update_display", "test-display-id"),
        ]
        assert live._display_initialised is True


class TestLiveVisualUpdateFlag:
    """
    Covers the opt-in `live_visual_update` flag. Default-off means no
    Jupyter cell push and no script-mode viewer subprocess, even when
    the surrounding environment would otherwise enable them.
    """

    def test_default_is_no_op_even_in_simulated_kernel(self, tmp_path, monkeypatch):
        """
        With ``live_visual_update=False`` (the default), ``LiveDisplay.update``
        must never call into IPython.display, even if a Jupyter kernel is
        present. This protects users who explicitly want PNGs-on-disk only.
        """
        import sys
        import types

        png_path = tmp_path / "subplot_fit.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        calls = []

        def fake_display(obj, display_id=None):
            calls.append("display")

        def fake_update_display(obj, display_id=None):
            calls.append("update_display")

        fake_display_module = types.ModuleType("IPython.display")
        fake_display_module.display = fake_display
        fake_display_module.update_display = fake_update_display
        fake_display_module.Image = type("Image", (), {"__init__": lambda self, **kw: None})

        monkeypatch.setitem(sys.modules, "IPython.display", fake_display_module)

        class Paths:
            image_path = tmp_path

        live = LiveDisplay(live_visual_update=False)
        live.update(Paths())

        assert calls == []
        assert live._display_initialised is False
        assert live._viewer_proc is None
        assert live._viewer_attempted is False

    def test_script_mode_spawns_viewer_subprocess(self, tmp_path, monkeypatch):
        """
        With ``live_visual_update=True`` outside a kernel, the first call
        to ``update`` must lazily spawn the live_viewer subprocess against
        the resolved PNG path.
        """
        png_path = tmp_path / "subplot_fit.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        spawn_calls = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                spawn_calls.append(args)
                self.args = args

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        import autofit.non_linear.quick_update as qu

        monkeypatch.setattr(qu.subprocess, "Popen", FakePopen)

        class Paths:
            image_path = tmp_path

        live = LiveDisplay(live_visual_update=True)
        # Force script-mode dispatch regardless of detection result.
        live._is_kernel = False

        live.update(Paths())
        live.update(Paths())  # second call must not re-spawn

        assert len(spawn_calls) == 1
        argv = spawn_calls[0]
        assert "autofit.non_linear.live_viewer" in argv
        assert str(png_path) in argv

        live.shutdown()
        assert live._viewer_proc is None
