import threading
import time

import numpy as np
import pytest

from autofit.non_linear.quick_update import BackgroundQuickUpdate, _convert_jax_to_numpy


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
