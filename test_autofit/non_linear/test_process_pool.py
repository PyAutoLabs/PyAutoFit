"""
The `Process` job pool behind `GridSearch` and `Sensitivity` never hangs: a job whose result cannot be
pickled back, or a worker that dies, raises a `JobError` instead of leaving the parent waiting forever
(the reason `GridSearch` could not default to `number_of_cores="auto"`).
"""
import os
import signal

import pytest

from autofit.non_linear.parallel import AbstractJob, Process
from autofit.non_linear.parallel.process import JobError


@pytest.fixture(autouse=True)
def fail_instead_of_hang():
    def handler(signum, frame):
        raise TimeoutError("the job pool hung")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


class Unpicklable:
    def __init__(self):
        self.function = lambda: None


class SquareJob(AbstractJob):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def perform(self):
        return self.value**2


class UnpicklableResultJob(AbstractJob):
    def perform(self):
        return Unpicklable()


class RaisingJob(AbstractJob):
    def perform(self):
        raise ValueError("job failed")


class DyingJob(AbstractJob):
    def perform(self):
        os._exit(1)


def test_results_are_returned():
    results = list(Process.run_jobs([SquareJob(value) for value in range(5)], 3))
    assert sorted(results) == [0, 1, 4, 9, 16]


def test_unpicklable_result_raises_instead_of_hanging():
    results = []
    with pytest.raises(AssertionError, match="cannot be pickled"):
        for result in Process.run_jobs([UnpicklableResultJob(), SquareJob(2)], 2):
            results.append(result)
    errors = [result for result in results if isinstance(result, JobError)]
    assert len(results) == 2
    assert len(errors) == 1
    assert "cannot be pickled" in str(errors[0])


def test_exception_result_counts_once():
    results = Process.run_jobs([RaisingJob(), SquareJob(3), SquareJob(4)], 2)
    received = []
    with pytest.raises(AssertionError, match="job failed"):
        for result in results:
            received.append(result)
    assert len(received) == 3
    assert sorted(r for r in received if not isinstance(r, Exception)) == [9, 16]


def test_dead_worker_raises_instead_of_hanging():
    with pytest.raises(JobError, match="returned no result"):
        list(Process.run_jobs([DyingJob(), SquareJob(2)], 2))
