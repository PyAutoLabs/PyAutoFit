import logging
import multiprocessing
import pickle
from abc import ABC, abstractmethod
from collections import abc
from itertools import count
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler
from typing import Iterable

from .context import fork_context

logger = logging.getLogger(
    __name__
)


class JobError(Exception):
    """
    A job run by a `Process` pool produced no usable result: its result (or the exception it raised)
    could not be pickled back to the parent process, or the worker processes exited with jobs still
    outstanding. Constructed from one message string, so it always pickles.
    """


class _PickledResult:
    """
    A job's result (or exception), pickled by the worker before it is queued.

    `multiprocessing.Queue.put` pickles in a background feeder thread, which only prints a pickling
    error and drops the item, so the parent would wait for it forever. Pickling in the worker turns
    that failure into a `JobError` the parent receives.
    """

    def __init__(self, payload: bytes):
        self.payload = payload

    @classmethod
    def from_result(cls, result, job) -> "_PickledResult":
        try:
            return cls(bytes(ForkingPickler.dumps(result)))
        except Exception as e:
            kind = "exception" if isinstance(result, BaseException) else "result"
            message = (
                f"Job {getattr(job, 'number', '?')} returned a {kind} of type {type(result).__name__} "
                f"that cannot be pickled back to the parent process ({e!r}). Results of parallel jobs "
                f"must be picklable: define classes they hold (e.g. an Analysis) at module level, not "
                f"inside a function, or run with number_of_cores=1."
            )
            if isinstance(result, BaseException):
                message += f" The job raised: {result!r}"
            logger.error(message)
            return cls(bytes(ForkingPickler.dumps(JobError(message))))

    def load(self):
        return pickle.loads(self.payload)


class AbstractJobResult(ABC):
    def __init__(self, number):
        self.number = number

    def __eq__(self, other):
        return self.number == other.number

    def __gt__(self, other):
        return self.number > other.number

    def __lt__(self, other):
        return self.number < other.number


class AbstractJob(ABC):
    """
    Task to be completed in parallel
    """
    _number = count()

    def __init__(self, number=None):
        if number is None:
            number = next(self._number)
        self.number = number

    @abstractmethod
    def perform(self, *args):
        """
        Perform the task and return the result
        """


class Process(fork_context().Process):

    def __init__(
            self,
            name: str,
            job_queue: multiprocessing.Queue,
            initializer=None,
            initargs=None,
            job_args=tuple()
    ):
        """
        A parallel process that consumes Jobs through the job queue and outputs results through its own queue.

        Parameters
        ----------
        name: str
            The name of the process
        job_queue: multiprocessing.Queue
            The queue through which jobs are submitted
        """

        super().__init__(name=name)
        self.logger = logging.getLogger(
            f"process {name}"
        )

        self.logger.info("created")

        self.job_queue = job_queue
        self.queue = fork_context().Queue()

        self.initializer = initializer
        self.initargs = initargs

        self.job_args = job_args

    def _init(self):
        if self.initializer is not None:
            if self.initargs is None:
                return self.initializer()
            if isinstance(
                    self.initargs,
                    abc.Iterable
            ):
                initargs = tuple(self.initargs)
            else:
                initargs = (self.initargs,)

            self.initializer(
                *initargs
            )

    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.
        """
        self._init()

        self.logger.debug("starting")
        while True:
            # The parent queues one `StopCommand` per worker after every job. Polling
            # `job_queue.empty()` instead races the parent's feeder thread: a worker that
            # starts before the jobs reach the pipe sees an empty queue and exits at once.
            job = self.job_queue.get()
            if isinstance(job, StopCommand):
                break
            try:
                result = job.perform(
                    *self.job_args
                )
            except Exception as e:
                logger.exception(e)
                result = e
            self.queue.put(
                _PickledResult.from_result(result, job)
            )
        self.logger.debug("terminating")
        self.job_queue.close()

    @classmethod
    def run_jobs(
            cls,
            jobs: Iterable[AbstractJob],
            number_of_cores: int,
            initializer=None,
            initargs=None,
            job_args=tuple()
    ):
        """
        Run the collection of jobs across n - 1 other cores.

        Parameters
        ----------
        job_args
        initargs
        initializer
        jobs
            Serializable concrete children of the AbstractJob class
        number_of_cores
            The number of cores this computer has. Must be at least 2.
        """
        if number_of_cores < 2:
            raise AssertionError(
                "The number of cores available must be at least 2 for parallel to run"
            )

        job_queue = fork_context().Queue()

        processes = [
            cls(
                str(number),
                job_queue,
                initializer=initializer,
                initargs=initargs,
                job_args=job_args
            )
            for number in range(number_of_cores - 1)
        ]

        total = 0
        for job in jobs:
            job_queue.put(job)
            total += 1
        for _ in processes:
            job_queue.put(StopCommand())

        logger.info(
            f"Running {total} jobs across {number_of_cores} processes"
        )

        for process in processes:
            process.start()

        logger.debug("Starting processes")

        process_count = 0
        exception = None

        while process_count < total:
            received = False
            for process in processes:
                while not process.queue.empty():
                    result = process.queue.get()
                    if isinstance(result, _PickledResult):
                        result = result.load()
                    received = True
                    process_count += 1
                    if isinstance(result, Exception):
                        logger.exception(result)
                        exception = result
                    yield result

            if received:
                continue

            # A worker's queue is flushed before it exits, so once every worker has exited and
            # every queue is empty no further result can arrive: a job's worker died (e.g. it was
            # killed or crashed) without returning one.
            if not any(process.is_alive() for process in processes) and all(
                process.queue.empty() for process in processes
            ):
                # Unread jobs would block interpreter exit while the feeder thread waits
                # to flush them into a pipe no worker reads.
                job_queue.cancel_join_thread()
                job_queue.close()
                raise JobError(
                    f"{total - process_count} of {total} parallel jobs returned no result: every "
                    f"worker process has exited (exit codes "
                    f"{[process.exitcode for process in processes]}). A worker that is killed or "
                    f"crashes mid-job loses that job; check the log above for the cause."
                )

            # Block until a running worker exits or a short time passes, rather than spinning.
            wait(
                [process.sentinel for process in processes if process.is_alive()],
                timeout=0.1,
            )

        job_queue.close()

        logger.debug("Finished jobs")

        for process in processes:
            process.join(timeout=1.0)

        logger.debug("Joining processes")

        if exception is not None:
            raise AssertionError(
                exception
            )


class StopCommand:
    """
    A command that can be passed into a process queue to gracefully stop
    the process
    """
