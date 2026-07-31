import logging
import multiprocessing
from abc import ABC, abstractmethod
from collections import abc
from itertools import count
from typing import Iterable

from .context import fork_context

logger = logging.getLogger(
    __name__
)


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
            if self.job_queue.empty():
                break
            else:
                job = self.job_queue.get()
                try:
                    self.queue.put(
                        job.perform(
                            *self.job_args
                        )
                    )
                except Exception as e:
                    logger.exception(e)
                    self.queue.put(e)
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

        logger.info(
            f"Running {total} jobs across {number_of_cores} processes"
        )

        for process in processes:
            process.start()

        logger.debug("Starting processes")

        process_count = 0
        exception = None

        while process_count < total:
            for process in processes:
                while not process.queue.empty():
                    result = process.queue.get()
                    if isinstance(result, Exception):
                        process_count += 1
                        logger.exception(result)
                        exception = result
                    process_count += 1
                    yield result

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
