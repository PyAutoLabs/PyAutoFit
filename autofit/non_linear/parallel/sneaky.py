import cProfile
import logging
import warnings
import multiprocessing as mp
import os
from typing import Iterable, Optional, Callable

from autonerves import conf

from autofit.non_linear.paths.abstract import AbstractPaths
from .process import AbstractJob, Process, StopCommand

logger = logging.getLogger(__name__)


def _is_likelihood_function(function) -> bool:
    """
    Is the function a callable used to evaluate likelihood?

    Naturally in Autofit this would be a child of the Fitness class.
    In Dynesty the likelihood function is wrapped in _function_wrapper
    and called 'loglikelihood'

    Parameters
    ----------
    function
        Some object

    Returns
    -------
    Is the object a log likelihood function?
    """
    from autofit.non_linear.fitness import Fitness
    from dynesty.dynesty import _function_wrapper
    from emcee.ensemble import _FunctionWrapper

    return any(
        [
            isinstance(function, Fitness),
            isinstance(function, _function_wrapper)
            and function.name == "loglikelihood",
            isinstance(function, _FunctionWrapper),
        ]
    )


class SneakyJob(AbstractJob):
    def __init__(self, function, *args):
        """
        A job performed on a process.

        If the function is the log likelihood function then it is set to None.

        If the log likelihood function is in the args, it is filtered from the args,
        but its index retained.

        This prevents large amounts of data comprised in an Analysis class from being
        copied over to processes multiple times.

        Parameters
        ----------
        function
            Some function to which a pool.map has been applied
        args
            The arguments to that function
        """

        super().__init__()
        if _is_likelihood_function(function):
            self.function = None
        else:
            self.function = function

        self.args = list()
        self.fitness_index = None

        for i, arg in enumerate(args):
            if _is_likelihood_function(arg):
                if self.fitness_index is not None:
                    raise AssertionError(
                        f"Two arguments of type NonLinear.Fitness passed to {function.__name__}"
                    )
                self.fitness_index = i
            else:
                self.args.append(arg)

    @property
    def is_not_fitness(self) -> bool:
        """
        If the map is not being applied to a fitness function and a fitness
        function is not one of the arguments then this map need not be sneaky
        """
        return self.function is not None and self.fitness_index is None

    def perform(self, likelihood_function):
        """
        Computes the log likelihood. The likelihood function
        is passed from a copy associated with the current process.

        Depending on whether the likelihood function itself is
        being mapped, or some function mapped onto the likelihood
        function as an argument, the likelihood function will be
        called or added to the arguments.

        Parameters
        ----------
        likelihood_function
            A likelihood function associated with the processes
            to avoid copying data for every single function call

        Returns
        -------
        The log likelihood
        """
        if self.function is None:
            return likelihood_function(self.args)
        if self.is_not_fitness:
            return self.function(self.args)
        args = (
            self.args[: self.fitness_index]
            + [likelihood_function]
            + self.args[self.fitness_index :]
        )
        return self.function(args)


class SneakyProcess(Process):
    def __init__(
        self,
        name: str,
        paths: AbstractPaths,
        initializer=None,
        initargs=None,
        job_args=tuple(),
    ):
        """
        Each SneakyProcess creates its own queue to avoid locking during
        highly parallel optimisations.
        """

        super().__init__(
            name,
            job_queue=mp.Queue(),
            initializer=initializer,
            initargs=initargs,
            job_args=job_args,
        )
        self.paths = paths

    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.

        The process continues to execute until a StopCommand is passed.
        This occurs when the SneakyMap goes out of scope.
        """

        if conf.instance["general"]["profiling"]["parallel_profile"]:
            pr = cProfile.Profile()
            pr.enable()

        self._init()
        self.logger.debug("starting")
        while True:
            job = self.job_queue.get()
            if job is StopCommand:
                self.logger.debug("StopCommand found")
                break
            try:
                self.queue.put(job.perform(*self.job_args))
            except Exception as e:
                self.logger.exception(e)
                self.queue.put(e)
        self.logger.debug("terminating process {}".format(self.name))
        self.job_queue.close()

        if conf.instance["general"]["test"]["parallel_profile"]:
            try:
                os.makedirs(self.paths.profile_path)
            except FileExistsError:
                pass

            sneaky_path = self.paths.profile_path / f"sneaky_{self.pid}.prof"

            pr.dump_stats(sneaky_path)
            pr.disable()

    def open_profiler(self):
        if conf.instance["general"]["test"]["parallel_profile"]:
            pr = cProfile.Profile()
            pr.enable()


class SneakyPool:
    def __init__(
        self,
        processes: int,
        fitness,
        paths: AbstractPaths,
        initializer=None,
        initargs=None,
    ):
        """
        Implements the same interface as multiprocessing's pool,
        but associates the fitness object with each process to
        prevent data being copies to each process for every function
        call.

        Parameters
        ----------
        processes
            The number of cores to be used simultaneously.
        fitness
            A class comprising data and a model which can be used
            to evaluate the likelihood of a live point
        initializer
        initargs
        """
        logger.info(f"Creating pool with {processes} processes")
        self._processes = [
            SneakyProcess(
                str(number),
                paths=paths,
                initializer=initializer,
                initargs=initargs,
                job_args=(fitness,),
            )
            for number in range(processes)
        ]
        for process in self.processes:
            process.start()

    @property
    def processes(self):
        return self._processes

    def map(self, function, args_list, log_info : bool = True):
        """
        Execute the function with the given arguments across all of the
        processes. The likelihood  argument is removed from each args in
        the args_list.

        Parameters
        ----------
        function
            Some function
        args_list
            An iterable of iterables of arguments passed to the function

        Yields
        ------
        Results from the function evaluation
        """
        jobs = [
            SneakyJob(
                function, *((args,) if not isinstance(args, Iterable) else tuple(args))
            )
            for args in args_list
        ]

        if log_info:
            logger.info(f"Running {len(jobs)} jobs across {self.processes} processes")

        for i, job in enumerate(jobs):
            process = self.processes[i % len(self.processes)]
            process.job_queue.put(job)

        target = len(jobs)
        count = 0

        exception = None

        while count < target:
            for process in self.processes:
                if not process.queue.empty():
                    item = process.queue.get()
                    count += 1
                    if isinstance(item, Exception):
                        exception = item
                    else:
                        yield item

        logger.debug("All jobs complete")

        if exception is not None:
            raise exception

    def __del__(self):
        """
        Called when the map goes out of scope.

        Tell each process to terminate with a StopCommand and then join
        each process with a timeout of one second.
        """
        logger.debug("Deconstructing SneakyMap")

        logger.debug("Terminating processes...")
        for process in self.processes:
            process.job_queue.put(StopCommand)

        logger.debug("Joining processes...")
        for process in self.processes:
            try:
                process.join(0.5)
            except Exception as e:
                logger.exception(e)


class FunctionCache:
    """
    Singleton class to cache the functions and optional arguments between calls
    """


def initializer(
    fitness,
    prior_transform,
    fitness_args,
    fitness_kwargs,
    prior_transform_args,
    prior_transform_kwargs,
):
    """
    Initialized function used to initialize the
    singleton object inside each worker of the pool
    """
    FunctionCache.fitness = fitness
    FunctionCache.prior_transform = prior_transform
    FunctionCache.fitness_args = fitness_args
    FunctionCache.fitness_kwargs = fitness_kwargs
    FunctionCache.prior_transform_args = prior_transform_args
    FunctionCache.prior_transform_kwargs = prior_transform_kwargs


def fitness_cache(x):
    """
    Likelihood function call
    """
    return FunctionCache.fitness(
        x, *FunctionCache.fitness_args, **FunctionCache.fitness_kwargs
    )


def prior_transform_cache(x):
    """
    Prior transform call
    """

    return FunctionCache.prior_transform(
        x, *FunctionCache.prior_transform_args, **FunctionCache.prior_transform_kwargs
    )


class SneakierPool:
    def __init__(
        self,
        processes: int,
        fitness: Callable,
        prior_transform: Optional[Callable] = None,
        fitness_args: Optional[Iterable] = None,
        fitness_kwargs: Optional[dict] = None,
        prior_transform_args: Optional[Iterable] = None,
        prior_transform_kwargs: Optional[dict] = None,
    ):
        self.fitness_init = fitness
        self.prior_transform_init = prior_transform
        self.fitness = fitness_cache
        self.prior_transform = prior_transform_cache
        self.fitness_args = fitness_args or ()
        self.fitness_kwargs = fitness_kwargs or {}
        self.prior_transform_args = prior_transform_args or ()
        self.prior_transform_kwargs = prior_transform_kwargs or {}
        self.processes = processes
        self.pool = None

        init_args = (
            self.fitness_init,
            self.prior_transform_init,
            self.fitness_args,
            self.fitness_kwargs,
            self.prior_transform_args,
            self.prior_transform_kwargs,
        )
        initializer(*init_args)

    def __enter__(self):
        """
        Activate the mp / mpi pool
        """

        logger.info("... using multiprocessing")

        self.pool = mp.Pool(
            processes=self.processes,
        )

        return self

    def map(self, function: Callable, iterable: Iterable):
        """
        Map a function over an iterable using the map method
        of the initialized pool.

        Parameters
        ----------
        function
            A function to map
        iterable
            An iterable to map over

        """
        return self.pool.map(function, iterable)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.pool.terminate()
        except:  # noqa
            pass
        try:
            del (
                FunctionCache.fitness,
                FunctionCache.prior_transform,
                FunctionCache.fitness_args,
                FunctionCache.fitness_kwargs,
                FunctionCache.prior_transform_args,
                FunctionCache.prior_transform_kwargs,
            )
        except:  # noqa
            pass

    @property
    def size(self):
        return self.njobs

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()
