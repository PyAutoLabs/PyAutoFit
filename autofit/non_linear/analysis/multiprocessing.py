import logging
import math
from itertools import count
from multiprocessing import Process
from multiprocessing import Queue

from autofit.non_linear.parallel.sneaky import StopCommand
from autofit.non_linear.paths.abstract import AbstractPaths

logger = logging.getLogger(__name__)


class AnalysisProcess(Process):
    _id = count()

    def __init__(self, analyses):
        """
        A process that performs one or more analyses on each
        instance passed to it

        Parameters
        ----------
        analyses
            A list of analyses this process performs
        """
        super().__init__()
        self.analyses = analyses
        self.instance_queue = Queue()
        self.queue = Queue()

        self.name = f"analysis_process_{next(self._id)}"

        self.logger = logging.getLogger(self.name)
        self.logger.info("Created")

    def _run(self):
        """
        Private run function permitting process to be stopped using a return
        when a StopCommand is passed
        """
        while True:
            instance = self.instance_queue.get()
            if instance is StopCommand:
                return
            for analysis in self.analyses:
                try:
                    if isinstance(instance, tuple):
                        command, *args = instance
                        result = getattr(analysis, command)(*args)
                    else:
                        result = analysis.log_likelihood_function(instance)
                    self.queue.put(result)
                except Exception as e:
                    self.queue.put(e)

    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.
        """

        self.logger.debug("starting")

        self._run()

        self.logger.debug("terminating")
        self.queue.close()


class AnalysisPool:
    def __init__(self, analyses, n_cores: int):
        """
        A pool which distributes analysis evenly across
        n_cores processes.

        Callable as a log-likelihood function.

        Parameters
        ----------
        analyses
            A list of analyses to be performed in parallel
        n_cores
            The number of cores available to perform analyses over
        """
        self.processes = list()

        # The one JAX fork rule (`autofit.non_linear.parallel.pool`). This pool always
        # forks (at least one process), so it refuses a JAX analysis outright.
        if any(getattr(analysis, "is_jax", False) for analysis in analyses):
            from autofit import exc
            from autofit.non_linear.parallel.pool import JAX_FORK_REASON

            raise exc.SearchException(
                f"AnalysisPool cannot evaluate a JAX analysis: {JAX_FORK_REASON}. "
                "Evaluate the analyses in this process instead."
            )

        self.analyses = analyses
        self.n_cores = n_cores

        self.n_analyses = len(analyses)
        n_processes = min(self.n_analyses, n_cores)

        analyses_per_process = math.ceil(self.n_analyses / n_processes)

        self.processes = list()

        for n in range(n_processes):
            analyses_ = analyses[
                n * analyses_per_process : (n + 1) * analyses_per_process
            ]
            process = AnalysisProcess(analyses=analyses_)

            self.processes.append(process)
            process.start()

    def __del__(self):
        """
        Called when the map goes out of scope.

        Tell each process to terminate with a StopCommand and then join
        each process with a timeout of one second.
        """
        self.terminate()

    def terminate(self):
        """
        Terminate each process and join it with a timeout of one second.
        """
        logger.debug("Deconstructing SneakyMap")

        logger.debug("Terminating processes...")
        for process in self.processes:
            process.instance_queue.put(StopCommand)

        logger.debug("Joining processes...")
        for process in self.processes:
            try:
                process.join(0.5)
            except Exception as e:
                logger.exception(e)

    def __call__(self, instance) -> float:
        """
        Evaluate the likelihood of an instance according to several
        analyses, in parallel.

        Parameters
        ----------
        instance
            Some instance

        Returns
        -------
        The total log likelihood
        """
        for process in self.processes:
            process.instance_queue.put(instance)

        return sum(self.results())

    def results(self):
        """
        Get the results from each process not necessarily in order.
        """
        count_ = 0
        results = []

        while count_ < self.n_analyses:
            for process in self.processes:
                if process.queue.empty():
                    continue
                result = process.queue.get()
                results.append(result)

                if isinstance(result, Exception):
                    raise result

                count_ += 1

        return results

    def map(
        self,
        function_name: str,
        paths: AbstractPaths,
        *args,
    ):
        """
        Call a function on each analysis in parallel.

        Parameters
        ----------
        function_name
            The name of the function to call
        paths
            The paths to the output directory
        args
            The arguments to pass to the function
        """
        for i, process in enumerate(self.processes):
            child_paths = paths.for_sub_analysis(analysis_name=f"analyses/analysis_{i}")
            process.instance_queue.put((function_name, child_paths, *args))

        self.results()
