import copy
import logging
import os
from os import path
from typing import List, Tuple, Union, Type, Optional, Dict

from autoconf.dictable import to_dict
from autofit import exc
from autofit.mapper import prior as p
from autofit.non_linear.parallel import Process
from autofit.text.formatter import write_table
from .job import Job
from .result import GridSearchResult
from .result_builder import ResultBuilder


class Sequential:
    # noinspection PyUnusedLocal
    @staticmethod
    def run_jobs(jobs, *_, **kwargs):
        for job_ in jobs:
            yield job_.perform()


class GridSearch:
    def __init__(
        self,
        search,
        number_of_steps: int = 4,
        number_of_cores: int = 1,
        result_output_interval: int = 100,
    ):
        """
        Performs a non linear optimiser search for each square in a grid. The dimensionality of the search depends on
        the number of distinct priors passed to the fit function. (1 / step_size) ^ no_dimension steps are performed
        per an optimisation.

        Parameters
        ----------
        number_of_steps
            The number of steps to go in each direction
        search
            The class of the search that is run at each step
        result_output_interval
            The interval between saving a GridSearchResult object via paths
        """
        self.number_of_steps = number_of_steps
        self.search = search

        self._logger = None

        self.logger.info("Creating grid search")

        self.number_of_cores = number_of_cores or 1

        self.logger.info(f"Using {number_of_cores} core(s)")

        self._result_output_interval = result_output_interval

    __exclude_identifier_fields__ = ("number_of_cores",)

    def __getstate__(self):
        """
        Remove the logger for pickling
        """
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        return state

    @property
    def logger(self):
        if not hasattr(self, "_logger") or self._logger is None:
            self._logger = logging.getLogger(f"GridSearch ({self.search.name})")
        return self._logger

    @property
    def paths(self):
        return self.search.paths

    @property
    def parallel(self) -> bool:
        """
        Should this grid search run in parallel?
        """
        return self.number_of_cores > 1

    @property
    def step_size(self):
        """
        Returns
        -------
        step_size: float
            The size of a step in any given dimension in hyper space.
        """
        return 1 / self.number_of_steps

    def make_physical_lists(self, grid_priors) -> List[List[float]]:
        lists = self.make_lists(grid_priors)
        return [
            [prior.value_for(value) for prior, value in zip(grid_priors, l)]
            for l in lists
        ]

    def make_lists(self, grid_priors):
        """
        Produces a list of lists of floats, where each list of floats represents the values in each dimension for one
        step of the grid search.

        Parameters
        ----------
        grid_priors: [p.Prior]
            A list of priors that are to be searched using the grid search.

        Returns
        -------
        lists: [[float]]
        """
        return make_lists(
            len(grid_priors), step_size=self.step_size, centre_steps=False
        )

    def make_arguments(self, values, grid_priors):
        arguments = {}
        for value, grid_prior in zip(values, grid_priors):
            try:
                grid_prior.lower_limit
                grid_prior.upper_limit
            except AttributeError:
                raise exc.PriorException(
                    "Priors passed to the grid search must have upper and lower limit (e.g. be UniformPrior)"
                )
            if (
                float("-inf") == grid_prior.lower_limit
                or float("inf") == grid_prior.upper_limit
            ):
                raise exc.PriorException(
                    "Priors passed to the grid search must have definite limits"
                )
            lower_limit = grid_prior.lower_limit + value * grid_prior.width
            upper_limit = (
                grid_prior.lower_limit + (value + self.step_size) * grid_prior.width
            )
            prior = p.UniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)
            arguments[grid_prior] = prior
        return arguments

    def model_mappers(self, model, grid_priors):
        grid_priors = model.sort_priors_alphabetically(set(grid_priors))
        lists = self.make_lists(grid_priors)
        for values in lists:
            arguments = self.make_arguments(values, grid_priors)
            yield model.mapper_from_partial_prior_arguments(arguments)

    def fit(
        self,
        model,
        analysis,
        grid_priors,
        info: Optional[Dict] = None,
    ):
        """
        Fit an analysis with a set of grid priors. The grid priors are priors associated with the model mapper
        of this instance that are replaced by uniform priors for each step of the grid search.

        Parameters
        ----------
        info
        model
        analysis: autofit.non_linear.non_linear.Analysis
            An analysis used to determine the fitness of a given model instance
        grid_priors: [p.Prior]
            A list of priors to be substituted for uniform priors across the grid.

        Returns
        -------
        result: GridSearchResult
            An object that comprises the results from each individual fit
        """
        self.paths.model = model
        self.paths.search = self

        self.logger.info("Running grid search...")

        process_class = Process if self.parallel else Sequential
        # noinspection PyArgumentList
        return self._fit(
            model=model,
            analysis=analysis,
            grid_priors=grid_priors,
            process_class=process_class,
            info=info,
        )

    def _fit(
        self,
        model,
        analysis,
        grid_priors,
        process_class=Union[Type[Process], Type[Sequential]],
        info: Optional[Dict] = None,
    ):
        """
        Perform the grid search in parallel, with all the optimisation for each grid square being performed on a
        different process.

        Parameters
        ----------
        model
        analysis
            An analysis
        grid_priors
            Priors describing the position in the grid
        process_class
            Class used to dispatch jobs

        Returns
        -------
        result: GridSearchResult
            The result of the grid search
        """
        grid_priors = model.sort_priors_alphabetically(set(grid_priors))
        lists = self.make_lists(grid_priors)

        jobs = self.make_jobs(model, analysis, grid_priors, info)

        builder = ResultBuilder(
            lists=lists,
            grid_priors=grid_priors,
            paths=[j.search_instance.paths for j in jobs],
        )

        self.save_metadata()

        def save_results():
            self.paths.save_json("result", to_dict(builder()))

        def write_results():
            self.logger.debug("Writing results")

            os.makedirs(self.paths.output_path, exist_ok=True)

            is_evidence = any(row[-1] is not None for row in results_list)

            headers = [
                "index",
                *map(model.name_for_prior, grid_priors),
                "log_likelihood_increase",
            ]
            rows = results_list

            if is_evidence:
                headers.append("log_evidence")
            else:
                rows = [row[:-1] for row in rows]

            write_table(
                headers,
                rows,
                self.paths.output_path / "results.csv",
            )

        results_list = []

        for i, job_result in enumerate(
            process_class.run_jobs(jobs, self.number_of_cores)
        ):
            builder.add(job_result)
            results_list.append(job_result.result_list_row)
            write_results()
            if i % self._result_output_interval == 0:
                save_results()

        save_results()
        self.paths.completed()

        return builder()

    def save_metadata(self):
        self.paths.save_unique_tag(is_grid_search=True)

    def make_jobs(self, model, analysis, grid_priors, info: Optional[Dict] = None):
        grid_priors = model.sort_priors_alphabetically(set(grid_priors))
        lists = self.make_lists(grid_priors)

        jobs = list()

        for index, values in enumerate(lists):
            jobs.append(
                self.job_for_analysis_grid_priors_and_values(
                    analysis=copy.deepcopy(analysis),
                    model=model,
                    grid_priors=grid_priors,
                    values=values,
                    index=index,
                    info=info,
                )
            )
        return jobs

    def job_for_analysis_grid_priors_and_values(
        self, model, analysis, grid_priors, values, index, info: Optional[Dict] = None
    ):
        arguments = self.make_arguments(values=values, grid_priors=grid_priors)
        model = model.mapper_from_partial_prior_arguments(arguments=arguments)

        labels = []
        for prior in sorted(arguments.values(), key=lambda pr: pr.id):
            labels.append(
                "{}_{:.2f}_{:.2f}".format(
                    model.name_for_prior(prior), prior.lower_limit, prior.upper_limit
                )
            )

        name_path = path.join(
            self.paths.name,
            self.paths.identifier,
            "_".join(labels),
        )

        search_instance = self.search_instance(name_path=name_path)
        search_instance.paths.model = model

        return Job(
            search_instance=search_instance,
            model=model,
            analysis=analysis,
            arguments=arguments,
            index=index,
            info=info,
        )

    def search_instance(self, name_path):
        search_instance = self.search.copy_with_paths(
            self.paths.create_child(
                name=name_path,
                path_prefix=self.paths.path_prefix,
                is_identifier_in_paths=False,
            )
        )

        for key, value in self.__dict__.items():
            if key not in ("model", "instance", "paths", "search", "number_of_cores"):
                try:
                    setattr(search_instance, key, value)
                except AttributeError:
                    pass

        if self.number_of_cores > 1:
            search_instance.number_of_cores = 1

        return search_instance


def grid(fitness, no_dimensions, step_size):
    """
    Grid2D search using a fitness function over a given number of dimensions and a given step size between inclusive
    limits of 0 and 1.

    Parameters
    ----------
    fitness: function
        A function that takes a tuple of floats as an argument
    no_dimensions: int
        The number of dimensions of the grid search
    step_size: float
        The step size of the grid search

    Returns
    -------
    best_arguments: tuple[float]
        The tuple of arguments that gave the highest fitness
    """
    best_fitness = float("-inf")
    best_arguments = None

    for arguments in make_lists(no_dimensions, step_size):
        fitness = fitness(tuple(arguments))
        if fitness > best_fitness:
            best_fitness = fitness
            best_arguments = tuple(arguments)

    return best_arguments


def make_lists(
    no_dimensions: int, step_size: Union[Tuple[float], float], centre_steps=True
):
    """
        Returns a list of lists of floats covering every combination across no_dimensions of points of integer step size
    between 0 and 1 inclusive.

    Parameters
    ----------
    no_dimensions
        The number of dimensions, that is the length of the lists
    step_size
        The step size. This can be a float or a tuple with the same number of dimensions
    centre_steps

    Returns
    -------
    lists: [[float]]
        A list of lists
    """
    if isinstance(step_size, float):
        step_size = tuple(step_size for _ in range(no_dimensions))

    if no_dimensions == 0:
        return [[]]

    sub_lists = make_lists(no_dimensions - 1, step_size[1:], centre_steps=centre_steps)
    step_size = step_size[0]
    return [
        [step_size * value + (0.5 * step_size if centre_steps else 0)] + sub_list
        for value in range(int((1 / step_size)))
        for sub_list in sub_lists
    ]
