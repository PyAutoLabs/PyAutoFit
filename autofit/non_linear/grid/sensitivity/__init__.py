import logging
import os
from copy import copy
import numpy as np
from pathlib import Path
from typing import List, Generator, Callable, ClassVar, Optional, Union, Tuple

from autonerves import cached_property
from autonerves.dictable import to_dict
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.grid.grid_search import make_lists, Sequential
from autofit.non_linear.grid.sensitivity.job import Job, MaskedJobResult
from autofit.non_linear.grid.sensitivity.job import JobResult
from autofit.non_linear.grid.sensitivity.result import SensitivityResult
from autofit.non_linear.parallel import Process
from autofit.text.formatter import write_table


class Sensitivity:
    def __init__(
        self,
        base_model: AbstractPriorModel,
        perturb_model: AbstractPriorModel,
        simulation_instance,
        paths,
        simulate_cls: Callable,
        base_fit_cls: Callable,
        perturb_fit_cls: Callable,
        job_cls: ClassVar = Job,
        batch_range : Tuple[int, int] = None,
        visualizer_cls: Optional[Callable] = None,
        perturb_model_prior_func: Optional[Callable] = None,
        number_of_steps: Union[Tuple[int, ...], int] = 4,
        mask: Optional[List[bool]] = None,
        number_of_cores: int = 2,
        limit_scale: int = 1,
    ):
        """
        Perform sensitivity mapping to evaluate whether a perturbation
        can be detected if it occurs in different parts of an image.

        For a range from 0 to 1 with step_size, for each dimension of the
        perturb_model, a perturbation is created and used in conjunction
        with the instance to create an image.

        For each of these images, a fit is run with just the model and with both
        the model and perturb_model to compare how much better the image
        can be fit if the perturbation is included.

        Parameters
        ----------
        base_model
            A model that fits the instance well
        perturb_model
            A model which provides a perturbations to be applied to the instance
            before creating images
        simulation_instance
            An instance of a model to which perturbations are applied prior to
            images being generated
        simulate_cls
            A class which simulates images from each perturb instance that sensitivity mapping is performed on.
        base_fit_cls
            The class which fits the base model to each simulated dataset of the sensitivity map.
        perturb_fit_cls
            The class which fits the perturb model to each simulated dataset of the sensitivity map.
        batch_range
            The integer range of sensitivity mapping jobs to perform. If None, all jobs are performed. If not None,
            only the jobs with indices within this range are performed. This means, for example, the range can be
            used to distribute jobs to different machines.
        visualizer_cls
            A class which can be used to visualize the results of the sensitivity mapping after each fit is performed,
            therefore providing visualization on the fly.
        number_of_steps
            The number of steps for each dimension of the sensitivity grid. If input as a float the dimensions are
            all that value. If input as a tuple of length the number of dimensions, each tuple value is the number of
            steps in that dimension.
        mask
            A mask to apply to the sensitivity grid, such that all `True` values are not included in the sensitivity
            mapping. This is useful for removing regions of the sensitivity grid that are expected to have no
            sensitivity, for example because they have no signal.
        number_of_cores
            How many cores does this computer have?
        limit_scale
            Scales the priors for each perturbation model.
                A scale of 1 means priors have limits the same size as the grid square.
                A scale of 2 means priors have limits larger than the grid square with
                    width twice a grid square.
                A scale of 0.5 means priors have limits smaller than the grid square
                    with width half a grid square.
        """
        self.logger = logging.getLogger(f"Sensitivity ({paths.name})")

        self.logger.info("Creating")

        self.instance = simulation_instance
        self.model = base_model
        self.perturb_model = perturb_model

        self.paths = paths

        self.simulate_cls = simulate_cls
        self.base_fit_cls = base_fit_cls
        self.perturb_fit_cls = perturb_fit_cls

        self.perturb_model_prior_func = perturb_model_prior_func

        self.job_cls = job_cls
        self.batch_range = batch_range
        self.visualizer_cls = visualizer_cls

        self.number_of_steps = number_of_steps
        self.mask = None

        if mask is not None:
            self.mask = np.asarray(mask)
            if self.shape != self.mask.shape:
                raise ValueError(
                    f"""
                    The mask of the Sensitivity object must have the same shape as the sensitivity grid.
                    
                    For your inputs, the shape of each are as follows:
                    
                    Sensitivity Grid: {self.shape}
                    Mask: {self.mask.shape}
                    """
                )

        self.number_of_cores = number_of_cores

        self.limit_scale = limit_scale

    def run(self) -> SensitivityResult:
        """
        Run fits and comparisons for all perturbations, returning
        a list of results.
        """
        self.logger.info("Running")

        self.paths.save_unique_tag(is_grid_search=True)

        headers = [
            "index",
            *self._headers,
            "log_evidence_increase",
            "log_likelihood_increase",
        ]
        physical_values = list(self._physical_values)

        process_class = Process if self.number_of_cores > 1 else Sequential

        results = []
        jobs = []

        for number in range(len(self._perturb_instances)):
            model = self.model.copy()
            model.perturb = self._perturb_models[number]
            results.append(
                MaskedJobResult(
                    number=number,
                    model=model,
                )
            )

            if not self._should_bypass(number=number):
                jobs.append(self._make_job(number))

        if self.batch_range is not None:
            jobs = jobs[self.batch_range[0]:self.batch_range[1]]

        for result in process_class.run_jobs(
            jobs, number_of_cores=self.number_of_cores
        ):
            if isinstance(result, Exception):
                raise result

            results[result.number] = result

            sensitivity_result = SensitivityResult(
                samples=[result.result.samples_summary for result in results],
                perturb_samples=[
                    result.perturb_result.samples_summary for result in results
                ],
                shape=self.shape,
                path_values=self.path_values,
            )

            if self.visualizer_cls is not None:
                self.visualizer_cls(
                    sensitivity_result=sensitivity_result, paths=self.paths
                )

            os.makedirs(self.paths.output_path, exist_ok=True)

            write_table(
                headers=headers,
                rows=[
                    [
                        result.number,
                        *physical_values[result.number],
                        result.log_evidence_increase,
                        result.log_likelihood_increase,
                    ]
                    for result in results
                ],
                filename=self.results_path,
            )

        # TODO : Had to repeat this code block to get certain unit tests to pass which presumably bypass run_jobs.

        sensitivity_result = SensitivityResult(
            samples=[result.result.samples_summary for result in results],
            perturb_samples=[
                result.perturb_result.samples_summary for result in results
            ],
            shape=self.shape,
            path_values=self.path_values,
        )

        self.paths.save_json("result", to_dict(sensitivity_result))

        return sensitivity_result

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the sensitivity grid.

        The shape is the number of steps performed for each dimension of the `perturb_model`. If sensitivity mapping
        is performed in 3D, the shape will therefore be a tuple of length 3.

        The `shape` can vary across dimensions if the `number_of_steps` parameter is input as a tuple.

        Returns
        -------
        The shape of the sensitivity grid.
        """

        if isinstance(self.number_of_steps, tuple):
            return self.number_of_steps

        return tuple(
            self.number_of_steps for _ in range(self.perturb_model.prior_count)
        )

    def shape_index_from_number(self, number: int) -> Tuple[int, ...]:
        """
        Returns the index of the sensitivity grid from a number.

        Parameters
        ----------
        number
            The number of the sensitivity grid.

        Returns
        -------
        The index of the sensitivity grid.
        """
        return np.unravel_index(number, self.shape)

    @property
    def step_size(self) -> Union[float, Tuple]:
        """
        Returns
        -------
        step_size
            The size of a step in any given dimension in hyper space.
        """
        if isinstance(self.number_of_steps, tuple):
            return tuple(
                1 / number_of_steps for number_of_steps in self.number_of_steps
            )
        return 1 / self.number_of_steps

    @property
    def results_path(self) -> Path:
        return self.paths.output_path / "results.csv"

    @cached_property
    def _lists(self) -> List[List[float]]:
        """
        A list of hypercube vectors, used to instantiate
        the perturb_model and create the individual
        perturbations.
        """
        return make_lists(self.perturb_model.prior_count, step_size=self.step_size)

    @cached_property
    def path_values(self):
        paths = [
            self.perturb_model.path_for_prior(prior)
            for prior in self.perturb_model.priors_ordered_by_id
        ]

        return {
            path: list(values) for path, *values in zip(paths, *self._physical_values)
        }

    @cached_property
    def _physical_values(self) -> List[List[float]]:
        """
        Lists of physical values for each grid square
        """
        return [
            [
                prior.value_for(unit_value)
                for prior, unit_value in zip(
                    self.perturb_model.priors_ordered_by_id, unit_values
                )
            ]
            for unit_values in self._lists
        ]

    @cached_property
    def _headers(self) -> Generator[str, None, None]:
        """
        A name for each of the perturb priors
        """
        for path, _ in self.perturb_model.prior_tuples:
            yield path

    @cached_property
    def _labels(self) -> List[str]:
        """
        One label for each perturbation, used to distinguish
        fits for each perturbation by placing them in separate
        directories.
        """
        labels = []
        for list_ in self._lists:
            strings = list()
            for value, prior_tuple in zip(list_, self.perturb_model.prior_tuples):
                path, prior = prior_tuple
                value = prior.value_for(value)
                strings.append(f"{path}_{value}")
            labels.append("_".join(strings))

        return labels

    @cached_property
    def _perturb_instances(self) -> List[ModelInstance]:
        """
        A list of instances each of which defines a perturbation to
        be applied to the image.
        """

        return [
            self.perturb_model.instance_from_unit_vector(list_) for list_ in self._lists
        ]

    @cached_property
    def _perturb_models(self) -> List[AbstractPriorModel]:
        """
        A list of models representing a perturbation at each grid square.

        By default models have priors with limits at the edges of a grid square.
        These limits can be scaled using the limit_scale variable. If the variable
        is 2 then the priors will have width twice the step size.
        """
        if isinstance(self.step_size, tuple):
            step_sizes = self.step_size
        else:
            step_sizes = (self.step_size,) * self.perturb_model.prior_count

        half_steps = [self.limit_scale * step_size / 2 for step_size in step_sizes]

        perturb_models = []
        for list_ in self._lists:
            limits = [
                (
                    prior.value_for(max(0.0, centre - half_step)),
                    prior.value_for(min(1.0, centre + half_step)),
                )
                for centre, prior, half_step in zip(
                    list_,
                    self.perturb_model.priors_ordered_by_id,
                    half_steps,
                )
            ]
            perturb_models.append(self.perturb_model.with_limits(limits))
        return perturb_models

    def _should_bypass(self, number: int) -> bool:
        shape_index = self.shape_index_from_number(number=number)
        return self.mask is not None and np.asarray(self.mask)[shape_index]

    def _make_jobs(self) -> Generator[Job, None, None]:
        for number, _ in enumerate(self._perturb_instances):
            yield self._make_job(number)

    def _make_job(self, number) -> Generator[Job, None, None]:
        """
        Create a list of jobs to be run on separate processes.

        Each job fits a perturb image with the original model
        and a model which includes a perturbation.
        """
        perturb_instance = self._perturb_instances[number]
        perturb_model = self._perturb_models[number]
        label = self._labels[number]

        if self.perturb_model_prior_func is not None:
            perturb_model = self.perturb_model_prior_func(
                perturb_instance=perturb_instance, perturb_model=perturb_model
            )

        simulate_instance = copy(self.instance)
        simulate_instance.perturb = perturb_instance

        paths = self.paths.for_sub_analysis(
            label,
        )

        return self.job_cls(
            simulate_instance=simulate_instance,
            model=self.model,
            perturb_model=perturb_model,
            base_instance=self.instance,
            simulate_cls=self.simulate_cls,
            base_fit_cls=self.base_fit_cls,
            perturb_fit_cls=self.perturb_fit_cls,
            paths=paths,
            number=number,
        )
