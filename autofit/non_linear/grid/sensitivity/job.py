from copy import copy
from itertools import count
from typing import Callable, Optional

from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.parallel import AbstractJob, AbstractJobResult
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result


class JobResult(AbstractJobResult):
    def __init__(self, number: int, result: Result, perturb_result: Result):
        """
        The result of a single sensitivity comparison

        Parameters
        ----------
        result
        perturb_result
        """
        super().__init__(number)
        self.result = result
        self.perturb_result = perturb_result

    @property
    def log_evidence_increase(self) -> Optional[float]:
        """
        Returns a tuple of the log evidence of the base model, the perturbed model and the difference between them.

        This is used to ouptut the sensitivity mapping results to .csv files.

        If the log evidence is not available, a tuple containing 3 None's is returned.
        """

        if hasattr(self.result.samples, "log_evidence"):
            if self.result.samples.log_evidence is not None and self.perturb_result.samples.log_evidence is not None:
                return float(
                    self.perturb_result.samples.log_evidence
                    - self.result.samples.log_evidence
                )

    @property
    def log_likelihood_increase(self) -> Optional[float]:
        """
        Returns a tuple of the log likelihood of the base model, the perturbed model and the difference between them.

        This is used to ouptut the sensitivity mapping results to .csv files.
        """

        return float(self.perturb_result.log_likelihood - self.result.log_likelihood)


class MaskedJobResult(AbstractJobResult):
    """
    A placeholder result for a job that has been masked out.
    """

    def __init__(self, number, model):
        super().__init__(number)
        self.model = model

    @property
    def result(self):
        return self

    @property
    def perturb_result(self):
        return self

    def __getattr__(self, item):
        return None

    @property
    def samples_summary(self):
        return self

    @property
    def log_evidence(self):
        return 0.0

    @property
    def log_likelihood(self):
        return 0.0


class Job(AbstractJob):
    _number = count()

    def __init__(
        self,
        model: AbstractPriorModel,
        simulate_cls: Callable,
        perturb_model: AbstractPriorModel,
        simulate_instance: ModelInstance,
        base_instance: ModelInstance,
        base_fit_cls: Callable,
        perturb_fit_cls: Callable,
        paths: AbstractPaths,
        number: int,
    ):
        """
        Job to run non-linear searches comparing how well a model and a model with a perturbation fit the image.

        Parameters
        ----------
        model
            A base model that fits the image without a perturbation
        perturb_model
            A model of the perturbation which has been added to the underlying image
        base_fit_cls
            A class which defines the function which fits the base model to each simulated dataset of the sensitivity
            map.
        perturb_fit_cls
            A class which defines the function which fits the perturbed model to each simulated dataset of the
            sensitivity map.
        paths
            The paths defining the output directory structure of the sensitivity mapping.
        """
        super().__init__(number=number)

        self.model = model
        self.simulate_cls = simulate_cls
        self.perturb_model = perturb_model
        self.simulate_instance = simulate_instance
        self.base_instance = base_instance
        self.base_fit_cls = base_fit_cls
        self.perturb_fit_cls = perturb_fit_cls
        self.paths = paths

    @property
    def base_paths(self):
        return self.paths.for_sub_analysis("[base]")

    @property
    def perturb_paths(self):
        return self.paths.for_sub_analysis("[perturb]")

    @property
    def is_complete(self) -> bool:
        """
        Returns True if the job has been completed, False otherwise.
        """
        return (self.base_paths.is_complete and self.perturb_paths.is_complete) or (
            (self.paths.output_path / "[base].zip").exists()
            and (self.paths.output_path / "[perturb].zip").exists()
        )

    def perform(self) -> JobResult:
        """
        - Create one model with a perturbation and another without
        - Fit each model against the perturbed image

        Returns
        -------
        An object comprising the results of the two fits
        """

        dataset = self.simulate_cls(
            instance=self.simulate_instance,
            simulate_path=self.paths.image_path.with_name("simulate"),
        )

        result = self.base_fit_cls(
            model=self.model,
            dataset=dataset,
            paths=self.base_paths,
            instance=self.simulate_instance,
        )

        perturb_model = copy(self.model)
        perturb_model.perturb = self.perturb_model

        perturb_result = self.perturb_fit_cls(
            model=perturb_model,
            dataset=dataset,
            paths=self.perturb_paths,
            instance=self.simulate_instance,
        )

        return JobResult(
            number=self.number,
            result=result,
            perturb_result=perturb_result,
        )
