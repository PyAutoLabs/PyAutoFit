from typing import List, Tuple, Union, Dict

from autofit.non_linear.grid.grid_list import GridList, as_grid_list
from autofit.non_linear.grid.grid_search.result import AbstractGridSearchResult
from autofit.non_linear.samples.interface import SamplesInterface


# noinspection PyTypeChecker
class SensitivityResult(AbstractGridSearchResult):
    def __init__(
        self,
        samples: List[SamplesInterface],
        perturb_samples: List[SamplesInterface],
        shape: Tuple[int, ...],
        path_values: Dict[Tuple[str, ...], List[float]],
    ):
        """
        The result of a sensitivity mapping

        Parameters
        ----------
        shape
            The shape of the sensitivity mapping grid.
        path_values
            A list of tuples of the path to the grid priors and the physical values themselves.
        """
        super().__init__(GridList(samples, shape))
        self.perturb_samples = GridList(perturb_samples, shape)
        self.shape = shape
        self.path_values = path_values

    def perturbed_physical_centres_list_from(
        self, path: Union[str, Tuple[str, ...]]
    ) -> GridList:
        """
        Returns the physical centres of the perturbed model for each sensitivity fit

        Parameters
        ----------
        path
            The path to the physical centres in the samples
        """
        if isinstance(path, str):
            path = tuple(path.split("."))
        return self.path_values[path]

    def __getitem__(self, item):
        return self.samples[item]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    @property
    @as_grid_list
    def log_evidences_base(self) -> GridList:
        """
        The log evidences of the base model for each sensitivity fit
        """
        return [sample.log_evidence for sample in self.samples]

    @property
    @as_grid_list
    def log_evidences_perturbed(self) -> GridList:
        """
        The log evidences of the perturbed model for each sensitivity fit
        """
        return [sample.log_evidence for sample in self.perturb_samples]

    @property
    @as_grid_list
    def log_evidence_differences(self) -> GridList:
        """
        The log evidence differences between the base and perturbed models
        """
        return [
            log_evidence_perturbed - log_evidence_base
            for log_evidence_perturbed, log_evidence_base in zip(
                self.log_evidences_perturbed, self.log_evidences_base
            )
        ]

    @property
    @as_grid_list
    def log_likelihoods_base(self) -> GridList:
        """
        The log likelihoods of the base model for each sensitivity fit
        """
        return [sample.log_likelihood for sample in self.samples]

    @property
    @as_grid_list
    def log_likelihoods_perturbed(self) -> GridList:
        """
        The log likelihoods of the perturbed model for each sensitivity fit
        """
        return [sample.log_likelihood for sample in self.perturb_samples]

    @property
    @as_grid_list
    def log_likelihood_differences(self) -> GridList:
        """
        The log likelihood differences between the base and perturbed models
        """
        return [
            log_likelihood_perturbed - log_likelihood_base
            for log_likelihood_perturbed, log_likelihood_base in zip(
                self.log_likelihoods_perturbed, self.log_likelihoods_base
            )
        ]

    def figure_of_merits(
        self,
        use_log_evidences: bool,
    ) -> GridList:
        """
        Convenience method to get either the log likelihoods difference or log evidence difference of the grid search.

        Parameters
        ----------
        use_log_evidences
            If true, the log evidences are returned, otherwise the log likelihoods are returned.
        """

        if use_log_evidences:
            return self.log_evidence_differences
        return self.log_likelihood_differences
