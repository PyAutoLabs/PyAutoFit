from typing import Optional, Dict

from autofit.non_linear.parallel import AbstractJob, AbstractJobResult


class JobResult(AbstractJobResult):
    def __init__(self, result, result_list_row, number):
        """
        The result of a job

        Parameters
        ----------
        result
            The result of a grid search
        result_list_row
            A row in the result list
        """
        super().__init__(number)
        self.result = result
        self.result_list_row = result_list_row


class Job(AbstractJob):
    def __init__(
        self,
        search_instance,
        model,
        analysis,
        arguments,
        index,
        info: Optional[Dict] = None,
    ):
        """
        A job to be performed in parallel.

        Parameters
        ----------
        search_instance
            An instance of an optimiser
        analysis
            An analysis
        arguments
            The grid search arguments
        """
        super().__init__(number=index)
        self.search_instance = search_instance
        self.analysis = analysis
        self.model = model
        self.arguments = arguments
        self.index = index
        self.info = info

    def perform(self):

        result = self.search_instance.fit(
            model=self.model,
            analysis=self.analysis,
            info=self.info,
        )
        result_list_row = [
            self.index,
            *[
                prior.lower_limit
                for prior in self.model.sort_priors_alphabetically(
                    self.arguments.values()
                )
            ],
            result.log_likelihood,
            result.samples_summary.log_evidence,
        ]

        return JobResult(result, result_list_row, self.number)
