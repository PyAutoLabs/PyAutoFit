from functools import cached_property
from typing import Tuple
from autofit.aggregator.search_output import SearchOutput


class Row:
    def __init__(
        self,
        result: SearchOutput,
        columns: list,
        number: int,
    ):
        """
        A row in the summary table corresponding to one search.

        Parameters
        ----------
        result
            The search output from the aggregator
        columns
            The columns to include in the summary. These are taken from samples_summary or latent_summary
        """
        self.result = result
        self._columns = columns
        self.number = number

    def _all_paths(self, path: Tuple[str, ...]):
        model = self.result.model
        return model.all_paths_for_prior(model.object_for_path(path))

    def _add_paths(self, kwargs):
        return {
            path: value
            for key, value in kwargs.items()
            for path in self._all_paths(
                key.split(".") if isinstance(key, str) else key,
            )
        }

    @cached_property
    def _latent_summary(self):
        return self.result.value("latent.latent_summary")

    @cached_property
    def median_pdf_sample_kwargs(self) -> dict:
        """
        The median_pdf_sample arguments for the search from the samples_summary and latent_summary.
        """
        samples_summary = self.result.samples_summary
        kwargs = self._add_paths(samples_summary.median_pdf_sample.kwargs)

        latent_summary = self._latent_summary
        if latent_summary is not None:
            kwargs.update(latent_summary.median_pdf_sample.kwargs)

        return kwargs

    @cached_property
    def max_likelihood_kwargs(self):
        """
        The median_pdf_sample arguments for the search from the samples_summary and latent_summary.
        """
        samples_summary = self.result.samples_summary
        kwargs = self._add_paths(samples_summary.median_pdf_sample.kwargs)

        latent_summary = self._latent_summary
        if latent_summary is not None:
            kwargs.update(latent_summary.max_log_likelihood_sample.kwargs)

        return kwargs

    @cached_property
    def values_at_sigma_1_kwargs(self) -> dict:
        """
        The values_at_sigma_1 arguments for the search from the samples_summary.
        """
        kwargs = self._add_paths(self.result.samples_summary.values_at_sigma_1)

        latent_summary = self._latent_summary
        if latent_summary is not None:
            kwargs.update(
                {
                    tuple(key.split(".")): value
                    for key, value in latent_summary.values_at_sigma_1.items()
                }
            )

        return kwargs

    @cached_property
    def values_at_sigma_3_kwargs(self) -> dict:
        """
        The values_at_sigma_3 arguments for the search from the samples_summary.
        """
        kwargs = self._add_paths(self.result.samples_summary.values_at_sigma_3)

        latent_summary = self._latent_summary
        if latent_summary is not None:
            kwargs.update(
                {
                    tuple(key.split(".")): value
                    for key, value in latent_summary.values_at_sigma_1.items()
                }
            )

        return kwargs

    def dict(self) -> dict:
        """
        The row as a dictionary including an id and one entry for each column.
        """
        row = {"id": self.result.id}
        for column in self._columns:
            value = column.value(self)
            if isinstance(value, dict):
                for key, item in value.items():
                    row[f"{column.name}_{key}" if key else column.name] = item
            else:
                row[column.name] = value

        return row

    @property
    def column_names(self):
        """
        The names of the columns.
        """
        return list(self.dict().keys())
