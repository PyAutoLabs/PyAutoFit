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
        self.reasons = []
        """Why values of this row are missing (``status_reason``), filled by columns."""

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
    def latent_summary(self):
        return self.result.value("latent.latent_summary")

    @property
    def _latent_summary(self):
        return self.latent_summary

    @cached_property
    def parameter_paths(self) -> set:
        """
        The argument paths of the model's parameters (from the samples_summary only).
        """
        samples_summary = self.result.samples_summary
        return set(self._sample_kwargs(samples_summary.median_pdf_sample)) | set(
            self._sample_kwargs(samples_summary.max_log_likelihood_sample)
        )

    def _sample_kwargs(self, sample) -> dict:
        """
        The arguments of a sample keyed by every path to each prior, or an
        empty dict when the summary holds no such sample (e.g. a search
        without a PDF has no median_pdf_sample).
        """
        if sample is None:
            return {}
        return self._add_paths(sample.kwargs)

    @cached_property
    def median_pdf_sample_kwargs(self) -> dict:
        """
        The median_pdf_sample arguments for the search from the samples_summary and latent_summary.
        """
        samples_summary = self.result.samples_summary
        kwargs = self._sample_kwargs(samples_summary.median_pdf_sample)

        latent_summary = self._latent_summary
        if latent_summary is not None and latent_summary.median_pdf_sample is not None:
            kwargs.update(latent_summary.median_pdf_sample.kwargs)

        return kwargs

    @cached_property
    def max_likelihood_kwargs(self):
        """
        The max_log_likelihood_sample arguments for the search from the samples_summary and latent_summary.
        """
        samples_summary = self.result.samples_summary
        kwargs = self._sample_kwargs(samples_summary.max_log_likelihood_sample)

        latent_summary = self._latent_summary
        if (
            latent_summary is not None
            and latent_summary.max_log_likelihood_sample is not None
        ):
            kwargs.update(latent_summary.max_log_likelihood_sample.kwargs)

        return kwargs

    @cached_property
    def values_at_sigma_1_kwargs(self) -> dict:
        """
        The values_at_sigma_1 arguments for the search from the samples_summary.
        """
        kwargs = self._add_paths(self.result.samples_summary.values_at_sigma_1 or {})

        latent_summary = self._latent_summary
        if latent_summary is not None:
            kwargs.update(
                {
                    tuple(key.split(".")): value
                    for key, value in (latent_summary.values_at_sigma_1 or {}).items()
                }
            )

        return kwargs

    @cached_property
    def values_at_sigma_3_kwargs(self) -> dict:
        """
        The values_at_sigma_3 arguments for the search from the samples_summary.
        """
        kwargs = self._add_paths(self.result.samples_summary.values_at_sigma_3 or {})

        latent_summary = self._latent_summary
        if latent_summary is not None:
            kwargs.update(
                {
                    tuple(key.split(".")): value
                    for key, value in (latent_summary.values_at_sigma_3 or {}).items()
                }
            )

        return kwargs

    @cached_property
    def known_paths(self) -> set:
        """
        Every argument path that can be resolved for this row, from the
        samples_summary and the latent_summary.
        """
        return (
            set(self.median_pdf_sample_kwargs)
            | set(self.max_likelihood_kwargs)
            | set(self.values_at_sigma_1_kwargs)
            | set(self.values_at_sigma_3_kwargs)
        )

    def dict(self, columns=None) -> dict:
        """
        The row as a dictionary including an id and one entry for each column (of
        ``columns``, default all).
        """
        row = {"id": self.result.id}
        for column in self._columns if columns is None else columns:
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
