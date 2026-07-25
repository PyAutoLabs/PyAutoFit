from autofit.mapper.prior_model.collection import Collection
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.summary import SamplesSummary


class MockSamplesSummary(SamplesSummary):
    def __init__(
        self,
        model=None,
        max_log_likelihood_sample=None,
        median_pdf_sample=None,
        log_evidence=None,
        max_log_likelihood_instance=None,
        prior_means=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            max_log_likelihood_sample=max_log_likelihood_sample,
            median_pdf_sample=median_pdf_sample,
            log_evidence=log_evidence,
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self._prior_means = prior_means
        self._kwargs = {path: 1.0 for path in self.model.paths} if self.model else {}

    @property
    def max_log_likelihood_sample(self):
        if self._max_log_likelihood_sample is not None:
            return self._max_log_likelihood_sample

        return Sample(
            log_likelihood=1.0,
            log_prior=0.0,
            weight=0.0,
            kwargs=self._kwargs,
        )

    @property
    def median_pdf_sample(self):
        if self._median_pdf_sample is not None:
            return self._median_pdf_sample

        return Sample(
            log_likelihood=1.0,
            log_prior=0.0,
            weight=0.0,
            kwargs=self._kwargs,
        )

    def max_log_likelihood(self, as_instance: bool = True):
        if self._max_log_likelihood_instance is None:
            try:
                return super().max_log_likelihood(as_instance=as_instance)
            except (KeyError, AttributeError):
                pass

        return self._max_log_likelihood_instance

    @property
    def prior_means(self):
        if self._prior_means is None:
            return super().prior_means

        return self._prior_means

    @classmethod
    def default(cls):
        return MockSamplesSummary(
            model=Collection(),
        )
