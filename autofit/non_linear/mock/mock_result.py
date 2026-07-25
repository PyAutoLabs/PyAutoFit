from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.non_linear.result import Result

from autofit.non_linear.mock.mock_samples_summary import MockSamplesSummary
from autofit.non_linear.mock.mock_samples import MockSamples


class MockResult(Result):
    def __init__(
        self,
        samples_summary: MockSamplesSummary = None,
        paths=None,
        samples=None,
        instance=None,
        analysis=None,
        search=None,
        model=None,
    ):
        super().__init__(
            samples_summary=samples_summary
            or MockSamplesSummary(
                model=model or ModelMapper(),
            ),
            paths=paths,
            samples=samples,
            search_internal=None,
        )

        self._instance = instance or ModelInstance()
        self._samples = samples or MockSamples(
            model=model
            or ModelMapper()
        )

        self.prior_means = None
        self.analysis = analysis
        self.search = search
        self.model = model

    @property
    def model_centred(self):
        try:
            return self.samples_summary.model_centred
        except AttributeError:
            return self.model

    def model_centred_absolute(self, a):
        try:
            return self.samples_summary.model_centred_absolute(a)
        except AttributeError:
            return self.model

    def model_centred_relative(self, r):
        try:
            return self.samples_summary.model_centred_relative(r)
        except AttributeError:
            return self.model

    @property
    def last(self):
        return self


class MockResultGrid(Result):
    def __init__(self, log_likelihood):
        # noinspection PyTypeChecker
        super().__init__(None, None)
        self._log_likelihood = log_likelihood
        self.model = log_likelihood

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @property
    def best_model(self):
        return self.model
