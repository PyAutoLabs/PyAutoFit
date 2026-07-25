from autofit.non_linear.samples import SamplesPDF, Sample, SamplesNest


def samples_with_log_likelihood_list(log_likelihood_list):
    return [
        Sample(log_likelihood=log_likelihood, log_prior=0, weight=0)
        for log_likelihood in log_likelihood_list
    ]


class MockSamples(SamplesPDF):
    def __init__(
        self,
        model=None,
        sample_list=None,
        samples_info=None,
        log_likelihood_list=None,
        prior_means=None,
        **kwargs,
    ):
        self._log_likelihood_list = log_likelihood_list

        self.model = model

        sample_list = sample_list or self.default_sample_list

        samples_info = samples_info or {"unconverged_sample_size": 0}

        super().__init__(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
            **kwargs,
        )

    @property
    def default_sample_list(self):
        return [
            Sample(
                log_likelihood=log_likelihood,
                log_prior=0.0,
                weight=0.0,
                kwargs={path: 1.0 for path in self.model.paths} if self.model else {},
            )
            for log_likelihood in range(3)
        ]

    @property
    def log_likelihood_list(self):
        if self._log_likelihood_list is None:
            return super().log_likelihood_list

        return self._log_likelihood_list

    @property
    def unconverged_sample_size(self):
        return self.samples_info["unconverged_sample_size"]


class MockSamplesNest(SamplesNest):
    def __init__(
        self,
        model,
        sample_list=None,
        samples_info=None,
    ):
        self.model = model

        if sample_list is None:
            sample_list = [
                Sample(log_likelihood=log_likelihood, log_prior=0.0, weight=0.0)
                for log_likelihood in self.log_likelihood_list
            ]

        super().__init__(
            model=model, sample_list=sample_list, samples_info=samples_info
        )
