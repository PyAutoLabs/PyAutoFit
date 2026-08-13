from autofit.non_linear.samples import SamplesPDF, Sample, SamplesNest


def samples_with_log_likelihood_list(log_likelihood_list):
    return [
        Sample(log_likelihood=log_likelihood, log_prior=0, weight=0)
        for log_likelihood in log_likelihood_list
    ]


def prior_median_kwargs(model):
    """
    Placeholder sample values for a model, taken as the median of each prior.

    A blanket value like ``1.0`` is invalid for any parameter whose prior
    constrains it, so filling a mock sample with one builds an unphysical
    instance. For example an elliptical profile's ``ell_comps`` must have a
    magnitude below 1, and ``(1.0, 1.0)`` raises on construction. Prior medians
    are always in range, so they are safe for every model.
    """
    if not model:
        return {}

    return {path: prior.value_for(0.5) for path, prior in model.path_priors_tuples}


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
                kwargs=prior_median_kwargs(self.model),
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
