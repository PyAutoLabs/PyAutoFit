from autofit.non_linear.analysis import Analysis
from autofit.non_linear.samples import Sample

def samples_with_log_likelihood_list(
        log_likelihood_list
):
    return [
        Sample(
            log_likelihood=log_likelihood,
            log_prior=0,
            weight=0
        )
        for log_likelihood
        in log_likelihood_list
    ]


class MockAnalysis(Analysis):
    prior_count = 2

    def __init__(self):
        super().__init__()
        self.fit_instances = list()

    def log_likelihood_function(self, instance):
        self.fit_instances.append(instance)
        return [1]

    def visualize(self, paths, instance, during_analysis):
        pass

    def log(self, instance):
        pass

