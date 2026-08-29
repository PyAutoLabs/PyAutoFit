import autofit as af


class ConstantAnalysis(af.Analysis):
    """
    An `Analysis` whose log likelihood is a fixed number, independent of the instance.

    The magnitude guard in `Fitness.call` is a pure function of the log likelihood, so a real fit
    would only add noise to the tests: what is under test is which values survive the guard, and
    this makes that value the input.
    """

    def __init__(self, log_likelihood, use_jax: bool = False):
        super().__init__(use_jax=use_jax)
        self.log_likelihood = log_likelihood

    def log_likelihood_function(self, instance):
        return self.log_likelihood
