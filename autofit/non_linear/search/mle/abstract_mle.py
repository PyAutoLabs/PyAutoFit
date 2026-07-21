from abc import ABC

from autonerves import conf
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.initializer import InitializerBall
from autofit.non_linear.plot import (
    subplot_parameters,
    log_likelihood_vs_iteration,
    figure_of_merit_vs_iteration,
)


class AbstractMLE(NonLinearSearch, ABC):

    def __init__(self, initializer=None, **kwargs):
        super().__init__(
            initializer=initializer
            or InitializerBall(lower_limit=0.49, upper_limit=0.51),
            **kwargs,
        )

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["mle"][name]

        plot_path = self.paths.image_path / "search"

        if should_plot("subplot_parameters"):
            subplot_parameters(samples=samples, path=plot_path, format="png")
            subplot_parameters(
                samples=samples, use_log_y=True, path=plot_path, format="png"
            )
            subplot_parameters(
                samples=samples, use_last_50_percent=True, path=plot_path, format="png"
            )

        if should_plot("log_likelihood_vs_iteration"):
            log_likelihood_vs_iteration(samples=samples, path=plot_path, format="png")
            log_likelihood_vs_iteration(
                samples=samples, use_log_y=True, path=plot_path, format="png"
            )
            log_likelihood_vs_iteration(
                samples=samples, use_last_50_percent=True, path=plot_path, format="png"
            )

        # No-op unless the samples carry a ``fom_history`` (the auto-convergence
        # gradient searches); LBFGS / Drawer leave it absent. Default-on and
        # tolerant of a config predating this key (older workspaces that shadow
        # the ``mle`` plots section) so it never KeyErrors on an MLE fit.
        try:
            plot_figure_of_merit = should_plot("figure_of_merit_vs_iteration")
        except KeyError:
            plot_figure_of_merit = True
        if plot_figure_of_merit:
            figure_of_merit_vs_iteration(samples=samples, path=plot_path, format="png")
