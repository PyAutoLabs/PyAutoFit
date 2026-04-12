import os

from autoconf import conf

from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.test_mode import skip_visualization

class Visualizer:

    def should_visualize(
        self, paths: AbstractPaths, during_analysis: bool = True
    ) -> bool:
        """
        Whether a visualize method should be called perform visualization, which depends on the following:

        1) If a model-fit has already completed, the default behaviour is for visualization to be bypassed in order
        to make model-fits run faster.

        2) If a model-fit has completed, but it is the final visualization output where `during_analysis` is False,
        it should be performed.

        3) Visualization can be forced to run via the `force_visualization_overwrite`, for example if a user
        wants to plot additional images that were not output on the original run.

        4) If the analysis is running a database session visualization is switched off.

        5) If PyAutoFit test mode is on visualization is disabled, irrespective of the `force_visualization_overwite`
        config input.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.

        Returns
        -------
        A bool determining whether visualization should be performed or not.
        """

        if skip_visualization():
            return False

        if isinstance(paths, DatabasePaths) or isinstance(paths, NullPaths):
            return False

        if conf.instance["general"]["output"]["force_visualize_overwrite"]:
            return True

        if not during_analysis:
            return True

        return not paths.is_complete

    """
    Methods associated with visualising analysis, model and data before, during
    or after an optimisation.
    """

    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: AbstractPaths,
        model: AbstractPriorModel,
    ):
        pass

    @staticmethod
    def visualize(
        analysis,
        paths: AbstractPaths,
        instance,
        during_analysis,
    ):
        pass

    @staticmethod
    def visualize_before_fit_combined(
        analyses,
        paths: AbstractPaths,
        model: AbstractPriorModel,
    ):
        pass

    @staticmethod
    def visualize_combined(
        analyses,
        paths: AbstractPaths,
        instance,
        during_analysis,
    ):
        pass
