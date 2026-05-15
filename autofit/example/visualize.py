import os
import numpy as np
from typing import List

import autofit as af


class VisualizerExample(af.Visualizer):
    """
    Methods associated with visualising analysis, model and data before, during
    or after an optimisation.
    """

    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        Before a model-fit begins, the `visualize_before_fit` method is called and is used to output images
        of quantities that do not change during the fit (e.g. the data).

        The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
        which is used to perform the visualization (e.g. it contains the data and noise map which are plotted).

        For your model-fitting problem this function will be overwritten with plotting functions specific to your
        problem.

        Parameters
        ----------
        analysis
            The analysis class used to perform the model-fit whose quantities are being visualized.
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        model
            The model which is fitted to the data, which may be used to customize the visualization.
        """

        import matplotlib.pyplot as plt

        xvalues = np.arange(analysis.data.shape[0])

        plt.errorbar(
            x=xvalues,
            y=analysis.data,
            yerr=analysis.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.title("The 1D Dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile normalization")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(paths.image_path / "data.png")
        plt.clf()
        plt.close()

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis : bool
    ):
        """
        During a model-fit, the `visualize` method is called throughout the non-linear search and is used to output
        images indicating the quality of the fit so far.

        The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
        which is used to perform the visualization (e.g. it generates the model data which is plotted).

        The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
        so far which can output on-the-fly images showing the best-fit model so far.

        For your model-fitting problem this function will be overwritten with plotting functions specific to your
        problem.

        Parameters
        ----------
        analysis
            The analysis class used to perform the model-fit whose quantities are being visualized.
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        during_analysis
            If True the visualization is being performed midway through the non-linear search before it is finished,
            which may change which images are output.
        """

        import matplotlib.pyplot as plt

        xvalues = np.arange(analysis.data.shape[0])
        model_data_1d_list = []

        try:
            for profile in instance:
                try:
                    model_data_1d_list.append(profile.model_data_from(xvalues=xvalues))
                except AttributeError:
                    pass
        except TypeError:
            model_data_1d_list.append(instance.model_data_from(xvalues=xvalues))

        model_data_1d = sum(model_data_1d_list)

        plt.errorbar(
            x=xvalues,
            y=analysis.data,
            yerr=analysis.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(xvalues, model_data_1d, color="r")

        for model_data_1d_profile in model_data_1d_list:

            plt.plot(xvalues, model_data_1d_profile, color="b", linestyle="--")

        plt.title("Model fit to multiple 1D profiles dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile normalization")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(paths.image_path / "model_fit.png")
        plt.clf()
        plt.close()

    @staticmethod
    def visualize_before_fit_combined(
        analyses,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        Multiple instances of the `Analysis` class can be summed together, meaning that the model is fitted to all
        datasets simultaneously via a summed likelihood function.

        The function receives as input a list of instances of every `Analysis` class which is being used to perform
        the summed analysis fit. This is used which is used to perform the visualization which combines the
        information spread across all analyses (e.g. plotting the data of each analysis on the same subplot).

        The `visualize_before_fit_combined` method is called before the model-fit begins and is used to output images
        of quantities that do not change during the fit (e.g. the data).

        When summed analysis is used, the `visualize_before_fit` method is also called for each individual analysis.
        Each individual dataset may therefore also be visualized in that function. This method is specifically for
        visualizing the combined information of all datasets.

        For your model-fitting problem this function will be overwritten with plotting functions specific to your
        problem.

        The example does not use analysis summing and therefore this function is not implemented.

        Parameters
        ----------
        analyses
            A list of the analysis classes used to perform the model-fit whose quantities are being visualized.
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        model
            The model which is fitted to the data, which may be used to customize the visualization.
        """
        pass

    @staticmethod
    def visualize_combined(
        analyses: List[af.Analysis],
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
        quick_update: bool = False,
    ):
        """
        Multiple instances of the `Analysis` class can be summed together, meaning that the model is fitted to all
        datasets simultaneously via a summed likelihood function.

        The function receives as input a list of instances of every `Analysis` class which is being used to perform
        the summed analysis fit. This is used which is used to perform the visualization which combines the
        information spread across all analyses (e.g. plotting the data of each analysis on the same subplot).

        The `visualize_combined` method is called throughout the non-linear search and is used to output images
        indicating the quality of the fit so far.

        When summed analysis is used, the `visualize_before_fit` method is also called for each individual analysis.
        Each individual dataset may therefore also be visualized in that function. This method is specifically for
        visualizing the combined information of all datasets.

        For your model-fitting problem this function will be overwritten with plotting functions specific to your
        problem.

        The example does not use analysis summing and therefore this function is not implemented.

        Parameters
        ----------
        analyses
            A list of the analysis classes used to perform the model-fit whose quantities are being visualized.
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        model
            The model which is fitted to the data, which may be used to customize the visualization.
        """
        pass