import numpy as np
from typing import Dict, Optional

import autofit as af

from autofit.example.result import ResultExample
from autofit.example.visualize import VisualizerExample

"""
The `analysis.py` module contains the dataset and log likelihood function which given a model instance (set up by
the non-linear search) fits the dataset and returns the log likelihood of that model.
"""


class LatentExample(af.Latent):
    """
    Example latent-variable catalogue, declared on `Analysis` as
    `Latent = LatentExample` (mirrors `Visualizer` / `Result`).

    A latent variable is not a model parameter but can be derived from it — here
    the full-width half maximum (FWHM) of the 1D Gaussian, derived from its
    `sigma`. Subclass `af.Latent` and override `keys` / `variables` to define
    your own; the values returned by `variables` are positionally aligned with
    `keys` and written to `latent/` alongside `samples.csv`.
    """

    @staticmethod
    def keys(analysis):
        return ["gaussian.fwhm"]

    @staticmethod
    def variables(analysis, parameters, model):
        instance = model.instance_from_vector(vector=parameters)
        try:
            return (instance.fwhm,)
        except AttributeError:
            try:
                return (instance[0].fwhm,)
            except AttributeError:
                return (instance[0].gaussian.fwhm,)


class Analysis(af.Analysis):

    """
    This over-write means the `Visualizer` class is used for visualization throughout the model-fit.

    This `VisualizerExample` object is in the `autofit.example.visualize` module and is used to customize the
    plots output during the model-fit.

    It has been extended with visualize methods that output visuals specific to the fitting of `1D` data.
    """
    Visualizer = VisualizerExample

    """
    This over-write means the `ResultExample` class is returned after the model-fit.

    This `ResultExample` object in the `autofit.example.result` module. 
    
    It has been extended, based on the model that is input into the analysis, to include a 
    property `max_log_likelihood_model_data`, which is the model data of the best-fit model.
    """
    Result = ResultExample

    Latent = LatentExample

    def __init__(
        self,
        data: np.ndarray,
        noise_map: np.ndarray,
        use_jax=False,
        share_model_data=False,
    ):
        """
        In this example the `Analysis` object only contains the data and noise-map. It can be easily extended,
        for more complex data-sets and model fitting problems.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D Gaussian) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
            metric.
        share_model_data
            If `True`, opt this `Analysis` into the `FactorGraphModel` cross-factor shared-state mechanism
            (see `shared_state_from`). This is only valid when the *entire* model is shared across every
            factor, so the model data is identical for all of them and can be computed once instead of being
            rebuilt by each factor. It is `False` by default, so the standard per-analysis behaviour is
            unchanged.
        """
        super().__init__(use_jax=use_jax)

        self.data = data
        self.noise_map = noise_map
        self.share_model_data = share_model_data

    def shared_state_from(self, instance: af.ModelInstance):
        """
        Compute the model data once so that it can be shared across the factors of a `FactorGraphModel`.

        This is the worked example of `Analysis.shared_state_from` (see that method). When every factor of
        the graph shares the *entire* model — for example several datasets fit by the same 1D profile via
        shared priors — the model data is identical for every factor, so it is wasteful to rebuild it once
        per factor. Returning it here means the `FactorGraphModel` computes it a single time on the lead
        factor and reuses it for all the others.

        In this toy the model data is cheap, but it stands in for an expensive shared computation: it is the
        1D analog of the lensing case, where the shared work (ray-tracing, the source-plane mapper, the
        mapping matrix and the curvature matrix) dominates the per-factor cost.

        Sharing is opt-in (`share_model_data`) because it is only correct when the model really is fully
        shared. If only some parameters are shared the model data differs between factors and this returns
        `None`, so each factor computes its own as usual.
        """
        if not self.share_model_data:
            return None

        return self.model_data_1d_from(instance=instance)

    def log_likelihood_function(
        self, instance: af.ModelInstance, shared=None, xp=np
    ) -> float:
        """
        Determine the log likelihood of a fit of multiple profiles to the dataset.

        Parameters
        ----------
        instance : af.Collection
            The model instances of the profiles.
        shared
            The model data shared across the factors of a `FactorGraphModel`, computed once by
            `shared_state_from` (see that method). When provided it is used directly instead of being
            recomputed here; when `None` (the default, e.g. a standalone fit) the model data is computed
            as normal.

        Returns
        -------
        The log likelihood value indicating how well this model fit the dataset.
        """
        if shared is None:
            model_data_1d = self.model_data_1d_from(instance=instance)
        else:
            model_data_1d = shared

        residual_map = self.data - model_data_1d
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood

    def model_data_1d_from(self, instance: af.ModelInstance) -> np.ndarray:
        """
        Returns the model data of a the 1D profiles.

        The way this is generated changes depending on if the model is a `Model` (therefore having only one profile)
        or a `Collection` (therefore having multiple profiles).

        If its a model, the model component's `model_data_from` is called and the output returned.
        For a collection, each components `model_data_from` is called, iterated through and summed
        to return the combined model data.

        Parameters
        ----------
        instance
            The model instance of the profile or collection of profiles.

        Returns
        -------
        The model data of the profiles.
        """

        xvalues = self._xp.arange(self.data.shape[0])
        model_data_1d = self._xp.zeros(self.data.shape[0])

        try:
            for profile in instance:
                try:
                    model_data_1d += profile.model_data_from(
                        xvalues=xvalues,
                        xp=self._xp
                    )
                except AttributeError:
                    pass
        except TypeError:
            model_data_1d += instance.model_data_from(xvalues=xvalues, xp=self._xp)

        return model_data_1d

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `files` folder such that they can be loaded after the analysis using PyAutoFit's database and
        aggregator tools.

        For this analysis the following are output:

        - The dataset's data.
        - The dataset's noise-map.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        paths.save_json(name="data", object_dict=self.data.tolist(), prefix="dataset")
        paths.save_json(
            name="noise_map", object_dict=self.noise_map.tolist(), prefix="dataset"
        )

    def make_result(
        self,
        samples_summary: af.SamplesSummary,
        paths: af.AbstractPaths,
        samples: Optional[af.SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ) -> Result:
        """
        Returns the `Result` of the non-linear search after it is completed.

        The result type is defined as a class variable in the `Analysis` class (see top of code under the python code
        `class Analysis(af.Analysis)`.

        The result can be manually overwritten by a user to return a user-defined result object, which can be extended
        with additional methods and attribute specific to the model-fit.

        This example class does example this, whereby the analysis result has been over written with the `ResultExample`
        class, which contains a property `max_log_likelihood_model_data_1d` that returns the model data of the
        best-fit model. This API means you can customize your result object to include whatever attributes you want
        and therefore make a result object specific to your model-fit and model-fitting problem.

        The `Result` object you return can be customized to include:

        - The samples summary, which contains the maximum log likelihood instance and median PDF model.

        - The paths of the search, which are used for loading the samples and search internal below when a search
        is resumed.

        - The samples of the non-linear search (e.g. MCMC chains) also stored in `samples.csv`.

        - The non-linear search used for the fit in its internal representation, which is used for resuming a search
        and making bespoke visualization using the search's internal results.

        - The analysis used to fit the model (default disabled to save memory, but option may be useful for certain
        projects).

        Parameters
        ----------
        samples_summary
            The summary of the samples of the non-linear search, which include the maximum log likelihood instance and
            median PDF model.
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        samples
            The samples of the non-linear search, for example the chains of an MCMC run.
        search_internal
            The internal representation of the non-linear search used to perform the model-fit.
        analysis
            The analysis used to fit the model.

        Returns
        -------
        Result
            The result of the non-linear search, which is defined as a class variable in the `Analysis` class.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=self,
        )
