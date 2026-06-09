(analysis)=

# Analysis

The `Analysis` class is the interface between the data and model, whereby a `log_likelihood_function` is defined
and called by the non-linear search to fit the model.

This cookbook provides an overview of how to use and extend `Analysis` objects in **PyAutoFit**.

**Contents:**

- **Example**: A simple example of an analysis class which can be adapted for you use-case.
- **Customization**: Customizing an analysis class with different data inputs and editing the `log_likelihood_function`.
- **Visualization**: Using a `visualize` method so that model-specific visuals are output to hard-disk.
- **Custom Result**: Return a custom Result object with methods specific to your model fitting problem.
- **Latent Variables**: Declaring a `Latent` class on the analysis to output latent variables to hard-disk.
- **Custom Output**: Add methods which output model-specific results to hard-disk in the `files` folder (e.g. as .json files) to aid in the interpretation of results.

## Example

An example simple `Analysis` class, to remind ourselves of the basic structure and inputs.

This can be adapted for your use case.

```python
class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        The `Analysis` class acts as an interface between the data and model in **PyAutoFit**.

        Its `log_likelihood_function` defines how the model is fitted to the data and it is
        called many times by the non-linear search fitting algorithm.

        In this example the `Analysis` `__init__` constructor only contains the `data`
        and `noise-map`, but it can be easily extended to include other quantities.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the
            workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing
            the goodness of fit metric, the log likelihood.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance) -> float:
        """
        Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

        The data is fitted using an `instance` of the `Gaussian` class where
        its `model_data_from` is called in order to create a
        model data representation of the Gaussian that is fitted to the data.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map ** 2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood
```

An instance of the analysis class is created as follows.

```python
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = Analysis(data=data, noise_map=noise_map)
```

## Customization

The `Analysis` class can be fully customized to be suitable for your model-fit.

For example, additional inputs can be included in the `__init__` constructor and used in the `log_likelihood_function`.
if they are required for your `log_likelihood_function` to work.

The example below includes three additional inputs:

- Instead of inputting a `noise_map`, a `noise_covariance_matrix` is input, which means that corrrlated noise is
  : accounted for in the `log_likelihood_function`.
- A `mask` is input which masks the data such that certain data points are omitted from the log likelihood
- A `kernel` is input which can account for certain blurring operations during data acquisition.

```python
class Analysis(af.Analysis):
    def __init__(
            self,
            data: np.ndarray,
            noise_covariance_matrix: np.ndarray,
            mask: np.ndarray,
            kernel: np.ndarray
    ):
        """
        The `Analysis` class which has had its inputs edited for a different model-fit.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted
            in the workspace examples.
        noise_covariance_matrix
            A 2D numpy array containing the noise values and their covariances
            for the data, used for computing the
            goodness of fit whilst accounting for correlated noise.
        mask
            A 1D numpy array containing a mask, where `True` values mean a data
            point is masked and is omitted from
            the log likelihood.
        kernel
            A 1D numpy array containing the blurring kernel of the data, used
            for creating the model data.
        """
        super().__init__()

        self.data = data
        self.noise_covariance_matrix = noise_covariance_matrix
        self.mask = mask
        self.kernel = kernel

    def log_likelihood_function(self, instance) -> float:
        """
        The `log_likelihood_function` now has access to
        the  `noise_covariance_matrix`, `mask` and `kernel`, input above.
        """
        print(self.noise_covariance_matrix)
        print(self.mask)
        print(self.kernel)

        """
        We do not provide a specific example of how to use these inputs
        in the `log_likelihood_function` as they are specific to your
        model fitting problem.

        The key point is that any inputs required to compute the log
        likelihood can be passed into the `__init__` constructor of the
        `Analysis` class and used in the `log_likelihood_function`.
        """

        log_likelihood = None

        return log_likelihood
```

An instance of the analysis class is created as follows.

```python
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))

noise_covariance_matrix = np.ones(shape=(data.shape[0], data.shape[0]))
mask = np.full(fill_value=False, shape=data.shape)
kernel = np.full(fill_value=1.0, shape=data.shape)

analysis = Analysis(
    data=data, noise_covariance_matrix=noise_covariance_matrix, mask=mask, kernel=kernel
)
```

## Visualization

If a `name` is input into a non-linear search, all results are output to hard-disk in a folder.

By overwriting the `Visualizer` object of an `Analysis` class with a custom `Visualizer` class, custom results of the
model-fit can be visualized during the model-fit.

The `Visualizer` below has the methods `visualize_before_fit` and `visualize`, which perform model specific
visualization will also be output into an `image` folder, for example as `.png` files.

This uses the maximum log likelihood model of the model-fit inferred so far.

Visualization of the results of the search, such as the corner plot of what is called the "Probability Density
Function", are also automatically output during the model-fit on the fly.

```python
class Visualizer(af.Visualizer):

   @staticmethod
   def visualize_before_fit(
       analysis,
       paths: af.DirectoryPaths,
       model: af.AbstractPriorModel
   ):
       """
       Before a model-fit, the `visualize_before_fit` method is called to perform visualization.

       The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
       which is used to perform the visualization (e.g. it contains the data and noise map which are plotted).

       This can output visualization of quantities which do not change during the model-fit, for example the
       data and noise-map.

       The `paths` object contains the path to the folder where the visualization should be output, which is determined
       by the non-linear search `name` and other inputs.
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
       plt.title("Maximum Likelihood Fit")
       plt.xlabel("x value of profile")
       plt.ylabel("Profile Normalization")
       plt.savefig(path.join(paths.image_path, f"data.png"))
       plt.clf()

   @staticmethod
   def visualize(
       analysis,
       paths: af.DirectoryPaths,
       instance,
       during_analysis
   ):
       """
       During a model-fit, the `visualize` method is called throughout the non-linear search.

       The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
       which is used to perform the visualization (e.g. it generates the model data which is plotted).

       The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
       so far and it can be used to provide on-the-fly images showing how the model-fit is going.

       The `paths` object contains the path to the folder where the visualization should be output, which is determined
       by the non-linear search `name` and other inputs.
       """
       xvalues = np.arange(analysis.data.shape[0])

       model_data = instance.model_data_from(xvalues=xvalues)
       residual_map = analysis.data - model_data

       """
       The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
       """
       import matplotlib.pyplot as plt

       plt.errorbar(
           x=xvalues,
           y=analysis.data,
           yerr=analysis.noise_map,
           color="k",
           ecolor="k",
           elinewidth=1,
           capsize=2,
       )
       plt.plot(xvalues, model_data, color="r")
       plt.title("Maximum Likelihood Fit")
       plt.xlabel("x value of profile")
       plt.ylabel("Profile Normalization")
       plt.savefig(path.join(paths.image_path, f"model_fit.png"))
       plt.clf()

       plt.errorbar(
           x=xvalues,
           y=residual_map,
           yerr=analysis.noise_map,
           color="k",
           ecolor="k",
           elinewidth=1,
           capsize=2,
       )
       plt.title("Residuals of Maximum Likelihood Fit")
       plt.xlabel("x value of profile")
       plt.ylabel("Residual")
       plt.savefig(path.join(paths.image_path, f"model_fit.png"))
       plt.clf()
```

The `Analysis` class is defined following the same API as before, but now with its `Visualizer` class attribute
overwritten with the `Visualizer` class above.

```python
class Analysis(af.Analysis):

    """
    This over-write means the `Visualizer` class is used for visualization throughout the model-fit.

    This `VisualizerExample` object is in the `autofit.example.visualize` module and is used to customize the
    plots output during the model-fit.

    It has been extended with visualize methods that output visuals specific to the fitting of `1D` data.
    """
    Visualizer = Visualizer

    def __init__(self, data, noise_map):
        """
        An Analysis class which illustrates visualization.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `log_likelihood_function` is identical to the example above
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood
```

## Custom Result

The `Result` object is returned by a non-linear search after running the following code:

```python
result = search.fit(model=model, analysis=analysis)
```

The result can be can be customized to include additional information about the model-fit that is specific to your
model-fitting problem.

For example, for fitting 1D profiles, the `Result` could include the maximum log likelihood model 1D data:

```python
print(result.max_log_likelihood_model_data_1d)
```

In other examples, this quantity has been manually computed after the model-fit has completed.

The custom result API allows us to do this. First, we define a custom `Result` class, which includes the property
`max_log_likelihood_model_data_1d`.

```python
class ResultExample(af.Result):

    @property
    def max_log_likelihood_model_data_1d(self) -> np.ndarray:
        """
        Returns the maximum log likelihood model's 1D model data.

        This is an example of how we can pass the `Analysis` class a custom `Result` object and extend this result
        object with new properties that are specific to the model-fit we are performing.
        """
        xvalues = np.arange(self.analysis.data.shape[0])

        return self.instance.model_data_from(xvalues=xvalues)
```

The custom result has access to the analysis class, meaning that we can use any of its methods or properties to
compute custom result properties.

To make it so that the `ResultExample` object above is returned by the search we overwrite the `Result` class attribute
of the `Analysis` and define a `make_result` object describing what we want it to contain:

```python
class Analysis(af.Analysis):

    """
    This overwrite means the `ResultExample` class is returned after the model-fit.
    """
    Result = ResultExample

    def __init__(self, data, noise_map):
        """
        An Analysis class which illustrates custom results.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `log_likelihood_function` is identical to the example above
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

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

        This example class does example this, whereby the analysis result has been overwritten with the `ResultExample`
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
            analysis=self
        )
```

For the sake of brevity, we do not run the code below, but the following code would work:

```python
result = search.fit(model=model, analysis=analysis)
print(result.max_log_likelihood_model_data_1d)
```

## Latent Variables

A latent variable is not a model parameter but can be derived from the model. Its value and errors may be of interest
and aid in the interpretation of a model-fit.

For example, for the simple 1D Gaussian example, it could be the full-width half maximum (FWHM) of the Gaussian.
This is not included in the model but can be easily derived from the Gaussian's sigma value.

We specify latent variables with a `Latent` class (subclass of `af.Latent`), declared on the `Analysis` via the
`Latent` class attribute — the same first-class, swappable mechanism used for the `Visualizer` and `Result`.
Override `keys` (the latent names) and `variables` (their values, positionally aligned with `keys`). If the
search has a `name`, these are output to a `latent.csv` file, which mirrors the `samples.csv` file.

There may also be a `latent.results` and `latent_summary.json` files output. The `output.yaml` config file
contains settings customizing what files are output and how often.

```python
class LatentFwhm(af.Latent):
    @staticmethod
    def keys(analysis):
        return ["gaussian.fwhm"]

    @staticmethod
    def variables(analysis, parameters, model):
        # Called per sample: build the instance from the parameter vector and
        # return a tuple of latent values, positionally aligned with `keys`.
        # `parameters` (a raw vector) rather than `instance` so it is JAX-jit /
        # vmap compatible.
        instance = model.instance_from_vector(vector=parameters)
        return (instance.fwhm,)


class Analysis(af.Analysis):

    Latent = LatentFwhm

    def __init__(self, data, noise_map):
        """
        An Analysis class which illustrates latent variables.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `log_likelihood_function` is identical to the example above
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood
```

Outputting latent variables manually after a fit is complete is simple, just call
the `analysis.compute_latent_samples(result.samples)` function.

For many use cases, the best set disables autofit latent variable output during a fit via
the `output.yaml` file and perform it manually after completing a successful model-fit. This will save computational
run time by not computing latent variables during a any model-fit which is unsuccessful.

```python
analysis = Analysis(data=data, noise_map=noise_map)

# You need to have run a fit to retrieve a result to do this.

analysis.compute_latent_samples(result.samples)
```

Analysing and interpreting latent variables is described fully in the result cookbook.

However, in brief, the `latent_samples` object is a `Samples` object and uses the same API as samples objects.

```python
print(latent_samples.median_pdf().fwhm)
```

## Custom Output

When performing fits which output results to hard-disc, a `files` folder is created containing .json / .csv files of
the model, samples, search, etc.

These files are human readable and help one quickly inspect and interpret results.

By extending an `Analysis` class with the methods `save_attributes` and `save_results`,
custom files can be written to the `files` folder to further aid this inspection.

These files can then also be loaded via the database, as described in the database cookbook.

```python
class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        Standard Analysis class example used throughout PyAutoFit examples.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance) -> float:
        """
        Standard log likelihood function used throughout PyAutoFit examples.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the non-linear search begins, this routine saves attributes
        of the `Analysis` object to the `files` folder such that they can
        be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis, it uses the `AnalysisDataset` object's method to
        output the following:

        - The dataset's data as a .json file.
        - The dataset's noise-map as a .json file.

        These are accessed using the aggregator via `agg.values("data")`
        and `agg.values("noise_map")`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where
            the non-linear search outputs are stored, visualization, and the
            pickled objects used by the aggregator output by this function.
        """
        # The path where data.json is saved, e.g. output/dataset_name/unique_id/files/data.json

        file_path = paths._files_path / "data.json"

        with open(file_path, "w+") as f:
            json.dump(self.data.tolist(), f, indent=4)

        # The path where noise_map.json is saved, e.g. output/noise_mapset_name/unique_id/files/noise_map.json

        file_path = paths._files_path / "noise_map.json"

        with open(file_path, "w+") as f:
            json.dump(self.noise_map.tolist(), f, indent=4)

    def save_results(self, paths: af.DirectoryPaths, result: af.Result):
        """
        At the end of a model-fit,  this routine saves attributes of the `Analysis`
        object to the `files` folder such that they can be loaded after the analysis
        using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The maximum log likelihood model data as a .json file.

        This is accessed using the aggregator via `agg.values("model_data")`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the
            non-linear search outputs are stored, visualization and the pickled
            objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples
            and maximum likelihood model.
        """
        xvalues = np.arange(self.data.shape[0])

        instance = result.max_log_likelihood_instance

        model_data = instance.model_data_from(xvalues=xvalues)

        # The path where model_data.json is saved, e.g. output/dataset_name/unique_id/files/model_data.json

        file_path = (path.join(paths._files_path, "model_data.json"),)

        with open(file_path, "w+") as f:
            json.dump(model_data, f, indent=4)
```
