(results)=

# Results

A non-linear search fits a model to a dataset, returning a `Result` object that contains a lot of information on the
model-fit.

This cookbook provides a concise reference to the result API.

The cookbook then describes how the results of a search can be output to hard-disk and loaded back into Python,
either using the `Aggregator` object or by building an sqlite database of results. Result loading supports
queries, so that only the results of interest are returned.

The samples of the non-linear search, which are used to estimate quantities the maximum likelihood model and
parameter errors, are described separately in the `samples.py` cookbook.

**Contents:**

An overview of the `Result` object's functionality is given in the following sections:

> - **Info**: Print the `info` attribute of the `Result` object to display a summary of the model-fit.
> - **Max Log Likelihood Instance**: Getting the maximum likelihood model instance.
> - **Samples**: Getting the samples of the non-linear search from a result.
> - **Custom Result**: Extending the `Result` object with custom attributes specific to the model-fit.

The cookbook next describes how results can be output to hard-disk and loaded back into Python via the `Aggregator`:

> - **Output To Hard-Disk**: Output results to hard-disk so they can be inspected and used to restart a crashed search.
> - **Files**: The files that are stored in the `files` folder that is created when results are output to hard-disk.
> - **Loading From Hard-disk**: Loading results from hard-disk to Python variables via the aggregator.
> - **Generators**: Why loading results uses Python generators to ensure memory efficiency.

The cookbook next gives examples of how to load all the following results from the database:

> - **Samples**: The samples of the non-linear search (e.g. all parameter values, log likelihoods, etc.).
> - **Model**: The model fitted by the non-linear search.
> - **Search**: The search used to perform the model-fit.
> - **Samples Info**: Additional information on the samples.
> - **Samples Summary**: A summary of the samples of the non-linear search (e.g. the maximum log likelihood model).
> - **Info**: The `info` dictionary passed to the search.

The output of results to hard-disk is customizeable and described in the following section:

> - **Custom Output**: Extend `Analysis` classes to output additional information which can be loaded via the aggregator.

Using queries to load specific results is described in the following sections\*\*:

> - **Querying Datasets**: Query based on the name of the dataset.
> - **Querying Searches**: Query based on the name of the search.
> - **Querying Models**: Query based on the model that is fitted.
> - **Querying Results**: Query based on the results of the model-fit.
> - **Querying Logic**: Use logic to combine queries to load specific results (e.g. AND, OR, etc.).

The final section describes how to use results built in an sqlite database file:

> - **Database**: Building a database file from the output folder.
> - **Unique Identifiers**: The unique identifier of each model-fit.
> - **Writing Directly To Database**: Writing results directly to the database.

## Model Fit

To illustrate results, we need to perform a model-fit in order to create a `Result` object.

We do this below using the standard API and noisy 1D signal example, which you should be familiar with from other
example scripts.

Note that the `Gaussian` and `Analysis` classes come via the `af.ex` module, which contains example model components
that are identical to those found throughout the examples.

```python
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    nwalkers=30,
    nsteps=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)
```

## Info

Printing the `info` attribute shows the overall result of the model-fit in a human readable format.

```python
print(result.info)
```

The output appears as follows:

```bash
Maximum Log Likelihood              -46.68992727
Maximum Log Posterior               -46.64963514

model                               Gaussian (N=3)

Maximum Log Likelihood Model:

centre                              49.892
normalization                       24.819
sigma                               9.844


Summary (3.0 sigma limits):

centre                              49.89 (49.52, 50.23)
normalization                       24.79 (23.96, 25.61)
sigma                               9.85 (9.53, 10.21)


Summary (1.0 sigma limits):

centre                              49.89 (49.83, 49.96)
normalization                       24.79 (24.65, 24.94)
sigma                               9.85 (9.78, 9.90)
```

The `max_log_likelihood_instance` is the model instance of the maximum log likelihood model, which is the model
that maximizes the likelihood of the data given the model.

```python
instance = result.max_log_likelihood_instance

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", instance.centre)
print("Normalization = ", instance.normalization)
print("Sigma = ", instance.sigma)
```

The `Samples` class contains all information on the non-linear search samples, for example the value of every parameter
sampled using the fit or an instance of the maximum likelihood model.

```python
samples = result.samples
```

The samples are described in detail separately in the `samples.py` cookbook.

## Custom Result

The result can be can be customized to include additional information about the model-fit that is specific to your
model-fitting problem.

For example, for fitting 1D profiles, the `Result` could include the maximum log likelihood model 1D data:

`print(result.max_log_likelihood_model_data_1d)`

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
class Analysis(af.ex.Analysis):

    """
    This overwrite means the `ResultExample` class is returned after the model-fit.
    """

    Result = ResultExample

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
            analysis=self,
        )
```

Using the `Analysis` class above, the `Result` object returned by the search is now a `ResultExample` object.

```python
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    nwalkers=30,
    nsteps=1000,
)

result = search.fit(model=model, analysis=analysis)

print(result.max_log_likelihood_model_data_1d)
```

## Output To Hard-Disk

By default, a non-linear search does not output its results to hard-disk and its results can only be inspected
in Python via the `result` object.

However, the results of any non-linear search can be output to hard-disk by passing the `name` and / or `path_prefix`
attributes, which are used to name files and output the results to a folder on your hard-disk.

This cookbook now runs the three searches with output to hard-disk enabled, so you can see how the results are output
to hard-disk and to then illustrate how they can be loaded back into Python.

Note that an `info` dictionary is also passed to the search, which includes the date of the model-fit and the exposure
time of the dataset. This information is stored output to hard-disk and can be loaded to help interpret the results.

```python
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

dataset_name_list = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

model = af.Collection(gaussian=af.ex.Gaussian)

model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.gaussian.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

for dataset_name in dataset_name_list:
    dataset_path = path.join("dataset", "example_1d", dataset_name)

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    search = af.DynestyStatic(
        name="multi_result_example",
        path_prefix=path.join("cookbooks", "result"),
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        nlive=50,
    )

    print(
        """
        The non-linear search has begun running.
        This Jupyter notebook cell with progress once search has completed, this could take a few minutes!
        """
    )

    result = search.fit(model=model, analysis=analysis, info=info)

print("Search has finished run - you may now continue the notebook.")
```

## Files

By outputting results to hard-disk, a `files` folder is created containing .json / .csv files of the model,
samples, search, etc, for each fit.

You should check it out now for the completed fits on your hard-disk.

A description of all files is as follows:

> - `model`: The `model` defined above and used in the model-fit (`model.json`).
> - `search`: The non-linear search settings (`search.json`).
> - `samples`: The non-linear search samples (`samples.csv`).
> - `samples_info`: Additional information about the samples (`samples_info.json`).
> - `samples_summary`: A summary of key results of the samples (`samples_summary.json`).
> - `info`: The info dictionary passed to the search (`info.json`).
> - `covariance`: The inferred covariance matrix (`covariance.csv`).
> - `data`: The 1D noisy data used that is fitted (`data.json`).
> - `noise_map`: The 1D noise-map fitted (`noise_map.json`).

The `samples` and `samples_summary` results contain a lot of repeated information. The `samples` result contains
the full non-linear search samples, for example every parameter sample and its log likelihood. The `samples_summary`
contains a summary of the results, for example the maximum log likelihood model and error estimates on parameters
at 1 and 3 sigma confidence.

Accessing results via the `samples_summary` is much faster, because as it does not reperform calculations using the full
list of samples. Therefore, if the result you want is accessible via the `samples_summary` you should use it
but if not you can revert to the samples.

## Loading From Hard-Disk

The multi-fits above wrote the results to hard-disk in three distinct folders, one for each dataset.

Their results are loaded using the `Aggregator` object, which finds the results in the output directory and can
load them into Python objects.

```python
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("multi_result_example"),
)
```

## Generators

Before using the aggregator to inspect results, lets discuss Python generators.

A generator is an object that iterates over a function when it is called. The aggregator creates all of the objects
that it loads from the database as generators (as opposed to a list, or dictionary, or another Python type).

This is because generators are memory efficient, as they do not store the entries of the database in memory
simultaneously. This contrasts objects like lists and dictionaries, which store all entries in memory all at once.
If you fit a large number of datasets, lists and dictionaries will use a lot of memory and could crash your computer!

Once we use a generator in the Python code, it cannot be used again. To perform the same task twice, the
generator must be remade it. This cookbook therefore rarely stores generators as variables and instead uses the
aggregator to create each generator at the point of use.

To create a generator of a specific set of results, we use the `values` method. This takes the `name` of the
object we want to create a generator of, for example inputting `name=samples` will return the results `Samples`
object.

## Loading Samples

```python
samples_gen = agg.values("samples")
```

By converting this generator to a list and printing it, it is a list of 3 `SamplesNest` objects, corresponding to
the 3 model-fits performed above.

```python
print("Samples:\n")
print(samples_gen)
print("Total Samples Objects = ", len(agg), "\n")
```

## Loading Model

The model used to perform the model fit for each of the 3 datasets can be loaded via the aggregator and printed.

```python
model_gen = agg.values("model")

for model in model_gen:
    print(model.info)
```

## Loading Search

The non-linear search used to perform the model fit can be loaded via the aggregator and printed.

```python
search_gen = agg.values("search")

for search in search_gen:
    print(search.info)
```

## Loading Samples

The `Samples` class contains all information on the non-linear search samples, for example the value of every parameter
sampled using the fit or an instance of the maximum likelihood model.

The `Samples` class is described fully in the results cookbook.

```python
for samples in agg.values("samples"):

    print("The tenth sample`s third parameter")
    print(samples.parameter_lists[9][2], "\n")

    instance = samples.max_log_likelihood()

    print("Max Log Likelihood `Gaussian` Instance:")
    print("Centre = ", instance.centre)
    print("Normalization = ", instance.normalization)
    print("Sigma = ", instance.sigma, "\n")
```

## Loading Samples Summary

The samples summary contains a subset of results access via the `Samples`, for example the maximum likelihood model
and parameter error estimates.

Using the samples method above can be slow, as the quantities have to be computed from all non-linear search samples
(e.g. computing errors requires that all samples are marginalized over). This information is stored directly in the
samples summary and can therefore be accessed instantly.

```python
for samples_summary in agg.values("samples_summary"):

    instance = samples_summary.max_log_likelihood()

    print("Max Log Likelihood `Gaussian` Instance:")
    print("Centre = ", instance.centre)
    print("Normalization = ", instance.normalization)
    print("Sigma = ", instance.sigma, "\n")
```

## Loading Info

The info dictionary passed to the search, discussed earlier in this cookbook, is accessible.

```python
for info in agg.values("info"):
    print(info["date_of_observation"])
    print(info["exposure_time"])
```

The API for querying is fairly self explanatory. Through the combination of info based queries, model based
queries and result based queries a user has all the tools they need to fit extremely large datasets with many different
models and load only the results they are interested in for inspection and analysis.

## Custom Output

The results accessible via the database (e.g. `model`, `samples`) are those contained in the `files` folder.

By extending an `Analysis` class with the methods `save_attributes` and `save_results`,
custom files can be written to the `files` folder and become accessible via the database.

To save the objects in a human readable and loaded .json format, the `data` and `noise_map`, which are natively stored
as 1D numpy arrays, are converted to a suitable dictionary output format. This uses the **PyAutoConf** method
`to_dict`.

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
        Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis, it uses the `AnalysisDataset` object's method to output the following:

        - The dataset's data as a .json file.
        - The dataset's noise-map as a .json file.

        These are accessed using the aggregator via `agg.values("data")` and `agg.values("noise_map")`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        from autonerves.dictable import to_dict

        paths.save_json(name="data", object_dict=to_dict(self.data))
        paths.save_json(name="noise_map", object_dict=to_dict(self.noise_map))

    def save_results(self, paths: af.DirectoryPaths, result: af.Result):
        """
        At the end of a model-fit,  this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The maximum log likelihood model data as a .json file.

        This is accessed using the aggregator via `agg.values("model_data")`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples and maximum likelihood model.
        """
        xvalues = np.arange(self.data.shape[0])

        instance = result.max_log_likelihood_instance

        model_data = instance.model_data_from(xvalues=xvalues)

        # The path where model_data.json is saved, e.g. output/dataset_name/unique_id/files/model_data.json

        paths.save_json(name="model_data", object_dict=model_data)
```

## Querying Datasets

The aggregator can query the database, returning only specific fits of interested.

We can query using the `dataset_name` string we input into the model-fit above, in order to get the results
of a fit to a specific dataset.

For example, querying using the string `gaussian_x1_1` returns results for only the fit using the
second `Gaussian` dataset.

```python
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "gaussian_x1_1")
```

As expected, this list has only 1 `SamplesNest` corresponding to the second dataset.

```python
print(agg_query.values("samples"))
print("Total Samples Objects via dataset_name Query = ", len(agg_query), "\n")
```

If we query using an incorrect dataset name we get no results.

```python
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "incorrect_name")
samples_gen = agg_query.values("samples")
```

## Querying Searches

We can query using the `name` of the non-linear search used to fit the model.

In this cookbook, all three fits used the same search, named `database_example`. Query based on search name in this
example is therefore somewhat pointless.

However, querying based on the search name is useful for model-fits which use a range of searches, for example
if different non-linear searches are used multiple times.

As expected, the query using search name below contains all 3 results.

```python
name = agg.search.name
agg_query = agg.query(name == "database_example")

print(agg_query.values("samples"))
print("Total Samples Objects via name Query = ", len(agg_query), "\n")
```

## Querying Models

We can query based on the model fitted.

For example, we can load all results which fitted a `Gaussian` model-component, which in this simple example is all
3 model-fits.

Querying via the model is useful for loading results after performing many model-fits with many different model
parameterizations to large (e.g. Bayesian model comparison).

\[Note: the code `agg.model.gaussian` corresponds to the fact that in the `Collection` above, we named the model
component `gaussian`. If this `Collection` had used a different name the code below would change
correspondingly. Models with multiple model components (e.g., `gaussian` and `exponential`) are therefore also easily
accessed via the database.\]

```python
gaussian = agg.model.gaussian
agg_query = agg.query(gaussian == af.ex.Gaussian)
print("Total Samples Objects via `Gaussian` model query = ", len(agg_query), "\n")
```

## Querying Results

We can query based on the results of the model-fit.

Below, we query the database to find all fits where the inferred value of `sigma` for the `Gaussian` is less
than 3.0 (which returns only the first of the three model-fits).

```python
gaussian = agg.model.gaussian
agg_query = agg.query(gaussian.sigma < 3.0)
print("Total Samples Objects In Query `gaussian.sigma < 3.0` = ", len(agg_query), "\n")
```

## Querying with Logic

Advanced queries can be constructed using logic.

Below, we combine the two queries above to find all results which fitted a `Gaussian` AND (using the & symbol)
inferred a value of sigma less than 3.0.

The OR logical clause is also supported via the symbol .

```python
gaussian = agg.model.gaussian
agg_query = agg.query((gaussian == af.ex.Gaussian) & (gaussian.sigma < 3.0))
print(
    "Total Samples Objects In Query `Gaussian & sigma < 3.0` = ", len(agg_query), "\n"
)
```

## Database

The default behaviour of model-fitting results output is to be written to hard-disc in folders. These are simple to
navigate and manually check.

For small model-fitting tasks this is sufficient, however it does not scale well when performing many model fits to
large datasets, because manual inspection of results becomes time consuming.

All results can therefore be output to an sqlite3 (<https://docs.python.org/3/library/sqlite3.html>) relational database,
meaning that results can be loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation.
This database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can
be loaded.

## Unique Identifiers

We have discussed how every model-fit is given a unique identifier, which is used to ensure that the results of the
model-fit are output to a separate folder on hard-disk.

Each unique identifier is also used to define every entry of the database as it is built. Unique identifiers
therefore play the same vital role for the database of ensuring that every set of results written to it are unique.

## Building From Output Folder

The fits above wrote the results to hard-disk in folders, not as an .sqlite database file.

We build the database below, where the `database_name` corresponds to the name of your output folder and is also the
name of the `.sqlite` database file that is created.

If you are fitting a relatively small number of datasets (e.g. 10-100) having all results written to hard-disk (e.g.
for quick visual inspection) and using the database for sample wide analysis is beneficial.

We can optionally only include completed model-fits but setting `completed_only=True`.

If you inspect the `output` folder, you will see a `database.sqlite` file which contains the results.

```python
database_name = "database"

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

agg.add_directory(directory=path.join("output", "cookbooks", database_name))
```

## Writing Directly To Database

Results can be written directly to the .sqlite database file, skipping output to hard-disk entirely, by creating
a session and passing this to the non-linear search.

The code below shows how to do this, but it is commented out to avoid rerunning the non-linear searches.

This is ideal for tasks where model-fits to hundreds or thousands of datasets are performed, as it becomes unfeasible
to inspect the results of all fits on the hard-disk.

Our recommended workflow is to set up database analysis scripts using ~10 model-fits, and then scaling these up
to large samples by writing directly to the database.

```python
session = af.db.open_database("database.sqlite")

search = af.DynestyStatic(
    name="multi_result_example",
    path_prefix=path.join("cookbooks", "result"),
    unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
    session=session,  # This can instruct the search to write to the .sqlite database.
    nlive=50,
)
```

If you run the above code and inspect the `output` folder, you will see a `database.sqlite` file which contains
the results.

The API for loading a database and creating an aggregator to query is as follows:

```python
agg = af.Aggregator.from_database("database.sqlite")
```

Once we have the Aggregator, we can use it to query the database and load results as we did before.
