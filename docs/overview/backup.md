# Extending Models

The model composition API is designed to make composing complex models, consisting of multiple components with many
free parameters, straightforward and scalable.

To illustrate this, we will extend our model to include a second component, representing a symmetric 1D Exponential
profile, and fit it to data generated with both profiles.

Lets begin by loading and plotting this data.

```python
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)
xvalues = range(data.shape[0])
plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.show()
plt.close()
```

The data appear as follows:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/images/data_2.png
:alt: Alternative text
:width: 600
```

We define a Python class for the `Exponential` model component, exactly as we did for the `Gaussian` above.

```python
class Exponential:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Exponentials``s model parameters.
        rate=0.01,
    ):
        """
        Represents a symmetric 1D Exponential profile.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the profile.
        ratw
            The decay rate controlling has fast the Exponential declines.
        """

        self.centre = centre
        self.normalization = normalization
        self.rate = rate

    def model_data_from(self, xvalues: np.ndarray):
        """
        Returns the symmetric 1D Exponential on an input list of Cartesian
        x coordinates.

        The input xvalues are translated to a coordinate system centred on
        the Exponential, via its ``centre``.

        The output is referred to as the ``model_data`` to signify that it
        is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.
        """

        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )
```

We can easily compose a model consisting of 1 `Gaussian` object and 1 `Exponential` object using the `af.Collection`
object:

```python
model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))
```

A `Collection` behaves analogous to a `Model`, but it contains a multiple model components.

We can see this by printing its `paths` attribute, where paths to all 6 free parameters via both model components
are shown.

The paths have the entries `.gaussian.` and `.exponential.`, which correspond to the names we input into
the `af.Collection` above.

```python
print(model.paths)
```

The output is as follows:

```bash
[
    ('gaussian', 'centre'),
    ('gaussian', 'normalization'),
    ('gaussian', 'sigma'),
    ('exponential', 'centre'),
    ('exponential', 'normalization'),
    ('exponential', 'rate')
]
```

We can use the paths to customize the priors of each parameter.

```python
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
```

All of the information about the model created via the collection can be printed at once using its `info` attribute:

```python
print(model.info)
```

The output appears as follows:

```bash
Total Free Parameters = 6
model                                       Collection (N=6)
        gaussian                            Gaussian (N=3)
        exponential                         Exponential (N=3)

    gaussian
        centre                              UniformPrior [13], lower_limit = 0.0, upper_limit = 100.0
        normalization                       UniformPrior [14], lower_limit = 0.0, upper_limit = 100.0
        sigma                               UniformPrior [15], lower_limit = 0.0, upper_limit = 30.0
    exponential
        centre                              UniformPrior [16], lower_limit = 0.0, upper_limit = 100.0
        normalization                       UniformPrior [17], lower_limit = 0.0, upper_limit = 100.0
        rate                                UniformPrior [18], lower_limit = 0.0, upper_limit = 10.0
```

A model instance can again be created by mapping an input `vector`, which now has 6 entries.

```python
instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01])
```

This `instance` contains each of the model components we defined above.

The argument names input into the `Collection` define the attribute names of the `instance`:

```python
print("Instance Parameters \n")
print("x (Gaussian) = ", instance.gaussian.centre)
print("normalization (Gaussian) = ", instance.gaussian.normalization)
print("sigma (Gaussian) = ", instance.gaussian.sigma)
print("x (Exponential) = ", instance.exponential.centre)
print("normalization (Exponential) = ", instance.exponential.normalization)
print("sigma (Exponential) = ", instance.exponential.rate)
```

The output appear as follows:

```bash

```

The `Analysis` class above assumed the `instance` contained only a single model-component.

We update its `log_likelihood_function` to use both model components in the `instance` to fit the data.

```python
class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        The `Analysis` class acts as an interface between the data and
        model in **PyAutoFit**.

        Its `log_likelihood_function` defines how the model is fitted to
        the data and it is called many times by the non-linear search
        fitting algorithm.

        In this example the `Analysis` `__init__` constructor only
        contains the `data` and `noise-map`, but it can be easily
        extended to include other quantities.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal)
            fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data,
            used for computing the goodness of fit metric, the log likelihood.
        """

        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance) -> float:
        """
        Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

        The data is fitted using an `instance` of multiple 1D profiles
        (e.g. a `Gaussian`, `Exponential`) where
        their `model_data_from` methods are called and summed
        in order to create a model data representation that is fitted to the data.
        """

        """
        The `instance` that comes into this method is an instance of the
        `Gaussian` and `Exponential` models above, which were created
        via `af.Collection()`.

        It contains instances of every class we instantiated it with, where
        each instance is named following the names given to the Collection,
        which in this example is a `Gaussian` (with name `gaussian) and
        Exponential (with name `exponential`).

        The parameter values are again chosen by the non-linear search,
        based on where it thinks the high likelihood regions of parameter
        space are. The lines of Python code are commented out below to
        prevent excessive print statements.


        # print("Gaussian Instance:")
        # print("Centre = ", instance.gaussian.centre)
        # print("Normalization = ", instance.gaussian.normalization)
        # print("Sigma = ", instance.gaussian.sigma)

        # print("Exponential Instance:")
        # print("Centre = ", instance.exponential.centre)
        # print("Normalization = ", instance.exponential.normalization)
        # print("Rate = ", instance.exponential.rate)
        """
        """
        Get the range of x-values the data is defined on, to evaluate
        the model of the Gaussian.
        """
        xvalues = np.arange(self.data.shape[0])

        """
        Internally, the `instance` variable is a list of all model
        omponents pass to the `Collection` above.

        we can therefore iterate over them and use their
        `model_data_from` methods to create the
        summed overall model data.
        """
        model_data = sum(
            [
                profile_1d.model_data_from(xvalues=xvalues)
                for profile_1d in instance
            ]
        )

        """
        Fit the model gaussian line data to the observed data, computing the residuals, chi-squared and log likelihood.
        """
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood
```

We can now fit this model to the data using the same API we did before.

```python
analysis = Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic(
    nlive=100,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)
```

The `info` attribute shows the result in a readable format, showing that all 6 free parameters were fitted for.

```python
print(result.info)
```

The output appears as follows:

```bash
Bayesian Evidence                       144.86032973
Maximum Log Likelihood                  181.14287034
Maximum Log Posterior                   181.14287034

model                                   Collection (N=6)
    gaussian                            Gaussian (N=3)
    exponential                         Exponential (N=3)

Maximum Log Likelihood Model:

gaussian
    centre                              50.223
    normalization                       26.108
    sigma                               9.710
exponential
    centre                              50.057
    normalization                       39.948
    rate                                0.048


Summary (3.0 sigma limits):

gaussian
    centre                              50.27 (49.63, 50.88)
    normalization                       26.22 (21.37, 32.41)
    sigma                               9.75 (9.25, 10.27)
exponential
    centre                              50.04 (49.60, 50.50)
    normalization                       40.06 (37.60, 42.38)
    rate                                0.05 (0.04, 0.05)


Summary (1.0 sigma limits):

gaussian
    centre                              50.27 (50.08, 50.49)
    normalization                       26.22 (24.33, 28.39)
    sigma                               9.75 (9.60, 9.90)
exponential
    centre                              50.04 (49.90, 50.18)
    normalization                       40.06 (39.20, 40.88)
    rate                                0.05 (0.05, 0.05)
```

We can again use the max log likelihood instance to visualize the model data of the best fit model compared to the
data.

```python
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.model_data_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.model_data_from(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()
```

The plot appears as follows:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/images/toy_model_fit.png
:alt: Alternative text
:width: 600
```
