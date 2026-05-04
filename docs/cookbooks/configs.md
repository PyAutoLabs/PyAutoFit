(configs)=

# Configs

Configuration files are used to control the behaviour model components in **PyAutoFit**, which perform the
following tasks:

- Specify the default priors of model components, so that a user does not have to manually specify priors every time they create a model.
- Specify labels of every parameter, which are used for plotting and visualizing results.

This cookbook illustrates how to create configuration files for your own model components, so that they can be used
with **PyAutoFit**.

**Contents:**

- **No Config Behaviour**: An example of what happens when a model component does not have a config file.
- **Template**: A template config file for specifying default model component priors.
- **Modules**: Writing prior config files based on the Python module the model component Python class is contained in.
- **Labels**: Config files which specify the labels of model component parameters for visualization.

## No Config Behaviour

The examples seen so far have used `Gaussian` and `Exponential` model components, which have configuration files in
the `autofit_workspace/config/priors` folder which define their priors and labels.

If a model component does not have a configuration file and we try to use it in a fit, **PyAutoFit** will raise an
error.

Lets illustrate this by setting up the usual Gaussian object, but naming it `GaussianNoConfig` so that it does
not have a config file.

```python
class GaussianNoConfig:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        """
        Represents a 1D `Gaussian` profile, which does not have a config file set up.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_from(self, xvalues: np.ndarray) -> np.ndarray:
        """
        The usual method that returns the 1D data of the `Gaussian` profile.
        """
        transformed_xvalues = xvalues - self.centre

        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )
```

When we try make this a `Model` and fit it, **PyAutoFit** raises an error, as it does not know where the priors
of the `GaussianNoConfig` are located.

```python
model = af.Model(GaussianNoConfig)

dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic()

result = search.fit(model=model, analysis=analysis)
```

In all other examples, the fits runs because the priors have been defined in one of two ways:

- They were manually input in the example script.
- They were loaded via config files "behind the scenes".

Checkout the folder `autofit_workspace/config/priors`, where .yaml files defining the priors of the `Gaussian` and
`Exponential` model components are located. These are the config files that **PyAutoFit** loads in the background
in order to setup the default priors of these model components.

If we do not manually override priors, these are the priors that will be used by default when a model-fit is performed.

## Templates

For your model-fitting task, you therefore should set up a config file for every model component you defining its
default priors.

Next, inspect the `TemplateObject.yaml` priors configuration file in `autofit_workspace/config/priors`.

You should see the following text:

```bash
parameter0:
  type: Uniform
  lower_limit: 0.0
  upper_limit: 1.0
parameter1:
  type: TruncatedGaussian
  mean: 0.0
  sigma: 0.1
  lower_limit: 0.0
  upper_limit: inf
parameter2:
  type: Uniform
  lower_limit: 0.0
  upper_limit: 10.0
```

This specifies the default priors on two parameters, named `parameter0` and `parameter1`.

The `type` is the type of prior assumed by **PyAutoFit** by default for its corresponding parameter, where in this
example:

- `parameter0` is given a `UniformPrior` with limits between 0.0 and 1.0.
- `parameter1` a `GaussianPrior` with mean 0.0 and sigma 1.0.
- `parameter2` is given a `UniformPrior` with limits between 0.0 and 10.0.

The `lower_limit` and `upper_limit` of a `GaussianPrior` define the boundaries of what parameter values are
physically allowed. If a model-component is given a value outside these limits during model-fitting the model is
instantly resampled and discarded.

We can easily adapt this template for any model component, for example the `GaussianNoConfig`.

First, copy and paste the `TemplateObject.yaml` file to create a new file called `GaussianNoConfig.yaml`.

The name of the class is matched to the name of the configuration file, therefore it is a requirement that the
configuration file is named `GaussianNoConfig.yaml` so that **PyAutoFit** can associate it with the `GaussianNoConfig`
Python class.

Now perform the follow changes to the `.yaml` file:

- Rename `parameter0` to `centre` and updates its uniform prior to be from a `lower_limit` of 0.0 and an `upper_limit` of 100.0.
- Rename `parameter1` to `normalization`.
- Rename `parameter2` to `sigma`.

The `.yaml` file should read as follows:

```bash
centre:
  type: Uniform
  lower_limit: 0.0
  upper_limit: 100.0
normalization:
  type: TruncatedGaussian
  mean: 0.0
  sigma: 0.1
  lower_limit: 0.0
  upper_limit: inf
sigma:
  type: Uniform
  lower_limit: 0.0
  upper_limit: 10.0
```

We should now be able to make a `Model` of the `GaussianNoConfig` class and fit it, without manually specifying
the priors.

You may need to reset your Jupyter notebook's kernel for the changes to the `.yaml` file to take effect.

```python
model = af.Model(GaussianNoConfig)

dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic()

result = search.fit(model=model, analysis=analysis)
```

## Modules

For larger projects, it may not be ideal to have to write a .yaml file for every Python class which acts as a model
component.

We instead would prefer them to be in their own dedicated Python module.

Suppose the `Gaussian` and `Exponential` model components were contained in a module named `profiles.py` in your
project's source code.

You could then write a priors .yaml config file following the format given in the example config file
`autofit_workspace/config/priors/profiles.yaml`, noting that there is a paring between the module name
(`profiles.py`) and the name of the `.yaml` file (`profiles.yaml`).

The file `autofit_workspace/config/priors/template_module.yaml` provides the tempolate for module based prior
configs and reads as follows:

```bash
ModelComponent0:
  parameter0:
    type: Uniform
    lower_limit: 0.0
    upper_limit: 1.0
  parameter1:
    type: LogUniform
    lower_limit: 1.0e-06
    upper_limit: 1000000.0
  parameter2:
    type: Uniform
    lower_limit: 0.0
    upper_limit: 25.0
ModelComponent1:
  parameter0:
    type: Uniform
    lower_limit: 0.0
    upper_limit: 1.0
  parameter1:
    type: LogUniform
    lower_limit: 1.0e-06
    upper_limit: 1000000.0
  parameter2:
    type: Uniform
    lower_limit: 0.0
    upper_limit: 1.0
```

This looks very similar to `TemplateObject`, the only differences are:

- It now contains the model-component class name in the configuration file, e.g. `ModelComponent0`, `ModelComponent1`.
- It includes multiple model-components, whereas `TemplateObject.yaml` corresponded to only one model component.

## Labels

There is an optional configs which associate model parameters with labels:

`autofit_workspace/config/notation.yaml`

It includes a `label` section which pairs every parameter with a label, which is used when visualizing results
(e.g. these labels are used when creating a corner plot).

```bash
label:
  label:
    sigma: \sigma
    centre: x
    normalization: norm
    parameter0: a
    parameter1: b
    parameter2: c
    rate: \lambda
```

It also contains a `superscript` section which pairs every model-component label with a superscript, so that
models with the same parameter names (e.g. `centre` can be distinguished).

```bash
label:
  superscript:
    Exponential: e
    Gaussian: g
    ModelComponent0: M0
    ModelComponent1: M1
```

The `label_format` section sets Python formatting options for every parameter, controlling how they display in
the `model.results` file.

```bash
label_format:
  format:
    sigma: '{:.2f}'
    centre: '{:.2f}'
    normalization: '{:.2f}'
    parameter0: '{:.2f}'
    parameter1: '{:.2f}'
    parameter2: '{:.2f}'
    rate: '{:.2f}'
```
