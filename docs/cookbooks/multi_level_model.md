(multi-level-model)=

# Multi Level Model

A multi level model is one where one or more of the input parameters in the model components `__init__`
constructor are Python classes, as opposed to a float or tuple.

The `af.Model()` object treats these Python classes as model components, enabling the composition of models where
model components are grouped within other Python classes, in an object oriented fashion.

This enables complex models which are intiutive and extensible to be composed.

This cookbook provides an overview of multi-level model composition.

**Contents:**

- **Python Class Template**: The template of multi level model components written as a Python class.
- **Model Composition**: How to compose a multi-level model using the `af.Model()` object.
- **Instances**: Creating an instance of a multi-level model via input parameters.
- **Why Use Multi-Level Models?**: A description of the benefits of using multi-level models compared to a `Collection`.
- **Model Customization**: Customizing a multi-level model (e.g. fixing parameters or linking them to one another).
- **Alternative API**: Alternative API for multi-level models which may be more concise and readable for certain models.
- **Json Output (Model)**: Output a multi-level model in human readable text via a .json file and loading it back again.

## Python Class Template

A multi-level model uses standard model components, which are written as a Python class with the usual format
where the inputs of the `__init__` constructor are the model parameters.

```python
class Gaussian:
    def __init__(
        self,
        normalization=1.0,  # <- **PyAutoFit** recognises these constructor arguments
        sigma=5.0,  # <- are the Gaussian``s model parameters.
    ):
        self.normalization = normalization
        self.sigma = sigma
```

The unique aspect of a multi-level model is that a Python class can then be defined where the inputs
of its `__init__` constructor are instances of these model components.

In the example below, the Python class which will be used to demonstrate a multi-level has an input `gaussian_list`,
which takes as input a list of instances of the `Gaussian` class above.

This class will represent many individual `Gaussian`'s, which share the same `centre` but have their own unique
`normalization` and `sigma` values.

```python
class MultiLevelGaussians:
    def __init__(
        self,
        higher_level_centre: float = 50.0,  # The centre of all Gaussians in the multi level component.
        gaussian_list: List[Gaussian] = None,  # Contains a list of Gaussians
    ):
        self.higher_level_centre = higher_level_centre

        self.gaussian_list = gaussian_list
```

## Model Composition

A multi-level model is instantiated via the af.Model() command, which is passed:

- `MultiLevelGaussians`: To tell it that the model component will be a `MultiLevelGaussians` object.
- `gaussian_list`: One or more `Gaussian`'s, each of which are created as an `af.Model()` object with free parameters.

```python
model = af.Model(
    MultiLevelGaussians, gaussian_list=[af.Model(Gaussian), af.Model(Gaussian)]
)
```

The multi-level model consists of two `Gaussian`'s, where their centres are shared as a parameter in the higher level
model component.

Total number of parameters is N=5 (x2 `normalizations`, ``` x2 ``sigma ```'s and x1 `higher_level_centre`).

```python
print(f"Model Total Free Parameters = {model.total_free_parameters}")
```

The structure of the multi-level model, including the hierarchy of Python classes, is shown in the `model.info`.

```python
print(model.info)
```

This gives the following output:

```bash
Total Free Parameters = 5

model                              MultiLevelGaussians (N=5)
    gaussian_list                  Collection (N=4)
        0                          Gaussian (N=2)
        1                          Gaussian (N=2)

higher_level_centre                UniformPrior [5], lower_limit = 0.0, upper_limit = 100.0
gaussian_list
    0
        normalization              LogUniformPrior [1], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                      UniformPrior [2], lower_limit = 0.0, upper_limit = 25.0
    1
        normalization              LogUniformPrior [3], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                      UniformPrior [4], lower_limit = 0.0, upper_limit = 25.0
```

## Instances

Instances of a multi-level model can be created, where an input `vector` of parameters is mapped to create an instance
of the Python class of the model.

We first need to know the order of parameters in the model, so we know how to define the input `vector`. This
information is contained in the models `paths` attribute.

```python
print(model.paths)
```

This gives the following output:

```bash
[
    ('gaussian_list', '0', 'normalization'),
    ('gaussian_list', '0', 'sigma'),
    ('gaussian_list', '1', 'normalization'),
    ('gaussian_list', '1', 'sigma'),
    ('higher_level_centre',)
]
```

We now create an instance via a multi-level model.

Its attributes are structured differently to models composed via the `Collection` object..

```python
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("Normalization (Gaussian 0) = ", instance.gaussian_list[0].normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_list[0].sigma)
print("Normalization (Gaussian 0) = ", instance.gaussian_list[1].normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_list[1].sigma)
print("Higher Level Centre= ", instance.higher_level_centre)
```

This gives the following output:

```bash
Model Instance:

<__main__.MultiLevelGaussians object at 0x7f5273ccd0f0>
Instance Parameters

Normalization (Gaussian 0) =  1.0
Sigma (Gaussian 0) =  2.0
Normalization (Gaussian 0) =  3.0
Sigma (Gaussian 0) =  4.0
Higher Level Centre=  5.0
```

## Why Use Multi Level Models?

An identical model in terms of functionality could of been created via the `Collection` object as follows:

```python
class GaussianCentre:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian``s model parameters.
        sigma=5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


model = af.Collection(gaussian_0=GaussianCentre, gaussian_1=GaussianCentre)

model.gaussian_0.centre = model.gaussian_1.centre
```

This raises the question of when to use a `Collection` and when to use multi-level models?

The answer depends on the structure of the models you are composing and fitting.

Many problems have models which have a natural multi-level structure.

For example, imagine a dataset had 3 separate groups of 1D `Gaussian`'s, where each group had multiple Gaussians with
a shared centre.

This model is concise and easy to define using the multi-level API:

```python
group_0 = af.Model(MultiLevelGaussians, gaussian_list=3 * [Gaussian])

group_1 = af.Model(MultiLevelGaussians, gaussian_list=3 * [Gaussian])

group_2 = af.Model(MultiLevelGaussians, gaussian_list=3 * [Gaussian])

model = af.Collection(group_0=group_0, group_1=group_1, group_2=group_2)
```

Composing the same model without the multi-level model is less concise, less readable and prone to error:

```python
group_0 = af.Collection(
    gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
)

group_0.gaussian_0.centre = group_0.gaussian_1.centre
group_0.gaussian_0.centre = group_0.gaussian_2.centre
group_0.gaussian_1.centre = group_0.gaussian_2.centre

group_1 = af.Collection(
    gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
)

group_1.gaussian_0.centre = group_1.gaussian_1.centre
group_1.gaussian_0.centre = group_1.gaussian_2.centre
group_1.gaussian_1.centre = group_1.gaussian_2.centre

group_2 = af.Collection(
    gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
)

group_2.gaussian_0.centre = group_2.gaussian_1.centre
group_2.gaussian_0.centre = group_2.gaussian_2.centre
group_2.gaussian_1.centre = group_2.gaussian_2.centre

model = af.Collection(group_0=group_0, group_1=group_1, group_2=group_2)
```

Here is what the `model.info` looks like:

```bash
Total Free Parameters = 21

model                              Collection (N=21)
    group_0                        MultiLevelGaussians (N=7)
        gaussian_list              Collection (N=6)
            0                      Gaussian (N=2)
            1                      Gaussian (N=2)
            2                      Gaussian (N=2)
    group_1                        MultiLevelGaussians (N=7)
        gaussian_list              Collection (N=6)
            0                      Gaussian (N=2)
            1                      Gaussian (N=2)
            2                      Gaussian (N=2)
    group_2                        MultiLevelGaussians (N=7)
        gaussian_list              Collection (N=6)
            0                      Gaussian (N=2)
            1                      Gaussian (N=2)
            2                      Gaussian (N=2)

group_0
    higher_level_centre            UniformPrior [6], lower_limit = 0.0, upper_limit = 100.0
    gaussian_list
        0
            normalization          LogUniformPrior [7], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [8], lower_limit = 0.0, upper_limit = 25.0
        1
            normalization          LogUniformPrior [9], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [10], lower_limit = 0.0, upper_limit = 25.0
        2
            normalization          LogUniformPrior [11], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [12], lower_limit = 0.0, upper_limit = 25.0
group_1
    higher_level_centre            UniformPrior [13], lower_limit = 0.0, upper_limit = 100.0
    gaussian_list
        0
            normalization          LogUniformPrior [14], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [15], lower_limit = 0.0, upper_limit = 25.0
        1
            normalization          LogUniformPrior [16], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [17], lower_limit = 0.0, upper_limit = 25.0
        2
            normalization          LogUniformPrior [18], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [19], lower_limit = 0.0, upper_limit = 25.0
group_2
    higher_level_centre            UniformPrior [20], lower_limit = 0.0, upper_limit = 100.0
    gaussian_list
        0
            normalization          LogUniformPrior [21], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [22], lower_limit = 0.0, upper_limit = 25.0
        1
            normalization          LogUniformPrior [23], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [24], lower_limit = 0.0, upper_limit = 25.0
        2
            normalization          LogUniformPrior [25], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                  UniformPrior [26], lower_limit = 0.0, upper_limit = 25.0
```

In many situations, multi-levels models are more extensible than the `Collection` API.

For example, imagine we wanted to add even more 1D profiles to a group with a shared `centre`. This can easily be
achieved using the multi-level API:

```python
multi = af.Model(
   MultiLevelGaussians,
   gaussian_list=[Gaussian, Gaussian, Exponential, YourProfileHere]
)
```

Composing the same model using just a `Model` and `Collection` is again possible, but would be even more cumbersome,
less readable and is not extensible.

## Model Customization

To customize the higher level parameters of a multi-level the usual model API is used:

```python
multi = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

multi.higher_level_centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
```

To customize a multi-level model instantiated via lists, each model component is accessed via its index:

```python
multi = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

group_level = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

group_level.gaussian_list[0].normalization = group_level.gaussian_list[1].normalization
```

Any combination of the API’s shown above can be used for customizing this model:

```python
gaussian_0 = af.Model(Gaussian)
gaussian_1 = af.Model(Gaussian)

gaussian_0.normalization = gaussian_1.normalization

group_level = af.Model(
    MultiLevelGaussians, gaussian_list=[gaussian_0, gaussian_1, af.Model(Gaussian)]
)

group_level.higher_level_centre = 1.0
group_level.gaussian_list[2].normalization = group_level.gaussian_list[1].normalization
```

Here is what the `model.info` looks like:

```bash
Total Free Parameters = 4

model                              MultiLevelGaussians (N=4)
    gaussian_list                  Collection (N=4)
        0                          Gaussian (N=2)
        1                          Gaussian (N=2)
        2                          Gaussian (N=2)

higher_level_centre                1.0
gaussian_list
    0
        normalization              LogUniformPrior [45], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                      UniformPrior [44], lower_limit = 0.0, upper_limit = 25.0
    1
        normalization              LogUniformPrior [45], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                      UniformPrior [46], lower_limit = 0.0, upper_limit = 25.0
    2
        normalization              LogUniformPrior [45], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                      UniformPrior [48], lower_limit = 0.0, upper_limit = 25.0
```

## Alternative API

A multi-level model can be instantiated where each model sub-component is setup using a name (as opposed to a list).

This means no list input parameter is required in the Python class of the model component, but we do need to include
the `**kwargs` input.

```python
class MultiLevelGaussians:
    def __init__(self, higher_level_centre=1.0, **kwargs):
        self.higher_level_centre = higher_level_centre


model = af.Model(
    MultiLevelGaussians, gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0])

print("Instance Parameters \n")
print("Normalization (Gaussian 0) = ", instance.gaussian_0.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_0.sigma)
print("Normalization (Gaussian 0) = ", instance.gaussian_1.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_1.sigma)
print("Higher Level Centre= ", instance.higher_level_centre)
```

This gives the following output:

```bash
Instance Parameters

Normalization (Gaussian 0) =  1.0
Sigma (Gaussian 0) =  2.0
Normalization (Gaussian 0) =  3.0
Sigma (Gaussian 0) =  4.0
Higher Level Centre=  5.0
```

The use of Python dictionaries illustrated in previous cookbooks can also be used with multi-level models.

```python
model_dict = {"gaussian_0": Gaussian, "gaussian_1": Gaussian}

model = af.Model(MultiLevelGaussians, **model_dict)

print(f"Multi-level Model Prior Count = {model.prior_count}")

instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0])

print("Instance Parameters \n")
print("Normalization (Gaussian 0) = ", instance.gaussian_0.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_0.sigma)
print("Normalization (Gaussian 0) = ", instance.gaussian_1.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_1.sigma)
print("Higher Level Centre= ", instance.higher_level_centre)
```

This gives the following output:

```bash
Instance Parameters

Normalization (Gaussian 0) =  1.0
Sigma (Gaussian 0) =  2.0
Normalization (Gaussian 0) =  3.0
Sigma (Gaussian 0) =  4.0
Higher Level Centre=  5.0
```

## JSon Outputs

A model has a `dict` attribute, which expresses all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.

```python
model = af.Model(Gaussian)

print(model.dict())
```

This gives the following output:

```bash
{
'class_path': '__main__.Gaussian', 'type': 'model',
'normalization': {'lower_limit': 1e-06, 'upper_limit': 1000000.0, 'type': 'LogUniform'},
'sigma': {'lower_limit': 0.0, 'upper_limit': 25.0, 'type': 'Uniform'}
}
```

The dictionary representation printed above can be saved to hard disk as a `.json` file.

This means we can save any multi-level model to hard-disk in a human readable format.

Checkout the file `autofit_workspace/*/cookbooks/jsons/group_level_model.json` to see the model written as a .json.

```python
model_path = path.join("scripts", "cookbooks", "jsons")

os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "multi_level_model.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)
```

We can load the model from its `.json` file, meaning that one can easily save a model to hard disk and load it
elsewhere.

```python
model = af.Model.from_json(file=model_file)
```

## Wrap Up

This cookbook shows how to multi-level models consisting of multiple components using the `af.Model()`
and `af.Collection()` objects.

You should think carefully about whether your model fitting problem can use multi-level models, as they can make
your model definition more concise and extensible.
