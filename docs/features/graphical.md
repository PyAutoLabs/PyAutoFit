(graphical)=

# Graphical Models

Throughout most examples, we compose a model and fit it to a single dataset. For simple model-fitting tasks this is
sufficient, however it is common for one to have multiple datasets and a desire to fit them simultaneously with a
unified model.

This is what graphical models enable. Here, we will show how to build a graphical model that fits multiple datasets
with **PyAutoFit**.

Using graphical models, **PyAutoFit** can compose and fit models that have 'local' parameters specific to each individual
dataset and higher-level model components that fit 'global' parameters. These higher level parameters will have
conditional dependencies with the local parameters.

The major selling point of **PyAutoFit**'s graphical modeling framework is the high level of customization it offers,
whereby:

- Specific `Analysis` classes can be defined for fitting differnent local models to different datasets.
- Each pairing of a local model-fit to data can be given its own non-linear search.
- Graphical model networks of any topology can be defined and fitted.

In this example, we demonstrate the API for composing and fitting a graphical model to multiple-datasets, using the
simple example of fitting noisy 1D Gaussians.

We begin by loading noisy 1D data containing 3 Gaussian's.

```bash
total_gaussians = 3

dataset_path = path.join("dataset", "example_1d")

data_list = []
noise_map_list = []

for dataset_index in range(total_gaussians):

    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    data_list.append(data)
    noise_map_list.append(noise_map)
```

This is what our three Gaussians look like:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/features/images/gaussian_x1_1__low_snr.png
:alt: Alternative text
:width: 600
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/features/images/gaussian_x1_2__low_snr.png
:alt: Alternative text
:width: 600
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/features/images/gaussian_x1_3__low_snr.png
:alt: Alternative text
:width: 600
```

They are much lower signal-to-noise than the Gaussian's in other examples. Graphical models extract a lot more information
from lower quantity datasets, something we demonstrate explic in the [HowToFit lectures on graphical models](https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models).

For each dataset we now create a corresponding `Analysis` class. By associating each dataset with an `Analysis`
class we are therefore associating it with a unique `log_likelihood_function`. If our dataset had many different
formats (e.g. images) it would be straight forward to write customized `Analysis` classes for each dataset.

```bash
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):

    analysis = Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)
```

We now compose the graphical model we will fit using the `Model` and `Collection` objects. We begin by setting up a
shared prior for their `centre` using a single `GaussianPrior`. This is passed to a unique `Model` for
each `Gaussian` and means that all three `Gaussian`'s are fitted wih the same value of `centre`. That is, we have
defined our graphical model to have a shared value of `centre` when it fits each dataset.

```bash
centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)
```

We now set up three `Model` objects, each of which contain a `Gaussian` that is used to fit each of the
datasets we loaded above. Because all three of these `Model`'s use the `centre_shared_prior` the dimensionality of
parameter space is N=7, corresponding to three `Gaussians` with local parameters (`normalization` and `sigma`) and
a global parameter value of `centre`.

```bash
model_list = []

for model_index in range(len(data_list)):

    gaussian = af.Model(p.Gaussian)

    gaussian.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!
    gaussian.normalization = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

    model_list.append(gaussian)
```

To build our graphical model which fits multiple datasets, we simply pair each model-component to each `Analysis`
class, so that **PyAutoFit** knows that:

- `gaussian_0` fits `data_0` via `analysis_0`.
- `gaussian_1` fits `data_1` via `analysis_1`.
- `gaussian_2` fits `data_2` via `analysis_2`.

The point where a `Model` and `Analysis` class meet is called a `AnalysisFactor`.

This term is used to denote that we are composing a 'factor graph'. A factor defines a node on this graph where we have
some data, a model, and we fit the two together. The 'links' between these different factors then define the global
model we are fitting **and** the datasets used to fit it.

```bash
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):

    analysis_factor = g.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)
```

We combine our `AnalysisFactor`'s into one, to compose the factor graph.

```bash
factor_graph = g.FactorGraphModel(*analysis_factor_list)
```

So, what does our factor graph look like? The `ModelPlotter` draws it, taking the factor graph's global model exactly
like it takes an ordinary model:

```bash
af.ModelPlotter(factor_graph.global_prior_model).figure()
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/images/model_figures/graphical_variable.png
:alt: The model figure of the graphical model above, with the shared centre hoisted into a card above a dashed plate standing for the three datasets.
:width: 600
```

The dashed frame is a **plate**: one `Gaussian` drawn once, standing for all three datasets, labelled `3 datasets`. The
`centre` shared by every dataset is hoisted out into the card above, joined to the plate by a blue line, and its chip
inside the plate carries a blue `shared` badge. The `normalization` and `sigma` chips are marked `independent`, because
each dataset gets a prior of its own, and the green `data` chip is the observed dataset. The footer counts the fit the
prose below describes: 1 shared across datasets, 2 per dataset × 3 datasets, 7 unique sampled scalars.

The factor graph above is made up of two components:

- **Nodes**: these are points on the graph where we have a unique set of data and a model that is made up of a subset of

our overall graphical model. This is effectively the `AnalysisFactor` objects we created above.

- **Links**: these define the model components and parameters that are shared across different nodes and thus retain the

same values when fitting different datasets.

We can now choose a non-linear search and fit the factor graph.

```bash
search = af.DynestyStatic()

result = search.fit(
    model=factor_graph.global_prior_model,
    analysis=factor_graph
)
```

This will fit the N=7 dimension parameter space where every Gaussian has a shared centre!

This is all expanded upon in the [HowToFit chapter on graphical models](https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models), where we will give a
more detailed description of why this approach to model-fitting extracts a lot more information than fitting each
dataset one-by-one.

## Shared Or Hierarchical

Sharing a parameter and drawing it hierarchically sound alike in words but are different models, and the figure is
the quickest way to tell them apart.

A **shared** parameter is one prior object used by every dataset, so the datasets fit literally the same number. Share
all three of `centre`, `normalization` and `sigma` and every chip carries the blue `shared` badge, the hoisted card
holds all three, and 3 datasets cost 3 sampled scalars:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/images/model_figures/graphical_shared.png
:alt: The model figure of a fully shared graphical model, with centre, normalization and sigma hoisted into one shared card and each chip badged shared.
:width: 600
```

A **hierarchical** parameter is different for every dataset, but each one is *drawn* from a parent distribution whose
own parameters are fitted. Drawn is not shared. Give each dataset's `centre` an `af.HierarchicalFactor` parent and the
figure hoists that parent into a violet card of its own, holding the `mean` and `sigma` of the distribution, with a
violet arrow running into the `centre` chip, which reads `centre · drawn`:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/images/model_figures/graphical_hierarchical.png
:alt: The model figure of a hierarchical graphical model, with a violet HierarchicalFactor card holding mean and sigma and a violet arrow into the drawn centre chip inside the plate.
:width: 600
```

The footers say what this costs: the shared figure counts 3 unique sampled scalars, the hierarchical one 2
hyper-parameters plus 3 per dataset × 3 datasets, so 11. This hierarchical composition is the model built in
[HowToFit chapter 3, tutorial 4](https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models/tutorial_4_hierachical_models.ipynb).

## Expectation Propagation

For large datasets, a graphical model may have hundreds, thousands, or *hundreds of thousands* of parameters. The
high dimensionality of such a parameter space can make it inefficient or impossible to fit the model.

Fitting high dimensionality graphical models in **PyAutoFit** can use an Expectation Propagation (EP) framework to
make scaling up feasible. This framework fits every dataset individually and pass messages throughout the graph to
inform every fit the expected
values of each parameter.

The following paper describes the EP framework in formal Bayesian notation:

<https://arxiv.org/pdf/1412.4869.pdf>

## Seeing The EP Run

An EP fit sweeps factor by factor for as long as it takes, and the thing that most often goes wrong is invisible in the
result: one factor whose update is rejected every sweep, whose reported posterior is therefore the message it started
with. `af.EPPlotter` draws the factor graph the `EPOptimiser` sweeps so that this is something you look at rather than
something you infer.

An `EPOptimiser` given `paths` writes two figures into its output directory, beside `graph.info`. `graph_model.png` is
the structure alone, written once at the start of the run:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/images/model_figures/ep_model.png
:alt: The EP model figure of the hierarchical graphical model, with square factor nodes, round variable pills, a dashed plate for the three datasets and the HierarchicalFactor above it.
:width: 600
```

Square boxes are factors, rounded pills are variables, and a line is an incidence: this variable is one of that
factor's arguments. The dashed frame is a plate, exactly as in the model figures above -- three `AnalysisFactor`s with
the same signature are drawn once, and a variable each of them has its own copy of is drawn once inside the plate
badged `x3`. The `HierarchicalFactor0` box collapses its three members the same way, reading `3 members`, with its own
`mean` and `sigma` as hyper-variables above it.

`graph_state.png` is the same graph with the run painted on it, rewritten on every `visualise_interval` tick, so the
file on disk always shows the sweep that just finished. Drawing it costs about a third of a second the first time and
under a tenth thereafter, so the interval is worth setting deliberately on a long run rather than left at 1:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/images/model_figures/ep_state_stale.png
:alt: The EP state figure of a three dataset run, with the plate noting one of three stale and the stalled AnalysisFactor2 expanded beside it in grey, badged zero updates in four sweeps.
:width: 600
```

Each factor carries its updates and sweeps, and its state. **Stale** is grey: the factor completed sweeps and none of
them updated it -- the same condition the optimiser prints as its `STALE FACTORS` warning at the end of a run.
**Reverted** is a dashed red edge, and it marks a `(factor, variable)` pair whose projection was confirmed rejected,
which is a stronger statement than a message that happens not to have moved: EP restarted from its own converged mean
field reproduces every message exactly and is marked nowhere. **Converged** is a green outline, from the same
convergence test `EPHistory` stops the fit on.

A plate never hides the member that failed. It shows the aggregate -- `3 datasets`, `4 updates / 4 sweeps` -- and then
names every member that departs from it, in red beneath the title, and draws that member as a node of its own beside
the plate. The figure above says `1 of 3 stale: AnalysisFactor2` and then draws `AnalysisFactor2`, grey, badged
`0 updates / 4 sweeps` with its age and its `BAD_PROJECTION` flag. A stale factor in a plate of thirty is as visible as
one in a plate of three.

The figures are behind `output.yaml`'s `model_figure` key -- the same key the per-search `model.png` is behind -- which
is read strictly, so a configuration that does not mention it writes neither file. Switch it on to get them:

```bash
output:
  model_figure: true
```

The same figure can be drawn straight from an `af.graphical.EPOptimiser` that has already run, which is the quickest way to look
at a fit that is sitting in memory:

```bash
af.EPPlotter(optimiser.factor_graph, ep_history=optimiser.ep_history).figure(kind="state")
```

Pass the optimiser's own `factor_graph`, never a factor graph model's `graph` property: that property builds a new
graph, renaming every prior factor as it goes, on every access -- and the history is keyed by the graph the optimiser
actually swept. `kind="model"` draws the structure alone and needs no history at all, while `kind="state"` without one
raises rather than quietly drawing a diagnostic figure with no diagnostics in it. The default is to show the figure;
`format="png"` with a `path` writes it instead.

## Hierarchical Models

A specific type of graphical model is a hierarchical model, where the shared parameter(s) of a graph are assumed
to be drawn from a common parent distribution. Fitting these datasets simultanoeusly enables better estimate
of this global distribution.

Hierarchical models can also be scaled up to large datasets via Expectation Propagation.
