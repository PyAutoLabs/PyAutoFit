.. _statistical_methods:

Statistical Methods
===================

**PyAutoFit** supports numerous statistical methods that allow for advanced Bayesian inference to be performed.

Graphical Models
----------------

For inference problems consisting of many datasets, the model composition is often very complex. Model parameters
can depend on multiple datasets, and the datasets themselves may be interdependent.

Graphical models concisely describe these model and dataset dependencies, allowing them to be fitted simultaneously.
This is achieved through a concise API and scientific workflow that ensures scalability to large datasets.

A full description of using graphical models is given below:

https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/features/graphical_models.ipynb

Hierarchical Models
-------------------

Hierarchical models are where multiple parameters in the model are assumed to be drawn from a common distribution.
The parameters of this parent distribution are themselves inferred from the data, enabling
more robust and informative model fitting.

A full description of using hierarchical models is given below:

https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models/tutorial_4_hierachical_models.ipynb

Model Comparison
----------------

Common questions when fitting a model to data are: what model should I use? How many parameters should the model have?
Is the model too complex or too simple?

Model comparison answers to these questions. It amounts to composing and fitting many different models to the data
and comparing how well they fit the data.

A full description of using model comparison is given below:

https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/features/model_comparison.ipynb

Interpolation
-------------

It is common to fit a model to many similar datasets, where it is anticipated that one or more model parameters vary
smoothly across the datasets.

For example, the datasets may be taken at different times, where the signal in the data and therefore model parameters
vary smoothly as a function of time.

It may be desirable to fit the datasets one-by-one and then interpolate the results in order
to determine the most likely model parameters at any point in time.

**PyAutoFit**'s interpolation feature allows for this, and a full description of its use is given below:

https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/features/interpolate.ipynb

Search Grid Search
------------------

A classic method to perform model-fitting is a grid search, where the parameters of a model are divided onto a grid of
values and the likelihood of each set of parameters on this grid is sampled. For low dimensionality problems this
simple approach can be sufficient to locate high likelihood solutions, however it scales poorly to higher dimensional
problems.

**PyAutoFit** can perform a search grid search, which allows one to perform a grid-search over a subset of parameters
within a model, but use a non-linear search to fit for the other parameters. The parameters over which the grid-search
is performed are also included in the model fit and their values are simply confined to the boundaries of their grid
cell.

This can help ensure robust results are inferred for complex models, and can remove multi modality in a parameter
space to further aid the fitting process.

A full description of using search grid searches is given below:

https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/features/search_grid_search.ipynb

Search Chaining
---------------

To perform a model-fit, we typically compose one model and fit it to our data using one non-linear search.

Search chaining fits many different models to a dataset using a chained sequence of non-linear searches. Initial
fits are performed using simplified models and faster non-linear fitting techniques. The results of these simplified
fits are then be used to initialize fits using a higher dimensionality model with a more detailed non-linear search.

To fit highly complex models search chaining allows us to therefore to granularize the fitting procedure into a series
of **bite-sized** searches which are faster and more reliable than fitting the more complex model straight away.

A full description of using search chaining is given below:

https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/features/search_grid_search.ipynb

Sensitivity Mapping
-------------------

Bayesian model comparison allows us to take a dataset, fit it with multiple models and use the Bayesian evidence to
quantify which model objectively gives the best-fit following the principles of Occam's Razor.

However, a complex model may not be favoured by model comparison not because it is the 'wrong' model, but simply
because the dataset being fitted is not of a sufficient quality for the more complex model to be favoured. Sensitivity
mapping addresses what quality of data would be needed for the more complex model to be favoured.

In order to do this, sensitivity mapping involves us writing a function that uses the model(s) to simulate a dataset.
We then use this function to simulate many datasets, for different models, and fit each dataset to quantify
how much the change in the model led to a measurable change in the data. This is called computing the sensitivity.

A full description of using sensitivity mapping is given below:

https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/features/sensitivity_mapping.ipynb


