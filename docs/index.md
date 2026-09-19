# PyAutoFit

**PyAutoFit** is a Python package for scientific model fitting and Bayesian inference. Bring your own model
classes, data and likelihood code; **PyAutoFit** provides priors, inference algorithms and tools for interpreting
and organising the results. It is domain agnostic: inference can sit around your existing scientific software.

We recommend getting started with [autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant), which lets
you **perform scientific inference using natural language**. Describe the model you want to fit, ask the assistant
to run the analysis, and explore the results through follow-up requests —
see [Natural Language Inference](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html).

Users can then set up a **PyAutoFit** scientific workflow, which enables streamlined modeling of small
datasets with tools to scale up to large datasets.

**PyAutoFit** supports advanced statistical methods, most
notably [a big data framework for Bayesian hierarchical analysis](https://pyautofit.readthedocs.io/en/latest/features/graphical.html).

## Getting Started

The following links are useful for new starters:

- [The autofit_assistant repository](https://github.com/PyAutoLabs/autofit_assistant), which lets you perform inference in natural language via a coding agent — the recommended starting point.
- [The PyAutoFit readthedocs](https://pyautofit.readthedocs.io/en/latest), which includes an [installation guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html) and an overview of **PyAutoFit**'s core features.
- [The introduction Jupyter Notebook on Colab](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.19.1/notebooks/overview/overview_1_the_basics.ipynb), where you can try **PyAutoFit** in a web browser (without installation).
- [The autofit_workspace GitHub repository](https://github.com/PyAutoLabs/autofit_workspace), which includes example scripts demonstrating **PyAutoFit**'s features.
- [The standalone HowToFit repository](https://github.com/PyAutoLabs/HowToFit), a series of Jupyter notebook lectures which give new users a step-by-step introduction to **PyAutoFit**.

## Community & Support

Questions, help with your code or your analysis, and ideas: the
[PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions).
Bug reports with a reproducer (a snippet, the traceback, your versions):
an issue on the library's tracker. The Slack is for collaborators, by
invitation.

Collaborators receive the latest **PyAutoFit** updates in the [Slack channel](https://pyautoFit.slack.com/).
Contact [James Nightingale](https://github.com/Jammy2211) about collaborator access.

## HowToFit

For users less familiar with Bayesian inference and scientific analysis you may wish to read through
the **HowToFits** lectures. These teach you the basic principles of Bayesian inference, with the
content pitched at undergraduate level and above.

The lectures are available in the [standalone HowToFit repository](https://github.com/PyAutoLabs/HowToFit).

## Overview

To illustrate **PyAutoFit** we use a toy model of fitting a one-dimensional Gaussian to noisy 1D data. Here's
the `data` (black) and the model (red) we'll fit:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/files/toy_model_fit.png
:width: 400
```

Start with [Quick Start](https://pyautofit.readthedocs.io/en/latest/overview/quick_start.html), which sets up
the assistant and gets inference running on your own likelihood in a few minutes. After that, there are two
ways to read the rest of these docs:

- [Natural Language Inference](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html)
  walks through this fit as a conversation with an assistant — composing the model, defining the likelihood,
  choosing a search, running the fit and inspecting the results, all described in words rather than written by
  hand. **This is the recommended route through the docs.**
- [The Python API](https://pyautofit.readthedocs.io/en/latest/overview/python_api.html) walks through the same
  fit in code, showing the `Model`, `Analysis`, search and `Result` objects the assistant writes for you, and
  which you can read, run and extend yourself.

Between them sit the [Scientific Workflow](https://pyautofit.readthedocs.io/en/latest/overview/scientific_workflow.html),
which makes inference scalable to large datasets, and the
[Statistical Methods](https://pyautofit.readthedocs.io/en/latest/overview/statistical_methods.html) overview,
covering hierarchical models, search chaining and Bayesian model comparison.

```{toctree}
:caption: 'Overview:'
:hidden: true
:maxdepth: 1

overview/quick_start
overview/natural_language
overview/scientific_workflow
overview/statistical_methods
overview/python_api
```

```{toctree}
:caption: 'Cookbooks:'
:hidden: true
:maxdepth: 1

cookbooks/model
cookbooks/analysis
cookbooks/search
cookbooks/result
cookbooks/samples
cookbooks/configs
cookbooks/multiple_datasets
cookbooks/multi_level_model
```

```{toctree}
:caption: 'Features:'
:hidden: true
:maxdepth: 1

features/graphical
features/interpolate
features/search_chaining
features/search_grid_search
features/sensitivity_mapping
```

```{toctree}
:caption: 'Installation:'
:hidden: true
:maxdepth: 1

installation/overview
installation/conda
installation/pip
installation/source
installation/troubleshooting
```

```{toctree}
:caption: 'General:'
:hidden: true
:maxdepth: 1

general/workspace
general/configs
general/roadmap
general/software
general/citations
general/credits
```

```{toctree}
:caption: 'Science Examples:'
:hidden: true
:maxdepth: 1

science_examples/astronomy
```

```{toctree}
:caption: 'API Reference:'
:hidden: true
:maxdepth: 1

api/model
api/priors
api/analysis
api/searches
api/plot
api/samples
api/database
api/source
```
