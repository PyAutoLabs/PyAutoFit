===================
Non-Linear Searches
===================

A non-linear search is an algorithm which fits a model to data.

**PyAutoFit** currently supports three types of non-linear search algorithms: nested samplers (nest),
Markov Chain Monte Carlo (MCMC) and Maximum Likelihood Estimators (MLE).

**Examples / Tutorials:**

- `readthedocs: example using non-linear searches <https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html>`_.
- `autofit_workspace: simple tutorial <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/overview/overview_1_the_basics.ipynb>`_
- `autofit_workspace: complex tutorial <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/overview/overview_2_scientific_workflow.ipynb>`_
- `HowToFit: tutorial lectures (detailed step-by-step examples) <https://github.com/PyAutoLabs/HowToFit>`_

Nested Samplers
---------------

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   DynestyDynamic
   DynestyStatic

MCMC
----

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Emcee
   Zeus

Maximum Likelihood Estimators
-----------------------------

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   BFGS
   LBFGS

There are also a number of tools which are used to customize the behaviour of non-linear searches in **PyAutoFit**,
including directory output structure, parameter sample initialization and MCMC auto correlation analysis.

Tools
-----

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   DirectoryPaths
   DatabasePaths
   Result
   InitializerBall
   InitializerPrior
   AutoCorrelationsSettings

**PyAutoFit** can perform a parallelized grid-search of non-linear searches, where a subset of parameters in the
model are fitted in over a discrete grid.

**Examples / Tutorials:**

- `readthedocs: example using a non-linear search grid search <https://pyautofit.readthedocs.io/en/latest/features/search_grid_search.html>`_.
- `autofit_workspace: example using a non-linear search grid search <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/search_grid_search.ipynb>`_

GridSearch
----------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   SearchGridSearch
   GridSearchResult
