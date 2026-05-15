======
Priors
======

The priors of parameters of every component of a mdoel, which is fitted to data, are customized using ``Prior`` objects.

**Examples / Tutorials:**

- `Model API Cookbooks (recommended) <https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>`_.

- `readthedocs: example using Model object <https://pyautofit.readthedocs.io/en/latest/overview/the_basics.html>`_.
- `readthedocs: example using Collection object <https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>`_.
- `autofit_workspace: simple tutorial <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/overview/overview_1_the_basics.ipynb>`_
- `autofit_workspace: complex tutorial <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/overview/overview_2_scientific_workflow.ipynb>`_
- `HowToFit: tutorial lectures (detailed step-by-step examples) <https://github.com/PyAutoLabs/HowToFit>`_

Priors
------

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   UniformPrior
   GaussianPrior
   LogUniformPrior
   LogGaussianPrior