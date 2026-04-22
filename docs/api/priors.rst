======
Priors
======

The priors of parameters of every component of a mdoel, which is fitted to data, are customized using ``Prior`` objects.

**Examples / Tutorials:**

- `Model API Cookbooks (recommended) <https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_1_basics.html>`_.

- `readthedocs: example using Model object <https://pyautofit.readthedocs.io/en/latest/overview/model_fit.html>`_.
- `readthedocs: example using Collection object <https://pyautofit.readthedocs.io/en/latest/overview/model_complex.html>`_.
- `autofit_workspace: simple tutorial <https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/overview/simple/fit.ipynb>`_
- `autofit_workspace: complex tutorial <https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/overview/complex/fit.ipynb>`_
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