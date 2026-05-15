=======
Samples
=======

Every sample of a model-fit and non-liner search are stored in a ``Samples`` object, which can be manipulated to
inspect the results in detail (e.g. perform parameter estimation with errors).

For example, for an MCMC model-fit, the ``Samples`` objects contains every sample of walker.

**Examples / Tutorials:**

- `readthedocs: example on using results <https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html>`_.
- `autofit_workspace: simple results tutorial <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/cookbooks/result.ipynb>`_
- `autofit_workspace: complex result tutorial <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/cookbooks/result.ipynb>`_
- `HowToFit: tutorial lectures (detailed step-by-step examples) <https://github.com/PyAutoLabs/HowToFit>`_

Samples
-------

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Samples
   SamplesPDF
   SamplesMCMC
   SamplesNest
   SamplesStored