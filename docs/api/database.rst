========
Database
========

PyAutoFit's database feature outputs all model-fitting results as a sqlite3 (https://docs.python.org/3/library/sqlite3.html)
relational database, such that all results can be efficiently loaded into a Jupyter notebook or Python script for
inspection, analysis and interpretation.

**Examples / Tutorials:**

- `readthedocs: example using database functionality <https://pyautofit.readthedocs.io/en/latest/features/database.html>`_
- `autofit_workspace: tutorial using database <https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/database.ipynb>`_
- `HowToFit: tutorial lectures (detailed step-by-step examples) <https://github.com/PyAutoLabs/HowToFit>`_

----------
Aggregator
----------

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Aggregator