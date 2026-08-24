========
Plotting
========

Create figures and subplots of the non-linear search specific visualization of every search algorithm supported
by **PyAutoFit**.

The plotting API is **functional**: import ``autofit.plot`` and call a plot function directly, passing it the
``Samples`` object of a completed fit. There are no ``Plotter`` classes.

.. code-block:: python

   import autofit.plot as aplt

   aplt.corner_cornerpy(samples=result.samples)

Every plot function forwards its ``**kwargs`` to the library it wraps (``corner.py``, ``anesthetic``,
``matplotlib``), so the wrapped library's own options are available directly:

.. code-block:: python

   aplt.corner_cornerpy(samples=result.samples, bins=5, show_titles=True)

Values **PyAutoFit** computes — the parameter labels, the sample weights, and the plot range that keeps
``corner.py`` from raising "no dynamic range" on a converged parameter — act as defaults a keyword argument
overrides. A name the wrapped library does not accept raises a ``TypeError`` naming it, rather than being
silently ignored.

**Examples / Tutorials:**

- `readthedocs: non-linear search example <https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html>`_
- `autofit_workspace: plot tutorials <https://github.com/PyAutoLabs/autofit_workspace/tree/main/notebooks/guides/plot>`_
- `HowToFit: tutorial lectures (detailed step-by-step examples) <https://github.com/PyAutoLabs/HowToFit>`_

Search Plot Functions [aplt]
----------------------------

.. currentmodule:: autofit.plot

**Posterior Corner Plots:**

.. autosummary::
   :toctree: _autosummary

   corner_cornerpy
   corner_anesthetic

**Sampling Trace Plots:**

.. autosummary::
   :toctree: _autosummary

   subplot_parameters
   log_likelihood_vs_iteration

Figure Output [aplt]
--------------------

Every plot function above takes ``path``, ``filename`` and ``format`` arguments controlling where the figure goes:
``format="show"`` (the default) displays it interactively, whereas ``"png"`` or ``"pdf"`` writes it to
``path/filename.format``. The shared helper that performs this is also public:

.. autosummary::
   :toctree: _autosummary

   output_figure
