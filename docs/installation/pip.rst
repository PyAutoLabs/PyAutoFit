.. _pip:

Installation with pip
=====================

We strongly recommend that you install **PyAutoFit** in a
`Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_, with the link attached
describing what a virtual environment is and how to create one.

The latest version of **PyAutoFit** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

.. code-block:: bash

    pip install autofit

If this raises no errors **PyAutoFit** is installed! If there is an error check out
the `troubleshooting section <https://pyautofit.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Next, clone the ``autofit_workspace`` at the tag matching your installed ``PyAutoFit`` version. Each
``PyAutoFit`` release tags a paired ``autofit_workspace`` snapshot, so cloning by tag guarantees that the
example scripts and notebooks were generated against the library version you installed:

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   AUTOFIT_VERSION=$(python -c "import autofit; print(autofit.__version__)")
   git clone https://github.com/Jammy2211/autofit_workspace --branch $AUTOFIT_VERSION --depth 1
   cd autofit_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py