.. _conda:

Installation with conda
=======================

Installation via a conda environment circumvents compatibility issues when installing certain libraries. This guide
assumes you have a working installation of `conda <https://conda.io/miniconda.html>`_.

First, create a conda environment (we name is ``autofit`` to signify it is for the **PyAutoFit** install).

The command below creates this environment with some of the bigger package requirements, the rest will be installed
with **PyAutoFit** via pip:

.. code-block:: bash

    conda create -n autofit numpy scipy

Activate the conda environment (you will have to do this every time you want to run **PyAutoFit**):

.. code-block:: bash

    conda activate autofit

The latest version of **PyAutoFit** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

.. code-block:: bash

    pip install autofit

Next, clone the ``autofit_workspace`` at the tag matching your installed ``PyAutoFit`` version. Each
``PyAutoFit`` release tags a paired ``autofit_workspace`` snapshot, so cloning by tag guarantees that the
example scripts and notebooks were generated against the library version you installed:

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   AUTOFIT_VERSION=$(python -c "import autofit; print(autofit.__version__)")
   git clone https://github.com/Jammy2211/autofit_workspace --branch $AUTOFIT_VERSION --depth 1
   cd autofit_workspace

Run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py