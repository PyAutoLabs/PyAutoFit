(conda)=

# Installation with conda

Installation via a conda environment circumvents compatibility issues when installing certain libraries. This guide
assumes you have a working installation of [conda](https://conda.io/miniconda.html).

First, create a conda environment (we name is `autofit` to signify it is for the **PyAutoFit** install).

The command below creates this environment with some of the bigger package requirements, the rest will be installed
with **PyAutoFit** via pip:

```bash
conda create -n autofit numpy scipy
```

Activate the conda environment (you will have to do this every time you want to run **PyAutoFit**):

```bash
conda activate autofit
```

The latest version of **PyAutoFit** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

```bash
pip install autofit
```

Next, clone the `autofit_workspace` (the line `--depth 1` clones only the most recent branch on
the `autofit_workspace`, reducing the download size):

```bash
cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
git clone https://github.com/PyAutoLabs/autofit_workspace --depth 1
cd autofit_workspace
```

Run the `welcome.py` script to get started!

```bash
python3 welcome.py
```
