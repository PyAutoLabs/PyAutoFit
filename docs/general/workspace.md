(workspace)=

# Workspace Tour

You should have downloaded and configured the [autofit_workspace](https://github.com/PyAutoLabs/autofit_workspace)
when you installed **PyAutoFit**. If you didn't, checkout the
[installation instructions](https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip)
for how to downloaded and configure the workspace.

The `README.md` files distributed throughout the workspace describe every folder and file, and specify if
examples are for beginner or advanced users.

New users should begin by checking out the following parts of the workspace.

## HowToFit

The **HowToFit** lecture series is a collection of Jupyter notebooks describing how to build a **PyAutoFit** model
fitting project and giving illustrations of different statistical methods and techniques.

HowToFit now lives in its own standalone repository at [PyAutoLabs/HowToFit](https://github.com/PyAutoLabs/HowToFit).
Clone or browse the repo for a full description of the lectures and the notebooks for every chapter.

## Scripts / Notebooks

There are numerous example describing how perform model-fitting with **PyAutoFit** and providing an overview of its
advanced model-fitting features. All examples are provided as Python scripts and Jupyter notebooks.

Descriptions of every configuration file and their input parameters are provided in the `README.md` in
the [config directory of the workspace](https://github.com/PyAutoLabs/autofit_workspace/tree/main/config)

## Config

Here, you'll find the configuration files used by **PyAutoFit** which customize:

> - The default settings used by every non-linear search.
> - Example priors and notation configs which associate model-component with model-fitting.
> - The `general.ini` config which customizes other aspects of **PyAutoFit**.

Checkout the [configuration](https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip)
section of the `readthedocs` for a complete description of every configuration file.

## Dataset

This folder stores the example dataset's used in examples in the workspace.

## Output

The folder where the model-fitting results of a non-linear search are stored.
