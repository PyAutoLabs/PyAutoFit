(pip)=

# Installation with pip

:::{note}
**PyAutoFit** requires **Python 3.12 or later**. If you are on Python
3.9, 3.10, or 3.11, `pip install autofit` will fail with a "no matching
distribution" error. Upgrade Python to 3.12+ before installing.
:::

We strongly recommend that you install **PyAutoFit** in a
[Python virtual environment](https://www.geeksforgeeks.org/python-virtual-environment/), with the link attached
describing what a virtual environment is and how to create one.

The latest version of **PyAutoFit** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

```bash
pip install autofit
```

If this raises no errors **PyAutoFit** is installed! If there is an error check out
the [troubleshooting section](https://pyautofit.readthedocs.io/en/latest/installation/troubleshooting.html).

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

## Legacy Python versions

We dropped support for Python 3.9, 3.10, and 3.11 in release `2026.4.5.3`
(April 2026). Pre-`2026.4.5.3` releases on PyPI have been yanked, so they
will not install via the standard `pip install autofit` command.

If you have an existing project that requires a pre-`2026.4.5.3` version,
you can still install it explicitly by pinning the version, e.g.:

```bash
pip install autofit==2025.10.6.1
```

Yanked releases remain available for explicit pins; only resolver-driven
fallback is blocked.
