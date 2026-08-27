(pip)=

# Installation with pip

:::{note}
**PyAutoFit** requires **Python 3.12 or later**. On Python 3.9, 3.10 or 3.11,
`pip install autofit` stops with an error telling you to upgrade — it will not
quietly install an older release instead. Upgrade Python to 3.12+ before
installing.
:::

We strongly recommend that you install **PyAutoFit** in a
[Python virtual environment](https://www.geeksforgeeks.org/python-virtual-environment/), with the link attached
describing what a virtual environment is and how to create one.

The latest version of **PyAutoFit** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

```bash
pip install autofit
```

This installs \[**JAX**\](<https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>) (and `optax`) by
default, which **PyAutoFit** uses for gradient-based searches and GPU acceleration (the older
`pip install autofit[jax]` command still works and installs the same thing). The default install is CPU-only
JAX; for GPU support, follow the official
\[JAX installation guide\](<https://jax.readthedocs.io/en/latest/installation.html>) **before** installing.
On Intel (x86_64) macOS, where JAX publishes no wheels, the install automatically excludes JAX and runs on
the slower NumPy path — a warning is printed at import to make this clear.

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

We dropped support for Python 3.9, 3.10 and 3.11 in release `2026.7.29.2`
(July 2026) — the first release published declaring `Requires-Python >=3.12`.

Raising that floor does not retract what is already published. Releases at or
below `2026.7.29.1` were published declaring `>=3.9`, and PyPI metadata is
immutable, so they remain valid candidates forever. Left alone, `pip install
autofit` on an older Python did not fail — it walked back to `2026.7.29.1` and
installed a months-old stack without JAX, reporting nothing.

Release `2026.7.29.1.post1` exists to stop that. It contains no code, declares
`Requires-Python <3.12`, and raises an error when pip tries to build it, so an
unsupported Python gets an explanation instead of a stale install.

If you need a historical release, pin it exactly — that still resolves on older
Pythons:

```bash
pip install autofit==2025.10.6.1
```

One gap remains: `pip install --only-binary=:all: autofit` skips source
distributions entirely, so it steps past `2026.7.29.1.post1` and installs the
old wheel silently. If you use that flag, pin the version you want.
