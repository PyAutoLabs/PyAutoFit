(source)=

# Building From Source

Building from source means that you clone (or fork) the **PyAutoFit** GitHub repository and run **PyAutoFit** from
there. Unlike `conda` and `pip` this provides a build of the source code that you can edit and change, to
contribute the development **PyAutoFit** or experiment with yourself!

First, clone (or fork) the **PyAutoFit** GitHub repository:

```bash
git clone https://github.com/PyAutoLabs/PyAutoFit
```

Next, install **PyAutoFit** and its dependencies in editable mode via pip:

```bash
pip install -e PyAutoFit
```

An editable install means changes you make to the source code are used straight away, so there is no need to add
the repository to your `PYTHONPATH`.

For unit tests to pass you will also need the optional requirements:

```bash
pip install -e "PyAutoFit[optional]"
```

Finally, check the **PyAutoFit** unit tests run and pass (you may need to install pytest via `pip install pytest`):

```bash
cd /path/to/PyAutoFit
python3 -m pytest
```
