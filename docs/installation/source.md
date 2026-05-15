(source)=

# Building From Source

Building from source means that you clone (or fork) the **PyAutoFit** GitHub repository and run **PyAutoFit** from
there. Unlike `conda` and `pip` this provides a build of the source code that you can edit and change, to
contribute the development **PyAutoFit** or experiment with yourself!

First, clone (or fork) the **PyAutoFit** GitHub repository:

```bash
git clone https://github.com/PyAutoLabs/PyAutoFit
```

Next, install the **PyAutoFit** dependencies via pip:

```bash
pip install -r PyAutoFit/requirements.txt
```

If you are using a `conda` environment, add the source repository as follows:

```bash
conda-develop PyAutoFit
```

Alternatively, if you are using a Python environment include the **PyAutoFit** source repository in your PYTHONPATH
(noting that you must replace the text `/path/to` with the path to the **PyAutoFit** directory on your computer):

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoFit
```

For unit tests to pass you will also need the following optional requirements:

```bash
pip install -r PyAutoFit/optional_requirements.txt
```

Finally, check the **PyAutoFit** unit tests run and pass (you may need to install pytest via `pip install pytest`):

```bash
cd /path/to/PyAutoFit
python3 -m pytest
```
