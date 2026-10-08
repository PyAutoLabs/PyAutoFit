# Configs

**PyAutoFit** uses a number of configuration files that customize the default behaviour of visualization, output,
priors, the grid search and other aspects of **PyAutoFit**.

The default settings of each non-linear search are not set by configuration files: they are the default values of
the arguments of each search class (e.g. `af.Nautilus`, `af.Emcee`), and are changed by passing those arguments when
the search is created.

Descriptions of every configuration file and their input parameters are provided in the `README.md` in
the [config directory of the workspace](https://github.com/PyAutoLabs/autofit_workspace/tree/main/config)

## Setup

By default, **PyAutoFit** looks for the config files in a `config` folder in the current working directory, which is
why we run autofit scripts from the `autofit_workspace` directory.

The configuration path can also be set manually in a script using the project **PyAutoNerves** and the following
command (the path to the `output` folder where the results of a non-linear search are stored is also set below):

```bash
from autonerves import conf

conf.instance.push(
    new_path="path/to/config",
    output_path=f"path/to/output"
)
```
