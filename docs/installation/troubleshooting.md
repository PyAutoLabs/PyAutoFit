(troubleshooting)=

# Troubleshooting

## Current Working Directory

**PyAutoFit** scripts assume that the `autofit_workspace` directory is the Python working directory. This means
that, when you run an example script, you should run it from the `autofit_workspace` as follows:

```bash
cd path/to/autofit_workspace (if you are not already in the autofit_workspace).
python3 scripts/overview/simple/fit.py
```

The reasons for this are so that **PyAutoFit** can:

> - Load configuration settings from config files in the `autofit_workspace/config` folder.
> - Load example data from the `autofit_workspace/dataset` folder.
> - Output the results of models fits to your hard-disk to the `autofit/output` folder.
> - Import modules from the `autofit_workspace`, for example `from autofit_workspace.transdimensional import pipelines`.

If you have any errors relating to importing modules, loading data or outputting results it is likely because you
are not running the script with the `autofit_workspace` as the working directory!

## Support

If you are still having issues with installation or using **PyAutoFit** in general, please raise an issue on the
[autofit_workspace issues page](https://github.com/Jammy2211/autofit_workspace/issues) with a description of the
problem and your system setup (operating system, Python version, etc.).
