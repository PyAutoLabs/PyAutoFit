import numpy as np
import warnings

from autoconf import conf

from autofit.non_linear.plot.plot_util import (
    skip_in_test_mode,
    log_plot_exception,
    output_figure,
)


@skip_in_test_mode
@log_plot_exception
def corner_anesthetic(samples, path=None, filename="corner_anesthetic", format="show", **kwargs):
    config_dict = conf.instance["visualize"]["plots_settings"]["corner_anesthetic"]

    from anesthetic.samples import NestedSamples
    from anesthetic import make_2d_axes
    import matplotlib.pylab as pylab

    params = {"font.size": int(config_dict["fontsize"])}
    pylab.rcParams.update(params)

    model = samples.model

    figsize = (
        model.total_free_parameters * config_dict["figsize_per_parammeter"],
        model.total_free_parameters * config_dict["figsize_per_parammeter"],
    )

    nested_samples = NestedSamples(
        np.asarray(samples.parameter_lists),
        weights=samples.weight_list,
        columns=model.parameter_labels_with_superscripts_latex,
    )

    from pandas.errors import SettingWithCopyWarning

    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    fig, axes = make_2d_axes(
        model.parameter_labels_with_superscripts_latex,
        figsize=figsize,
        facecolor=config_dict["facecolor"],
    )

    warnings.filterwarnings("default", category=SettingWithCopyWarning)

    nested_samples.plot_2d(
        axes,
        alpha=config_dict["alpha"],
        label="posterior",
    )
    axes.iloc[-1, 0].legend(
        bbox_to_anchor=(len(axes) / 2, len(axes)),
        loc="lower center",
        ncols=2,
    )

    output_figure(path=path, filename=filename, format=format)
