import numpy as np

from autoconf import conf

from autofit.non_linear.plot.plot_util import skip_in_test_mode, output_figure


@skip_in_test_mode
def corner_cornerpy(samples, path=None, filename="corner", format="show", **kwargs):
    import matplotlib.pylab as pylab

    config_dict = conf.instance["visualize"]["plots_settings"]["corner_cornerpy"]

    params = {"font.size": int(config_dict["fontsize"])}
    pylab.rcParams.update(params)

    import corner

    corner.corner(
        data=np.asarray(samples.parameter_lists),
        weight_list=samples.weight_list,
        labels=samples.model.parameter_labels_with_superscripts_latex,
    )

    output_figure(path=path, filename=filename, format=format)
