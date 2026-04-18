import logging

import numpy as np

from autoconf import conf

from autofit.non_linear.plot.plot_util import skip_in_test_mode, output_figure

logger = logging.getLogger(__name__)


@skip_in_test_mode
def corner_cornerpy(samples, path=None, filename="corner", format="show", **kwargs):
    data = np.asarray(samples.parameter_lists)
    if data.ndim < 2 or data.shape[0] <= data.shape[1]:
        logger.info(
            "corner_cornerpy: skipping corner plot, only %s sample(s) for %s parameter(s) "
            "(e.g. PYAUTO_TEST_MODE bypass or an early-iteration update).",
            data.shape[0] if data.ndim >= 1 else 0,
            data.shape[1] if data.ndim >= 2 else 0,
        )
        return

    import matplotlib.pylab as pylab

    config_dict = conf.instance["visualize"]["plots_settings"]["corner_cornerpy"]

    params = {"font.size": int(config_dict["fontsize"])}
    pylab.rcParams.update(params)

    import corner

    corner.corner(
        data=data,
        weight_list=samples.weight_list,
        labels=samples.model.parameter_labels_with_superscripts_latex,
    )

    output_figure(path=path, filename=filename, format=format)
