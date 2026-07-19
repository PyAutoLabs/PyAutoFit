import logging

import numpy as np

from autonerves import conf

from autofit.non_linear.plot.plot_util import skip_in_test_mode, output_figure

logger = logging.getLogger(__name__)


def _corner_range_from(data):
    """Per-column ``(min, max)`` plot ranges for a corner figure.

    ``corner`` raises "no dynamic range" if any column has zero spread (every
    sample equal). That is a legitimate input — a reduced-iteration run (e.g.
    ``PYAUTO_TEST_MODE=1``) or a parameter that has converged flat — so we hand
    it an explicit per-column range rather than let it crash. Columns with real
    spread keep their ``(min, max)``; degenerate columns are widened to a small
    non-zero window centred on the value so the plot still renders.
    """
    mins = np.asarray(data).min(axis=0)
    maxs = np.asarray(data).max(axis=0)
    plot_range = []
    for lower, upper in zip(mins, maxs):
        if lower == upper:
            pad = max(abs(lower) * 1e-4, 1e-8)
            lower, upper = lower - pad, upper + pad
        plot_range.append((lower, upper))
    return plot_range


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
        range=_corner_range_from(data),
    )

    output_figure(path=path, filename=filename, format=format)
