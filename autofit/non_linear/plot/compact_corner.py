"""Opt-in array renderer for a narrowly supported anesthetic compact corner.

Keep the public anesthetic axes and 1D KDE. Only the repeated dataframe dispatch
and histogram range calculation are replaced. Unsupported inputs are rejected
by ``prepare`` before any artists are drawn, so callers can use the normal path.
No cache or patched library function survives a figure.
"""

# Histogram compatibility semantics adapted from anesthetic.plot.hist_plot_2d.
# MIT License
#
# Copyright (c) 2019 Will Handley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from importlib.metadata import version

import numpy as np


KINDS = {"diagonal": "kde_1d", "lower": "hist_2d"}
# The adapter reproduces this backend's post-processing. Other versions retain
# the established renderer until they have their own real-runtime parity tests.
VERSIONS = {"anesthetic": "2.16.0", "pandas": "3.0.6", "matplotlib": "3.11.2"}
COMMON = {"alpha", "label", "color", "q", "density"}
LOWER = COMMON | {"bins", "range", "levels", "cmin", "cmax", "vmin", "vmax", "snap"}
DIAGONAL = COMMON | {"bw_method", "bw_scale", "ngrid_kde", "order", "clip_to_zero"}


def prepare(data, weights, axes_settings, plot_settings):
    """Return figure-local arrays/options, or None to select the full backend."""
    if any(version(package) != expected for package, expected in VERSIONS.items()):
        return None
    if axes_settings.get("upper") is not False:
        return None
    if set(axes_settings) - {"upper", "figsize", "facecolor", "dpi"}:
        return None
    if plot_settings.get("kinds") != KINDS:
        return None
    if set(plot_settings) - (COMMON | {"kinds", "lower_kwargs", "diagonal_kwargs"}):
        return None
    lower = plot_settings.get("lower_kwargs", {})
    diagonal = plot_settings.get("diagonal_kwargs", {})
    if not isinstance(lower, dict) or not isinstance(diagonal, dict):
        return None
    if set(lower) - LOWER or set(diagonal) - DIAGONAL:
        return None
    common = {k: v for k, v in plot_settings.items() if k in COMMON}
    # Samples.plot_2d gives common kwargs precedence over panel kwargs.
    lower, diagonal = {**lower, **common}, {**diagonal, **common}
    from matplotlib.colors import is_color_like

    if any("color" in opts and not is_color_like(opts["color"]) for opts in (lower, diagonal)):
        return None
    data, weights = np.asarray(data), np.asarray(weights)
    if (
        data.ndim != 2 or data.dtype != np.float64 or data.shape[0] < 2
        or data.shape[1] < 2 or weights.shape != (data.shape[0],)
        or weights.dtype != np.float64 or not np.isfinite(data).all()
        or not np.isfinite(weights).all() or np.any(weights < 0)
        or not np.isfinite(weights.sum()) or np.count_nonzero(weights) < 2
    ):
        return None
    # Anesthetic jitters constant KDE columns using global random state. Keep
    # that behavior on the original path rather than inventing a new policy.
    if np.any(np.ptp(data[weights > 0], axis=0) == 0):
        return None
    from anesthetic.utils import quantile
    from anesthetic.plot import quantile_plot_interval

    q = quantile_plot_interval(lower.get("q", 5))
    bounds = [quantile(column, [q[0], q[-1]], weights) for column in data.T]
    return data, weights, bounds, lower, diagonal


def histogram(ax, x, y, weights, bounds, options):
    """Weighted linear histogram with precomputed anesthetic quantile bounds."""
    from anesthetic.plot import basic_cmap
    from anesthetic.utils import iso_probability_contours
    from matplotlib.patches import Rectangle

    opts = options.copy()
    opts.pop("q", None)
    density = opts.pop("density", False)
    bins = opts.pop("bins", 10)
    limits = opts.pop("range", bounds)
    levels = opts.pop("levels", None)
    cmin, cmax = opts.pop("cmin", None), opts.pop("cmax", None)
    label = opts.pop("label", None)
    # Match the native color-cycle advancement, even with an explicit color.
    color = opts.pop("color", ax._get_lines.get_next_color())
    cmap = basic_cmap(color)
    pdf, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=limits, density=density, weights=weights
    )
    if levels is not None:
        levels = iso_probability_contours(pdf, levels) + [pdf.max()]
        pdf = np.asarray(levels)[np.digitize(pdf, levels, right=True)]
        pdf = np.ma.masked_array(pdf, pdf < levels[1])
        if cmin is not None:
            pdf[pdf < cmin] = np.ma.masked
        if cmax is not None:
            pdf[pdf > cmax] = np.ma.masked
    image = ax.pcolormesh(
        xedges, yedges, pdf.T, cmap=cmap, vmin=opts.pop("vmin", 0),
        snap=opts.pop("snap", True), **opts
    )
    ax.add_patch(Rectangle((0, 0), 0, 0, fc=cmap(.999), ec=cmap(.32), lw=2, label=label))
    return image


def draw(axes, prepared):
    """Draw the prepared compact figure and apply the shared-axis pass once."""
    from anesthetic.plot import kde_plot_1d
    from matplotlib import rcParams
    from pandas.plotting._matplotlib.core import handle_shared_axes

    data, weights, bounds, lower, diagonal = prepared
    columns = list(axes.columns)
    for y, row in axes.iterrows():
        j = columns.index(y)
        for i, (x, ax) in enumerate(row.items()):
            if ax is None:
                continue
            if x == y:
                kde_plot_1d(ax.twin, data[:, i], weights=weights, **diagonal)
                ax.twin.grid(rcParams["axes.grid"])
                ax.twin.set_yticks([])
                ax.twin.set_ylim(bottom=0)
            else:
                histogram(ax, data[:, i], data[:, j], weights, (bounds[i], bounds[j]), lower)
                ax.grid(rcParams["axes.grid"])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
    axes._set_logticks()
    fig = axes.iloc[-1, 0].figure
    all_axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
    nrows = len({ax.get_position().y0 for ax in all_axes})
    ncols = len({ax.get_position().x0 for ax in all_axes})
    handle_shared_axes(axarr=all_axes, nplots=len(all_axes), naxes=nrows*ncols,
                       nrows=nrows, ncols=ncols, sharex=False, sharey=False)
    for ax in all_axes:
        twin = getattr(ax, "twin", None)
        if twin is not None and twin is not ax:
            twin.set_xlim(ax.get_xlim(), auto=None)
