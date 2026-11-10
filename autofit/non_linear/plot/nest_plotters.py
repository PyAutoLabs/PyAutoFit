import numpy as np
import warnings
from contextlib import contextmanager

from autonerves import conf

from autofit.non_linear.plot.plot_util import (
    skip_in_test_mode,
    log_plot_exception,
    output_figure,
    accepted_kwarg_names,
    checked_kwargs,
    PlotKwargsError,
)

# Above this many plotted parameters a corner plot is drawn in a compact style
# (2D histograms instead of KDE contours, no upper triangle) unless the caller
# or the config says otherwise. A KDE contour panel costs about a second and
# there are n(n-1)/2 of them, so 20 parameters would spend minutes on contours.
DEFAULT_MAX_DETAILED_PARAMETERS = 10


def _setting(config_dict, name, default):
    """A ``corner_anesthetic`` setting, or ``default`` for an older config without it."""
    try:
        return config_dict[name]
    except KeyError:
        return default


@contextmanager
def _deferred_shared_axes():
    """
    Run pandas' shared-axis label pass once at the end instead of once per panel.

    Every panel anesthetic draws goes through a pandas plot call, and each one
    finishes with ``handle_shared_axes``: a pass over *every* axis of the figure
    that hides the tick and axis labels of inner panels. A corner plot of n
    parameters makes about n^2/2 such calls over about n^2 axes, so the label
    pass costs n^4 -- the bulk of a 20-parameter plot's 40+ minutes. The pass is
    idempotent, so deferring it and replaying each distinct call once after the
    last panel gives the same figure for a fraction of the cost: pixel-identical
    for the styles ``corner_anesthetic`` draws (checked against per-panel
    passes for the default, compact, subset and latent plots). A caller's own
    ``kinds`` can still differ slightly in axis padding (``hist_1d`` diagonals
    on two parameters), never in where the posterior is drawn.

    ``handle_shared_axes`` is private to pandas: if it is missing, or is ever
    called in a form this does not recognise, the plot just runs unpatched.
    """
    try:
        from pandas.plotting._matplotlib import core as pandas_core

        original = pandas_core.handle_shared_axes
    except (ImportError, AttributeError):
        yield
        return

    pending = {}

    def defer(*args, **kwargs):
        try:
            key = tuple(
                (name, value) for name, value in kwargs.items() if name != "axarr"
            )
            hash(key)
        except TypeError:
            key = None

        if args or key is None:
            return original(*args, **kwargs)

        # Keep the latest call per argument set: its ``axarr`` covers every
        # axis drawn so far.
        pending.pop(key, None)
        pending[key] = kwargs

    pandas_core.handle_shared_axes = defer
    try:
        yield
    finally:
        pandas_core.handle_shared_axes = original

    for kwargs in pending.values():
        original(**kwargs)

        # A diagonal panel draws its 1D curve on a twin of the panel's axes,
        # which must span the same x range as the panel and its column. Run per
        # panel, the label pass settled each twin's limits as it went; run once
        # at the end, the last twin settles on its own curve's range and the
        # curve is drawn shifted against the tick labels. Re-impose the
        # parent's range, leaving every autoscale setting as it was.
        for ax in kwargs.get("axarr", ()):
            twin = getattr(ax, "twin", None)
            if twin is not None and twin is not ax:
                twin.set_xlim(ax.get_xlim(), auto=None)


@skip_in_test_mode
@log_plot_exception
def corner_anesthetic(
    samples,
    path=None,
    filename="corner_anesthetic",
    format="show",
    parameter_indices=None,
    latent=False,
    **kwargs,
):
    """
    Corner plot of the posterior via anesthetic, over every parameter of the model
    or, with ``parameter_indices``, over a subset of them (the parameters of one
    model component, say) so that a many-parameter fit stays readable.

    With more plotted parameters than the ``max_detailed_parameters`` setting
    (``visualize/plots_settings``, default 10) the plot is drawn compactly: 2D
    histograms instead of KDE contours in the lower triangle and no upper
    triangle, as the full style would take minutes. Pass ``kinds`` or
    ``upper`` to choose yourself.

    Parameters
    ----------
    samples
        The samples of the non-linear search, whose ``model`` labels the axes.
    parameter_indices
        The indices (in model order, or in the latent set's order when
        ``latent``) of the parameters to plot; all by default.
    latent
        Plot the model's latent parameters instead of its free ones: each
        component's declared conversion is applied to the samples (a shear's
        components become its magnitude and angle, say) and parameters without
        one pass through; see ``AbstractPriorModel.latent_parameters_from``.
    compact_backend
        Optional keyword: ``"anesthetic"`` (default) or experimental ``"array"``.
        The array backend reuses weighted histogram ranges on finite float64
        compact plots. Unsupported styles, data and dependency versions use
        the established anesthetic backend; no samples are thinned.
    """
    config_dict = conf.instance["visualize"]["plots_settings"]["corner_anesthetic"]
    compact_backend = kwargs.pop(
        "compact_backend", _setting(config_dict, "compact_backend", "anesthetic")
    )
    if compact_backend not in ("anesthetic", "array"):
        raise PlotKwargsError("compact_backend must be 'anesthetic' or 'array'")

    from anesthetic.samples import NestedSamples
    from anesthetic import make_2d_axes
    import matplotlib.pylab as pylab

    params = {"font.size": int(config_dict["fontsize"])}
    pylab.rcParams.update(params)

    model = samples.model

    parameter_array = np.asarray(samples.parameter_lists)

    if latent:
        latent_parameters = model.latent_parameters_from(
            parameter_array, weights=samples.weight_list, recentre=True
        )
        labels = latent_parameters.parameter_labels_with_superscripts_latex
        parameter_array = latent_parameters.parameter_array
    else:
        labels = model.parameter_labels_with_superscripts_latex

    if parameter_indices is not None:
        parameter_indices = list(parameter_indices)
        labels = [labels[i] for i in parameter_indices]
        parameter_array = parameter_array[:, parameter_indices]

    figsize = (
        len(labels) * config_dict["figsize_per_parammeter"],
        len(labels) * config_dict["figsize_per_parammeter"],
    )

    # anesthetic keys its columns by name, but LaTeX labels repeat whenever a model
    # has several components of one type -- every Gaussian of a multi-Gaussian basis
    # is labelled $\epsilon_{1}^{g}$. A repeated key makes plot_2d raise KeyError,
    # which @log_plot_exception reports as an unconverged posterior, so the plot
    # silently never appears. Key the columns by position, which is always unique,
    # and pass the LaTeX labels for display only.
    #
    # The labels are not attached to the samples themselves: anesthetic then keys
    # its columns by a (name, label) MultiIndex and copies it for every panel,
    # which is half the cost of a 20-parameter plot. The axes are labelled from
    # ``labels_map`` instead, both when made and again after ``plot_2d``, which
    # labels the edge panels by column name.
    columns = [f"p{i}" for i in range(len(labels))]
    labels_map = dict(zip(columns, labels))

    nested_samples = NestedSamples(
        parameter_array,
        weights=samples.weight_list,
        columns=columns,
    )

    try:
        from pandas.errors import SettingWithCopyWarning
    except ImportError:  # pandas >= 2.2 removed SettingWithCopyWarning
        SettingWithCopyWarning = None

    if SettingWithCopyWarning is not None:
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    # ``kwargs`` splits by destination: what shapes the figure goes to
    # ``make_2d_axes``, everything else styles the plot via ``plot_2d``, which
    # forwards what it does not name straight to matplotlib — matplotlib raises
    # on an unknown property, so unlike ``corner`` there is no silent sink here
    # to guard against and the only check needed is that the caller is not
    # overriding the sample array.
    #
    # ``figsize`` / ``facecolor`` / ``dpi`` are named explicitly because
    # ``make_2d_axes`` takes them through its own ``**fig_kw`` rather than
    # declaring them — including the two this function computes below, which a
    # caller must be able to override.
    axes_names = (accepted_kwarg_names(make_2d_axes) - {"params"}) | {
        "figsize",
        "facecolor",
        "dpi",
    }
    kwargs = checked_kwargs(
        kwargs,
        reserved=("data", "weights", "columns", "labels"),
        target="anesthetic",
    )

    axes_settings = dict(figsize=figsize, facecolor=config_dict["facecolor"])
    axes_settings.update(
        {key: value for key, value in kwargs.items() if key in axes_names}
    )

    plot_settings = dict(alpha=config_dict["alpha"], label="posterior")

    # Above ``max_detailed_parameters`` a corner plot is drawn in its compact
    # style: 2D histograms in the lower triangle instead of KDE contours (about a
    # second per panel) and no upper triangle, whose scatter panels repeat the
    # lower triangle's pairs and double the axes to build, plot and save. A
    # caller's ``kind`` / ``kinds`` / ``upper`` always wins.
    max_detailed_parameters = _setting(
        config_dict, "max_detailed_parameters", DEFAULT_MAX_DETAILED_PARAMETERS
    )
    if len(labels) > max_detailed_parameters and not (
        {"kind", "kinds", "upper"} & set(kwargs)
    ):
        plot_settings["kinds"] = {"diagonal": "kde_1d", "lower": "hist_2d"}
        axes_settings["upper"] = False

    plot_settings.update(
        {key: value for key, value in kwargs.items() if key not in axes_names}
    )

    prepared = None
    if compact_backend == "array":
        from autofit.non_linear.plot import compact_corner

        prepared = compact_corner.prepare(
            parameter_array, samples.weight_list, axes_settings, plot_settings
        )

    fig, axes = make_2d_axes(
        columns,
        labels=labels_map,
        **axes_settings,
    )

    if SettingWithCopyWarning is not None:
        warnings.filterwarnings("default", category=SettingWithCopyWarning)

    if prepared is not None:
        compact_corner.draw(axes, prepared)
    else:
        with _deferred_shared_axes():
            nested_samples.plot_2d(axes, **plot_settings)
    axes.set_labels(labels_map)
    axes.iloc[-1, 0].legend(
        bbox_to_anchor=(len(axes) / 2, len(axes)),
        loc="lower center",
        ncols=2,
    )

    output_figure(path=path, filename=filename, format=format)
