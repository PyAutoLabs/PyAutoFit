from autofit.non_linear.plot.plot_util import (
    skip_in_test_mode,
    output_figure,
    checked_kwargs,
)


def _line_kwargs(kwargs, defaults):
    """
    Merge caller ``kwargs`` into the trace-line defaults these plots draw with.

    The traces are plain matplotlib lines, so a caller's ``**kwargs`` are
    ``Line2D`` properties. Validating against the artist's own setters (aliases
    included) means a mistyped property is named here rather than surfacing as a
    ``Line2D.set()`` error from deep inside matplotlib.
    """
    from matplotlib.artist import ArtistInspector
    from matplotlib.lines import Line2D

    inspector = ArtistInspector(Line2D)
    accepts = set(inspector.get_setters())
    accepts |= {alias for aliases in inspector.aliasd.values() for alias in aliases}

    settings = dict(defaults)
    settings.update(checked_kwargs(kwargs, accepts=accepts, target="matplotlib"))
    return settings


@skip_in_test_mode
def subplot_parameters(
    samples,
    use_log_y=False,
    use_last_50_percent=False,
    path=None,
    filename="subplot_parameters",
    format="show",
    **kwargs,
):
    import matplotlib.pyplot as plt

    line_kwargs = _line_kwargs(kwargs, {"c": "k"})

    model = samples.model
    parameter_lists = samples.parameters_extract

    plt.subplots(model.total_free_parameters, 1, figsize=(12, 3 * len(parameter_lists)))

    for i, parameters in enumerate(parameter_lists):
        iteration_list = range(len(parameter_lists[0]))

        plt.subplot(model.total_free_parameters, 1, i + 1)

        if use_last_50_percent:
            iteration_list = iteration_list[int(len(iteration_list) / 2) :]
            parameters = parameters[int(len(parameters) / 2) :]

        if use_log_y:
            plt.semilogy(iteration_list, parameters, **line_kwargs)
        else:
            plt.plot(iteration_list, parameters, **line_kwargs)

        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel(model.parameter_labels_with_superscripts_latex[i], fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    actual_filename = filename
    if use_log_y:
        actual_filename += "_log_y"
    if use_last_50_percent:
        actual_filename += "_last_50_percent"

    output_figure(path=path, filename=actual_filename, format=format)


@skip_in_test_mode
def log_likelihood_vs_iteration(
    samples,
    use_log_y=False,
    use_last_50_percent=False,
    path=None,
    filename="log_likelihood_vs_iteration",
    format="show",
    **kwargs,
):
    import matplotlib.pyplot as plt

    line_kwargs = _line_kwargs(kwargs, {"c": "k"})

    log_likelihood_list = samples.log_likelihood_list
    iteration_list = range(len(log_likelihood_list))

    if use_last_50_percent:
        iteration_list = iteration_list[int(len(iteration_list) / 2) :]
        log_likelihood_list = log_likelihood_list[int(len(log_likelihood_list) / 2) :]

    plt.figure(figsize=(12, 12))

    if use_log_y:
        plt.semilogy(iteration_list, log_likelihood_list, **line_kwargs)
    else:
        plt.plot(iteration_list, log_likelihood_list, **line_kwargs)

    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Log Likelihood", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    title = "Log Likelihood vs Iteration"
    if use_log_y:
        title += " (Log Scale)"
    if use_last_50_percent:
        title += " (Last 50 Percent)"

    plt.title(title, fontsize=24)

    actual_filename = filename
    if use_log_y:
        actual_filename += "_log_y"
    if use_last_50_percent:
        actual_filename += "_last_50_percent"

    output_figure(path=path, filename=actual_filename, format=format)


@skip_in_test_mode
def figure_of_merit_vs_iteration(
    samples,
    path=None,
    filename="figure_of_merit_vs_iteration",
    format="show",
    **kwargs,
):
    """
    Plot the global-best figure-of-merit trace of an auto-convergence gradient
    search versus step, so the plateau the search stopped on can be inspected.

    The trace is read from ``samples.samples_info["fom_history"]`` (the per-step
    global best figure-of-merit, ``-2 * log_posterior``, that the multi-start
    gradient searches record). Searches that do not record it (e.g. ``LBFGS`` /
    ``Drawer``) leave it absent and this plot is skipped.
    """
    fom_history = samples.samples_info.get("fom_history")

    if not fom_history:
        return

    import matplotlib.pyplot as plt

    line_kwargs = _line_kwargs(kwargs, {"c": "k"})

    iteration_list = range(len(fom_history))

    plt.figure(figsize=(12, 12))
    plt.plot(iteration_list, fom_history, **line_kwargs)

    plt.xlabel("Step", fontsize=16)
    plt.ylabel("Global-Best Figure of Merit (-2 ln posterior)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    stop_reason = samples.samples_info.get("stop_reason")
    title = "Global-Best Figure of Merit vs Step (auto-convergence trace)"
    if stop_reason is not None:
        title += f"\nstopped: {stop_reason}"

    plt.title(title, fontsize=20)

    output_figure(path=path, filename=filename, format=format)
