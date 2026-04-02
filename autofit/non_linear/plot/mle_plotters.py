from autofit.non_linear.plot.plot_util import skip_in_test_mode, output_figure


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
            plt.semilogy(iteration_list, parameters, c="k")
        else:
            plt.plot(iteration_list, parameters, c="k")

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

    log_likelihood_list = samples.log_likelihood_list
    iteration_list = range(len(log_likelihood_list))

    if use_last_50_percent:
        iteration_list = iteration_list[int(len(iteration_list) / 2) :]
        log_likelihood_list = log_likelihood_list[int(len(log_likelihood_list) / 2) :]

    plt.figure(figsize=(12, 12))

    if use_log_y:
        plt.semilogy(iteration_list, log_likelihood_list, c="k")
    else:
        plt.plot(iteration_list, log_likelihood_list, c="k")

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
