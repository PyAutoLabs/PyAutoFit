import datetime as dt
from typing import List

from autonerves import conf
from autofit.mapper.prior_model.representative import find_groups
from autofit.text import formatter as frm, samples_text
from autofit.tools.util import info_whitespace


def padding(item, target=6):
    string = str(item)
    difference = target - len(string)
    prefix = difference * " "
    return f"{prefix}{string}"


def result_max_lh_info_from(max_log_likelihood_sample : List[float], max_log_likelihood : float, model) -> List[str]:
    """
    Output the maximum log likelihood model only, for quick reference.
    """
    results = []

    results += [
        frm.add_whitespace(
            str0="Maximum Log Likelihood ",
            str1="{:.8f}".format(max_log_likelihood),
            whitespace=info_whitespace(),
        )
    ]

    results += ["\n\n", model.parameterization]

    results += ["\n\nMaximum Log Likelihood Model:\n\n"]

    formatter = frm.TextFormatter(line_length=info_whitespace())

    paths = []

    for (_, prior), value in zip(
            model.unique_path_prior_tuples,
            max_log_likelihood_sample,
    ):
        for path in model.all_paths_for_prior(prior):
            paths.append((path, value))

    for path, value in find_groups(paths):
        formatter.add(path, format_str().format(value))
    results += [formatter.text + "\n"]

    return results


def result_info_from(samples) -> str:
    """
    Output the full model.results file, which include the most-likely model, most-probable model at 1 and 3
    sigma confidence and information on the maximum log likelihood.
    """
    from autofit.non_linear.test_mode import skip_fit_output

    if skip_fit_output():
        return "[fit output skipped — PYAUTO_SKIP_FIT_OUTPUT=1]"

    results = []

    if hasattr(samples, "log_evidence"):
        if samples.log_evidence is not None:
            results += [
                frm.add_whitespace(
                    str0="Bayesian Evidence ",
                    str1="{:.8f}".format(samples.log_evidence),
                    whitespace=info_whitespace(),
                )
            ]
            results += ["\n"]

    max_log_likelihood_sample = samples.max_log_likelihood(as_instance=False)

    results += result_max_lh_info_from(
        max_log_likelihood_sample=max_log_likelihood_sample,
        max_log_likelihood=(max(samples.log_likelihood_list)),
        model=samples.model,
    )

    if hasattr(samples, "pdf_converged"):
        if samples.pdf_converged:
            results += samples_text.summary(
                samples=samples, sigma=3.0, indent=4, line_length=info_whitespace()
            )
            results += ["\n"]
            results += samples_text.summary(
                samples=samples, sigma=1.0, indent=4, line_length=info_whitespace()
            )

        else:
            results += [
                "\n WARNING: The samples have not converged enough to compute a PDF and model errors. \n "
                "The model below over estimates errors. \n\n"
            ]
            results += samples_text.summary(
                samples=samples, sigma=1.0, indent=4, line_length=info_whitespace()
            )

        results += ["\n\ninstances\n"]

    formatter = frm.TextFormatter(line_length=info_whitespace())

    for path, value in find_groups(samples.model.path_float_tuples):
        formatter.add(path, value)

    results += ["\n" + formatter.text]

    return "".join(results)


def search_summary_from_samples(samples) -> [str]:
    line = [f"Total Samples = {samples.total_samples}\n"]
    if hasattr(samples, "total_accepted_samples"):
        line.append(f"Total Accepted Samples = {samples.total_accepted_samples}\n")
        line.append(f"Acceptance Ratio = {samples.acceptance_ratio}\n")

    # Per-step non-finite accounting from the gradient searches
    # (``MultiStartGradient``). Guarded on the key rather than the search type,
    # following the duck-typed MCMC block above: this function is on every
    # search's summary path (``paths/abstract.py`` -> ``search_summary_to_file``),
    # so a search whose ``samples_info`` lacks these keys — Nautilus, which is
    # jit-only and has no gradient to be non-finite — must emit nothing at all.
    #
    # Rates, not just counts: raw totals are not comparable across runs, since
    # they scale with the lane-step budget. 797 on a 16x3000 run and 10 on an
    # 8x300 run differ 80x raw but only ~2x per lane-step.
    #
    # Named as the neutral facts they are. These are counts of undefined and
    # non-differentiable likelihood evaluations; they are deliberately NOT
    # presented as a smoothness or sampler-difficulty metric, because the
    # correlation with (for example) HMC divergence rates is unvalidated.
    samples_info = getattr(samples, "samples_info", None) or {}

    if "n_value_nan_lane_steps" in samples_info:
        n_value_nan = int(samples_info.get("n_value_nan_lane_steps", 0))
        n_grad_nan = int(samples_info.get("n_grad_nan_lane_steps", 0))

        line.append(f"Resurrections = {int(samples_info.get('n_resurrections', 0))}\n")
        line.append(f"Value-NaN Lane-Steps = {n_value_nan}\n")
        line.append(f"Gradient-NaN Lane-Steps = {n_grad_nan}\n")

        # ``n_starts * total_steps`` is the number of lane-steps actually taken.
        # A search that died before its first step has a zero denominator, so
        # the rates are omitted rather than reported as a division error.
        lane_steps = int(samples_info.get("n_starts", 0)) * int(
            samples_info.get("total_steps", 0)
        )
        if lane_steps > 0:
            line.append(f"Value-NaN Lane-Step Rate = {n_value_nan / lane_steps}\n")
            line.append(f"Gradient-NaN Lane-Step Rate = {n_grad_nan / lane_steps}\n")

    if samples.time is not None:
        line.append(f"Time To Run = {dt.timedelta(seconds=float(samples.time))}\n")
        line.append(
            f"Time Per Sample (seconds) = {float(samples.time) / samples.total_samples}\n"
        )
    return line


def search_summary_to_file(
        samples,
        log_likelihood_function_time,
        filename,
        visualization_time=None,
):
    summary = search_summary_from_samples(samples=samples)
    summary.append(
        f"Log Likelihood Function Evaluation Time (seconds) = {log_likelihood_function_time}\n"
    )

    expected_time = dt.timedelta(
        seconds=float(samples.total_samples * log_likelihood_function_time)
    )
    summary.append(f"Expected Time To Run (seconds) = {expected_time}\n")

    try:
        speed_up_factor = float(expected_time.total_seconds()) / float(samples.time)
        summary.append(
            f"Speed Up Factor (e.g. due to parallelization) = {speed_up_factor}\n"
        )
    except TypeError:
        pass

    if visualization_time is not None:
        summary.append(
            f"Visualization Time (seconds) = {visualization_time}"
        )

    frm.output_list_of_strings_to_file(file=filename, list_of_strings=summary)


def format_str() -> str:
    """The format string for the model.results file, describing to how many decimal points every parameter
    estimate is output in the model.results file.
    """
    decimal_places = conf.instance["general"]["output"]["model_results_decimal_places"]
    return f"{{:.{decimal_places}f}}"
