"""
Read-only results-inspector tools over PyAutoFit output directories.

Each function is a thin wrapper over an existing public PyAutoFit aggregator
API (`autofit.aggregator.Aggregator`, `af.SearchOutput`): argument parsing, one call, and
JSON-friendly serialization — nothing more. Any behaviour beyond that belongs
in PyAutoFit itself, not here ("glue, not code").

This module deliberately has no MCP dependency: `server.py` registers these
functions as MCP tools, and the test suite exercises them without the `mcp`
package installed.
"""

import contextlib
import json
import sys
from pathlib import Path

from autonerves.dictable import to_dict

import autofit as af

# The directory-backed aggregator — af.Aggregator is the database-backed one,
# and the alias also keeps the API audit from resolving it there.
from autofit.aggregator import Aggregator as DirectoryAggregator


@contextlib.contextmanager
def _stdout_to_stderr():
    """
    An MCP stdio server must keep stdout clean — it carries the JSON-RPC
    channel — but the directory aggregator prints progress to stdout,
    so every autofit call runs with stdout redirected to stderr.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _float_or_none(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _search_row(search) -> dict:
    summary = search.samples_summary
    max_lh_sample = getattr(summary, "max_log_likelihood_sample", None)
    return dict(
        name=search.name,
        unique_tag=search.unique_tag,
        directory=str(search.directory),
        is_complete=search.is_complete,
        log_evidence=_float_or_none(getattr(summary, "log_evidence", None)),
        max_log_likelihood=_float_or_none(
            getattr(max_lh_sample, "log_likelihood", None)
        ),
        model_free_parameters=getattr(search.model, "prior_count", None),
    )


def list_searches(
    directory: str,
    sort_by: str = "log_evidence",
    limit: int = 20,
    completed_only: bool = False,
) -> list:
    with _stdout_to_stderr():
        aggregator = DirectoryAggregator.from_directory(
            directory, completed_only=completed_only
        )
        rows = [_search_row(search) for search in aggregator]
    rows.sort(
        key=lambda row: (row.get(sort_by) is not None, row.get(sort_by)),
        reverse=True,
    )
    return rows[:limit] if limit else rows


def get_model(directory: str) -> dict:
    with _stdout_to_stderr():
        model = af.SearchOutput(Path(directory)).model
    return dict(info=model.info, model=to_dict(model))


def get_result_summary(directory: str) -> str:
    with _stdout_to_stderr():
        return af.SearchOutput(Path(directory)).model_results


def get_samples_summary(directory: str) -> dict:
    with _stdout_to_stderr():
        search_output = af.SearchOutput(Path(directory))
        summary = search_output.samples_summary
        if summary is None:
            raise FileNotFoundError(
                f"No samples summary found under {directory}/files/ — "
                "is this a completed search output directory?"
            )
        model = search_output.model
        return dict(
            log_evidence=_float_or_none(summary.log_evidence),
            max_log_likelihood=_float_or_none(
                summary.max_log_likelihood_sample.log_likelihood
            ),
            parameter_paths=[".".join(path) for path in model.paths],
            max_log_likelihood_parameters=[
                float(value)
                for value in summary.max_log_likelihood_sample.parameter_lists_for_model(
                    model
                )
            ],
            # MLE searches (e.g. LBFGS) have no PDF, so no median-PDF sample.
            median_pdf_parameters=None
            if summary.median_pdf_sample is None
            else [
                float(value)
                for value in summary.median_pdf_sample.parameter_lists_for_model(
                    model
                )
            ],
        )


def get_search_info(directory: str) -> dict:
    search_json = Path(directory) / "files" / "search.json"
    with _stdout_to_stderr():
        search_output = af.SearchOutput(Path(directory))
        return dict(
            name=search_output.name,
            unique_tag=search_output.unique_tag,
            is_complete=search_output.is_complete,
            search=json.loads(search_json.read_text())
            if search_json.exists()
            else None,
        )


def list_images(directory: str) -> list:
    # Visualization outputs live under <directory>/image/ (SearchOutput.image
    # reads there, despite its docstring saying `files/`).
    return sorted(path.name for path in (Path(directory) / "image").glob("*.png"))


def fetch_image(directory: str, name: str = "subplot_fit"):
    with _stdout_to_stderr():
        return af.SearchOutput(Path(directory)).image(name)
