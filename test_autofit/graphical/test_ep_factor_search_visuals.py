"""
EP factor searches skip their per-search visuals by default (PyAutoFit #1642).

Expectation propagation re-enters the same factor search once per factor per EP
step. At the small per-factor fits EP is built on, a search's own visuals
(`analysis.visualize`, the corner / fit plots of `plot_results`) dominated the
wall time, and each step overwrote the previous step's images anyway.

`general.yaml -> output -> visualize_ep_factor_searches` (default False) now
governs them: off, only the first search of each factor draws its before-fit
visuals and no factor search draws per-search visuals; on, the old behaviour.
Samples, summaries and the EP optimiser's own output (`graph.info`,
`graph.png`, `ep_history.csv`) are written either way.
"""

import numpy as np
import pytest

from autonerves import conf

import autofit as af

pytestmark = [
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]


# Module level (not local classes / closures) so the analysis pickles with the
# dynesty checkpoint.
CALLS = []
PLOT_CALLS = []


class RecordingVisualizer(af.Visualizer):
    @staticmethod
    def visualize_before_fit(analysis, paths, model):
        CALLS.append(("visualize_before_fit", paths.analysis_name))

    @staticmethod
    def visualize(analysis, paths, instance, during_analysis):
        CALLS.append(("visualize", paths.analysis_name))


class RecordingAnalysis(af.ex.Analysis):
    Visualizer = RecordingVisualizer


def _run_ep(tmp_path, monkeypatch):
    """
    Two AnalysisFactors sharing a centre, each fitted by a named DynestyStatic,
    two EP steps.

    Returns the recorded Visualizer calls, the recorded `plot_results` calls
    and the EP output path.
    """
    CALLS.clear()
    PLOT_CALLS.clear()

    def recording_plot_results(self, samples):
        PLOT_CALLS.append(self.paths.analysis_name)

    monkeypatch.setattr(af.DynestyStatic, "plot_results", recording_plot_results)

    x = np.arange(50)
    centre = af.GaussianPrior(mean=25.0, sigma=10.0)

    # One search object shared by both factors, as a user would build it; the
    # prior factors fall back to the Laplace default optimiser.
    search = af.DynestyStatic(
        name="ep_visuals", nlive=20, maxcall=200, number_of_cores=1
    )

    factors = []
    for i in range(2):
        data = af.ex.Gaussian(centre=25.0, normalization=10.0, sigma=5.0).model_data_from(
            xvalues=x
        )
        model = af.Model(
            af.ex.Gaussian,
            centre=centre,
            normalization=af.GaussianPrior(mean=10.0, sigma=5.0),
            sigma=af.GaussianPrior(mean=5.0, sigma=2.0),
        )
        factors.append(
            af.AnalysisFactor(
                model,
                analysis=RecordingAnalysis(data=data, noise_map=np.ones(50)),
                name=f"factor_{i}",
                optimiser=search,
            )
        )

    factor_graph = af.FactorGraphModel(*factors)

    paths = af.DirectoryPaths(name="ep_factor_search_visuals", path_prefix=str(tmp_path))

    factor_graph.optimise(
        af.LaplaceOptimiser(),
        paths=paths,
        ep_history=af.EPHistory(kl_tol=None),
        max_steps=2,
    )

    return list(CALLS), list(PLOT_CALLS), paths.output_path


def test__default__factor_searches_skip_per_search_visuals(tmp_path, monkeypatch):
    calls, plot_calls, output_path = _run_ep(tmp_path, monkeypatch)

    assert [c for c in calls if c[0] == "visualize"] == []
    assert plot_calls == []

    # Before-fit visuals once per factor: only its first search draws them.
    before = sorted(c[1] for c in calls if c[0] == "visualize_before_fit")
    assert before == ["factor_0/optimization_0", "factor_1/optimization_0"]

    # The EP optimiser's own output is untouched.
    assert (output_path / "graph.info").exists()
    assert (output_path / "graph.png").exists()
    assert (output_path / "ep_history.csv").exists()


def test__key_on__factor_searches_draw_per_search_visuals(tmp_path, monkeypatch):
    output = conf.instance["general"]["output"]
    original = output.get("visualize_ep_factor_searches", False)
    output["visualize_ep_factor_searches"] = True
    try:
        calls, plot_calls, _ = _run_ep(tmp_path, monkeypatch)
    finally:
        output["visualize_ep_factor_searches"] = original

    visualized = {c[1] for c in calls if c[0] == "visualize"}
    assert visualized == {
        "factor_0/optimization_0",
        "factor_0/optimization_1",
        "factor_1/optimization_0",
        "factor_1/optimization_1",
    }
    assert len(plot_calls) == 4

    before = sorted(c[1] for c in calls if c[0] == "visualize_before_fit")
    assert before == [
        "factor_0/optimization_0",
        "factor_0/optimization_1",
        "factor_1/optimization_0",
        "factor_1/optimization_1",
    ]


def test__visualize_switches_reset_after_optimise(tmp_path, monkeypatch):
    """`optimise` restores the switches, so the search behaves normally afterwards."""
    _run_ep(tmp_path, monkeypatch)

    search = af.DynestyStatic(name="after", number_of_cores=1)
    assert search._visualize_fit is True
    assert search._visualize_before_fit is True
