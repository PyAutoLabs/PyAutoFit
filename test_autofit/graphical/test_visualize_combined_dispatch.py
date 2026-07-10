import autofit as af


class _RecordingVisualizerA(af.Visualizer):
    calls = []

    @staticmethod
    def visualize_combined(analyses, paths, instance, during_analysis, quick_update=False):
        _RecordingVisualizerA.calls.append((list(analyses), list(instance)))


class _RecordingVisualizerB(af.Visualizer):
    calls = []

    @staticmethod
    def visualize_combined(analyses, paths, instance, during_analysis, quick_update=False):
        _RecordingVisualizerB.calls.append((list(analyses), list(instance)))


class _AnalysisA(af.mock.MockAnalysis):
    Visualizer = _RecordingVisualizerA


class _AnalysisB(af.mock.MockAnalysis):
    Visualizer = _RecordingVisualizerB


def _factor_graph(analyses):
    model = af.Model(af.ex.Gaussian)
    factors = [af.AnalysisFactor(prior_model=model, analysis=analysis) for analysis in analyses]
    return af.FactorGraphModel(*factors)


def test__homogeneous_graph__single_combined_call_with_all_factors():
    _RecordingVisualizerA.calls = []

    analyses = [_AnalysisA(), _AnalysisA(), _AnalysisA()]
    graph = _factor_graph(analyses)

    graph.visualize_combined(instance=["i0", "i1", "i2"], paths=None, during_analysis=True)

    assert len(_RecordingVisualizerA.calls) == 1
    called_analyses, called_instances = _RecordingVisualizerA.calls[0]
    assert called_analyses == analyses
    assert called_instances == ["i0", "i1", "i2"]


def test__mixed_graph__one_combined_call_per_visualizer_type():
    _RecordingVisualizerA.calls = []
    _RecordingVisualizerB.calls = []

    analysis_a0, analysis_b, analysis_a1 = _AnalysisA(), _AnalysisB(), _AnalysisA()
    graph = _factor_graph([analysis_a0, analysis_b, analysis_a1])

    graph.visualize_combined(instance=["a0", "b0", "a1"], paths=None, during_analysis=True)

    # A-group: both A analyses, order preserved, with their matching instances.
    assert len(_RecordingVisualizerA.calls) == 1
    called_analyses, called_instances = _RecordingVisualizerA.calls[0]
    assert called_analyses == [analysis_a0, analysis_a1]
    assert called_instances == ["a0", "a1"]

    # B-group: the lone B analysis with its own instance.
    assert len(_RecordingVisualizerB.calls) == 1
    called_analyses, called_instances = _RecordingVisualizerB.calls[0]
    assert called_analyses == [analysis_b]
    assert called_instances == ["b0"]
