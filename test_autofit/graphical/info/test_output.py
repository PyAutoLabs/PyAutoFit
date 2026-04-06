import pytest

import autofit as af

from autoconf.conf import with_config
from autofit import graphical as g, DirectoryPaths

MAX_STEPS = 2


class MockResult(af.m.MockResult):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def projected_model(self):
        return self.model


class MockSearch(af.m.MockSearch):
    def fit(
            self,
            model,
            analysis,
            info=None,
    ):
        super().fit(model, analysis)
        return MockResult(model)


def _run_optimisation(factor_graph_model, paths=None):
    search = MockSearch()
    factor_graph_model.optimise(
        search,
        max_steps=MAX_STEPS,
        log_interval=1,
        visualise_interval=1,
        output_interval=1,
        paths=paths or DirectoryPaths(
            name="name",
            is_identifier_in_paths=False,
        ),
    )


@with_config("general", "output", "remove_files", value=False)
def test_output(output_directory, factor_graph_model):
    factor_graph_model.model_factors[0]._name = "factor_1"
    factor_graph_model.model_factors[1]._name = "factor_2"
    _run_optimisation(factor_graph_model)

    path = output_directory / "name/factor_1"

    assert path.exists()
    assert (output_directory / "name/factor_2").exists()

    for number in range(MAX_STEPS):
        assert (path / f"optimization_{number}").exists()


@with_config("general", "output", "remove_files", value=False)
def test_default_output(output_directory, factor_graph_model):
    _run_optimisation(factor_graph_model)
    assert (output_directory / "name/AnalysisFactor0").exists()
    assert (output_directory / "name/AnalysisFactor1").exists()


@with_config("general", "output", "remove_files", value=False)
def test_path_prefix(output_directory, factor_graph_model):
    paths = DirectoryPaths(
        path_prefix="path_prefix",
        name="name",
        is_identifier_in_paths=False,
    )
    optimiser = g.EPOptimiser(
        factor_graph=factor_graph_model.graph,
        paths=paths,
        default_optimiser=MockSearch(),
    )

    assert optimiser.output_path == output_directory / "path_prefix/name"

    _run_optimisation(factor_graph_model, paths=paths)
    assert (output_directory / "path_prefix/name/AnalysisFactor0").exists()
    assert (output_directory / "path_prefix/name/AnalysisFactor1").exists()
