"""
`NonLinearSearch.start_points(model, fitness, n)`: the one place the searches draw their
initial points, which makes the start-point plot exactly once (search-extensibility A2).
"""
import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _search_and_fitness(monkeypatch, use_jax=False, number_of_cores=1):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    search = af.Emcee(number_of_cores=number_of_cores)
    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(data=np.ones(10), noise_map=np.ones(10), use_jax=use_jax)
    return search, model, search.make_fitness(analysis=analysis, model=model)


def _count_plots(monkeypatch, search):
    plotted = []
    monkeypatch.setattr(
        search,
        "plot_start_point",
        lambda parameter_vector, model, analysis: plotted.append(parameter_vector),
    )
    return plotted


def test_plots_the_first_point_once(monkeypatch):
    search, model, fitness = _search_and_fitness(monkeypatch)
    plotted = _count_plots(monkeypatch, search)

    unit, parameters, figures_of_merit = search.start_points(
        model=model, fitness=fitness, n=4
    )

    assert len(parameters) == len(unit) == len(figures_of_merit) == 4
    assert plotted == [parameters[0]]


def test_plot_false_skips_the_plot(monkeypatch):
    search, model, fitness = _search_and_fitness(monkeypatch)
    plotted = _count_plots(monkeypatch, search)

    search.start_points(model=model, fitness=fitness, n=2, plot=False)

    assert plotted == []


def test_initializer_gets_the_effective_core_count(monkeypatch):
    pytest.importorskip("jax")
    from autofit.non_linear.parallel import PoolFactory

    search, model, fitness = _search_and_fitness(
        monkeypatch, use_jax=True, number_of_cores=2
    )
    _count_plots(monkeypatch, search)

    seen = {}
    original = search.initializer.samples_from_model

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(search.initializer, "samples_from_model", spy)

    search._fit_pools = PoolFactory(search, fitness.analysis)
    search.start_points(model=model, fitness=fitness, n=2)

    assert seen["n_cores"] == 1
