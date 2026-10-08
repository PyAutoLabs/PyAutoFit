"""
`NonLinearSearch.make_fitness`, the one `Fitness` construction site the searches share
(search-extensibility phase A2).

It replaces `test_quick_update_wiring.py`, which AST-scanned every `Fitness(...)` call
under the search tree for a forwarded `iterations_per_quick_update`: a search that
forgot it silently never fired quick updates (PyAutoFit#1434). With every search
building its `Fitness` through `make_fitness`, which always forwards the cadence, that
scan has nothing left to guard; what remains is that `make_fitness` forwards it, and that
no search builds a `Fitness` any other way (NSS, the last exception, moved onto it in A3b).
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search import capabilities as cap

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

SEARCH_ROOT = Path(__file__).parents[3] / "autofit" / "non_linear" / "search"


def _analysis():
    return af.ex.Analysis(data=np.ones(10), noise_map=np.ones(10), use_jax=False)


def _fitness_for(search_cls, **overrides):
    search = search_cls(iterations_per_quick_update=7)
    return search, search.make_fitness(
        analysis=_analysis(), model=af.Model(af.ex.Gaussian), **overrides
    )


@pytest.mark.parametrize(
    "search_cls, fom_is_log_likelihood, convert_to_chi_squared, resample",
    [
        (af.Emcee, False, False, -np.inf),
        (af.DynestyStatic, True, False, -1.0e99),
        (af.Nautilus, True, False, -1.0e99),
        (af.Drawer, False, False, -np.inf),
        (af.BFGS, False, True, -np.inf),
        (af.LBFGS, False, True, -np.inf),
    ],
)
def test_convention_follows_the_declared_target(
    search_cls, fom_is_log_likelihood, convert_to_chi_squared, resample
):
    search, fitness = _fitness_for(search_cls)

    assert fitness.fom_is_log_likelihood is fom_is_log_likelihood
    assert fitness.convert_to_chi_squared is convert_to_chi_squared
    assert fitness.resample_figure_of_merit == resample


def test_neg2_resample_reaches_the_backend_as_the_declared_invalid_value():
    search, fitness = _fitness_for(af.BFGS)

    assert search.objective_target.quantity == cap.ObjectiveQuantity.NEG2_LOG_POSTERIOR
    assert fitness.resample_figure_of_merit * -2.0 == type(search).invalid_value


def test_quick_update_settings_are_always_forwarded():
    search, fitness = _fitness_for(af.Emcee)

    assert fitness.iterations_per_quick_update == search.iterations_per_quick_update == 7
    assert fitness.live_visual_update == search.live_visual_update
    assert fitness.paths is search.paths


def test_overrides_win():
    _, fitness = _fitness_for(af.Nautilus, batched=True, store_history=True)

    assert fitness.batched is True
    assert fitness.store_history is True


def _fitness_constructions():
    for path in sorted(SEARCH_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "Fitness":
                yield path.relative_to(SEARCH_ROOT).as_posix()


def test_only_make_fitness_constructs_a_fitness():
    """
    NSS, the last search that built its own ``Fitness``, moved onto ``run(ctx)`` in
    search-extensibility A3b; ``make_fitness`` is now the only construction site.
    """
    assert sorted(_fitness_constructions()) == ["abstract_search.py"]
