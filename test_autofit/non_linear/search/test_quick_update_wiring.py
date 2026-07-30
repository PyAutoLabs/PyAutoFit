import ast
from pathlib import Path

import pytest

# Every search builds its own `Fitness`, and the on-the-fly quick-update machinery
# only runs if that construction forwards `iterations_per_quick_update`. Forgetting
# it fails *silently*: `Fitness.manage_quick_update` returns at its `is None` guard,
# so the search runs perfectly and simply never updates.
#
# That is not hypothetical. Before PyAutoFit#1434 only Nautilus forwarded it, so
# setting a real cadence did nothing under Dynesty, Emcee, Zeus, BlackJAX NUTS,
# BFGS or Drawer — and once the cadence is announced in the startup log, a missing
# forward turns a dead feature into a false claim in the CLI.
#
# Checked structurally rather than by running each search: the samplers are
# optional dependencies and their `_fit` bodies need real data, while the defect
# is entirely visible in the call site.

SEARCH_ROOT = Path(__file__).parents[3] / "autofit" / "non_linear" / "search"

REQUIRED_KWARG = "iterations_per_quick_update"

# path suffix -> why this construction site legitimately omits the kwarg.
EXEMPT = {
    "mle/multi_start_gradient/search.py": (
        "MultiStartGradient differentiates `fitness.call` inside its own "
        "jit/vmap step loop, so the Python-side counter in `call_wrap` would run "
        "once at trace time rather than per step. Its progress reporting is "
        "handled separately -- see PyAutoFit#1433."
    ),
}


def _fitness_call_sites():
    """Every `Fitness(...)` construction under the search tree, as
    (path-relative-to-search-root, set-of-keyword-names)."""
    for path in sorted(SEARCH_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)

            if name != "Fitness":
                continue

            keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
            yield path.relative_to(SEARCH_ROOT).as_posix(), keywords


def test__the_scan_finds_the_call_sites_it_is_meant_to_guard():
    # Without this, a refactor that renamed or re-exported `Fitness` would make
    # every assertion below vacuously pass.
    sites = list(_fitness_call_sites())

    assert len(sites) >= 8, sites
    assert any("nautilus" in path for path, _ in sites)
    assert any("dynesty" in path for path, _ in sites)


@pytest.mark.parametrize(
    "relative_path, keywords",
    list(_fitness_call_sites()),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test__every_search_forwards_the_quick_update_cadence(relative_path, keywords):
    if relative_path in EXEMPT:
        pytest.skip(EXEMPT[relative_path])

    assert REQUIRED_KWARG in keywords, (
        f"{relative_path} builds a Fitness without `{REQUIRED_KWARG}`, so "
        "quick updates silently never fire for this search while the startup "
        "log still announces a cadence. Forward it, or add the site to EXEMPT "
        "with the reason."
    )
