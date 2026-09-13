"""
Determinism -- the same graph must give the same EP figure, twice and everywhere.

The same acceptance as the model figure's own
``test_autofit/model_figure/test_determinism.py``, and for the same reason: a
figure that is not byte-reproducible cannot be diffed, cannot be committed as
evidence, and cannot be written every sweep of a fit without polluting the
output directory with noise. The layout dict is the artefact (every coordinate
rounded to 4 dp); the PNG bytes are checked directly.

The EP figure has one source of drift the model figure does not -- the
``networkx`` layout it is seeded from -- which is why the fresh-interpreter leg
matters here even more than it does there.
"""

import hashlib
import itertools
import json
import subprocess
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")

import autofit as af
from autofit.model_figure.ep.layout import build_ep_layout
from autofit.model_figure.ep.presentation import build_ep_presentation
from autofit.model_figure.ep.render import draw, save
from autofit.model_figure.ep.spec import EPGraphSpec
from autofit.tools.namer import namer
from test_autofit.graph_spec.graphical_doubles import hierarchical_graph


def _reset_ids():
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
    namer.reset()


def _layout(factor_graph, state=None):
    return build_ep_layout(
        build_ep_presentation(EPGraphSpec.from_factor_graph(factor_graph), state)
    )


def _layout_json(factor_graph):
    return json.dumps(_layout(factor_graph).to_dict())


def test_laying_out_the_same_graph_twice_is_byte_identical(hierarchical_factor_graph):
    assert _layout_json(hierarchical_factor_graph) == _layout_json(
        hierarchical_factor_graph
    )


def test_rebuilding_the_graph_is_byte_identical():
    """
    The figure is a function of the model, not of the process: a fresh build --
    new priors, new ids, new ``namer`` counters -- gives the same dictionary.
    """
    _reset_ids()
    first = _layout_json(hierarchical_graph().graph)

    _reset_ids()
    second = _layout_json(hierarchical_graph().graph)

    assert first == second


def test_the_layout_is_identical_in_a_fresh_interpreter(tmp_path):
    """
    Text metrics come from a scratch Agg canvas and the base coordinates come
    from ``networkx``: neither may depend on anything the test session has
    already done.
    """
    _reset_ids()
    expected = _layout_json(hierarchical_graph().graph)

    target = tmp_path / "layout.json"
    script = textwrap.dedent(
        f"""
        import itertools, json, pathlib
        import matplotlib
        matplotlib.use("Agg")
        import autofit as af
        from autofit.model_figure.ep.layout import build_ep_layout
        from autofit.model_figure.ep.presentation import build_ep_presentation
        from autofit.model_figure.ep.spec import EPGraphSpec
        from autofit.tools.namer import namer
        from test_autofit.graph_spec.graphical_doubles import hierarchical_graph

        af.ModelObject._ids = itertools.count()
        af.Prior._ids = itertools.count()
        namer.reset()

        spec = EPGraphSpec.from_factor_graph(hierarchical_graph().graph)
        layout = build_ep_layout(build_ep_presentation(spec))
        pathlib.Path({str(target)!r}).write_text(json.dumps(layout.to_dict()))
        """
    )
    subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert target.read_text() == expected


def test_png_bytes_are_identical_across_two_saves(hierarchical_factor_graph, tmp_path):
    for filename in ("first", "second"):
        save(
            draw(_layout(hierarchical_factor_graph)),
            path=tmp_path,
            filename=filename,
            format="png",
        )

    first = hashlib.sha256((tmp_path / "first.png").read_bytes()).hexdigest()
    second = hashlib.sha256((tmp_path / "second.png").read_bytes()).hexdigest()
    assert first == second


def test_a_state_view_is_deterministic_too(stalled_plate, tmp_path):
    """
    The overlay is written every sweep of a fit, so it is the view that would
    fill an output directory with noise if it were not stable.
    """
    from autofit.model_figure.ep.spec import factor_by_key
    from autofit.model_figure.ep.state import EPState

    factor_graph, history, _ = stalled_plate
    spec = EPGraphSpec.from_factor_graph(factor_graph)
    state = EPState.from_history(spec, history, factor_by_key(factor_graph))

    first = json.dumps(_layout(factor_graph, state).to_dict())
    second = json.dumps(_layout(factor_graph, state).to_dict())

    assert first == second


def test_importing_the_ep_package_does_not_import_matplotlib():
    """
    ``af.EPPlotter`` is exported from ``autofit/__init__.py``, so every drawing
    import must sit inside a function -- otherwise every ``import autofit``
    pays for matplotlib.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, autofit.model_figure.ep; "
            "print([m for m in sys.modules if m.split('.')[0] == 'matplotlib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]"
