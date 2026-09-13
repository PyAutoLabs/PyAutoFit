"""
Layer 1 of the EP figure: the structure read off a ``DeclarativeFactorGraph``.

The fixtures are the phase-4 graphical doubles, so a spec failure and a figure
regression are the same failure. They are built the way ``EPOptimiser`` builds
its graph -- ``AbstractDeclarativeFactor.graph``, once -- because that is the
object the optimiser sweeps and the one ``EPHistory`` is keyed by.
"""

import json

import pytest

import autofit as af
from autofit.graphical.declarative.abstract import AbstractDeclarativeFactor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.model_figure.ep.spec import EPGraphSpec, factor_by_key, label_for_variable
from test_autofit.graph_spec.graphical_doubles import (
    hierarchical_graph,
    shared_graph,
    variable_graph,
)


@pytest.fixture
def hierarchical_factor_graph():
    """
    The ``DeclarativeFactorGraph`` ``EPOptimiser`` would be handed: three
    ``AnalysisFactor``s, one ``HierarchicalFactor`` decomposed into three
    members, and one ``PriorFactor`` per prior.
    """
    return hierarchical_graph().graph


@pytest.fixture
def spec(hierarchical_factor_graph):
    return EPGraphSpec.from_factor_graph(hierarchical_factor_graph)


def test_the_graph_is_read_once_and_never_rebuilt(monkeypatch):
    """
    ``AbstractDeclarativeFactor.graph`` rebuilds the whole graph and renames
    every ``PriorFactor`` on each access, and ``FactorGraph.graph`` builds a
    networkx view; the spec must touch neither, and must walk
    ``factor_graph.factors`` exactly once.
    """
    rebuilds = []
    networkx_views = []
    factor_reads = []

    declarative_graph = AbstractDeclarativeFactor.graph.fget
    networkx_graph = FactorGraph.graph.fget
    factors = FactorGraph.factors.fget

    def counted(log, original):
        def wrapper(self):
            log.append(1)
            return original(self)

        return property(wrapper)

    monkeypatch.setattr(
        AbstractDeclarativeFactor, "graph", counted(rebuilds, declarative_graph)
    )
    monkeypatch.setattr(FactorGraph, "graph", counted(networkx_views, networkx_graph))
    monkeypatch.setattr(FactorGraph, "factors", counted(factor_reads, factors))

    factor_graph = hierarchical_graph().graph
    assert len(rebuilds) == 1

    before = len(factor_reads)
    EPGraphSpec.from_factor_graph(factor_graph)

    assert len(factor_reads) - before == 1
    assert len(rebuilds) == 1
    assert networkx_views == []


def test_counts(spec):
    """
    Six drawn factors (the eleven ``PriorFactor``s are hidden), five variable
    nodes -- three collapsed inside the dataset plate plus the two free
    hyper-parameters -- and one edge per (factor, variable) pair.
    """
    assert len(spec.factors) == 6
    assert len(spec.variables) == 5
    assert len(spec.incidences) == 18
    assert len(spec.plates) == 2


def test_the_three_analysis_factors_are_one_plate(spec):
    (plate,) = [plate for plate in spec.plates if plate.kind == "analysis"]

    assert plate.count == 3
    assert plate.member_keys == ("factor-0", "factor-1", "factor-2")
    assert plate.representative_key == "factor-0"

    members = [factor for factor in spec.factors if factor.plate_key == plate.key]
    assert [factor.name for factor in members] == [
        "AnalysisFactor0",
        "AnalysisFactor1",
        "AnalysisFactor2",
    ]
    assert [factor.member_index for factor in members] == [0, 1, 2]
    assert all(factor.kind == "analysis" for factor in members)


def test_the_three_hierarchical_factors_are_one_group(spec):
    """
    A ``HierarchicalFactor`` decomposes into one factor per drawn variable, all
    carrying ``distribution_model.name`` -- the optimiser's own group key.
    """
    (plate,) = [plate for plate in spec.plates if plate.kind == "hierarchical"]

    assert plate.count == 3
    assert plate.member_keys == ("factor-3", "factor-4", "factor-5")
    assert plate.signature == ("hierarchical", "HierarchicalFactor0")

    members = [factor for factor in spec.factors if factor.plate_key == plate.key]
    assert {factor.name for factor in members} == {"HierarchicalFactor0"}
    assert [factor.member_index for factor in members] == [0, 1, 2]


def test_a_variable_of_one_member_is_collapsed_into_the_plate(spec):
    """
    Each dataset carries its own ``centre``, ``normalization`` and ``sigma``:
    one node apiece, inside the plate, standing for three variables.
    """
    plated = [
        variable for variable in spec.variables if variable.plate_key == "plate-0"
    ]

    assert [variable.label for variable in plated] == [
        "centre",
        "normalization",
        "sigma",
    ]
    assert [variable.count for variable in plated] == [3, 3, 3]


def test_a_variable_of_every_member_is_shared_outside_the_plate(spec):
    """
    The parent distribution's ``mean`` and ``sigma`` are incident on all three
    hierarchical members, so they are drawn once, outside.
    """
    shared = [variable for variable in spec.variables if variable.plate_key is None]

    assert [variable.label for variable in shared] == [
        "HierarchicalFactor0.mean",
        "HierarchicalFactor0.sigma",
    ]
    assert [variable.count for variable in shared] == [1, 1]
    assert [variable.kind for variable in shared] == ["hyper", "hyper"]


def test_the_drawn_variable_is_marked_as_drawn(spec):
    (centre,) = [variable for variable in spec.variables if variable.label == "centre"]
    assert centre.kind == "drawn"

    # ... and every hierarchical member reaches into the dataset plate for it
    hierarchical = [factor for factor in spec.factors if factor.kind == "hierarchical"]
    assert all(centre.key in factor.variable_keys for factor in hierarchical)


def test_prior_factors_are_hidden_by_default(spec):
    assert [factor for factor in spec.factors if factor.kind == "prior"] == []
    assert all(
        not incidence.factor_key.endswith("/prior") for incidence in spec.incidences
    )

    # but the spec still records which variables have one, so the render layer
    # can draw a stub without eleven extra nodes
    assert set(spec.prior_factor_variable_keys) == {
        variable.key for variable in spec.variables
    }


def test_prior_factors_are_drawn_when_asked_for(hierarchical_factor_graph):
    shown = EPGraphSpec.from_factor_graph(
        hierarchical_factor_graph, show_prior_factors=True
    )

    stubs = [factor for factor in shown.factors if factor.kind == "prior"]

    # one per *variable node*, not one per prior: the plate collapses the three
    # copies of each per-dataset prior into one
    assert len(stubs) == 5
    assert {stub.key for stub in stubs} == {
        f"{variable.key}/prior" for variable in shown.variables
    }
    for stub in stubs:
        (variable_key,) = stub.variable_keys
        (variable,) = [v for v in shown.variables if v.key == variable_key]
        assert stub.plate_key == variable.plate_key

    assert len(shown.incidences) == len(
        EPGraphSpec.from_factor_graph(hierarchical_factor_graph).incidences
    ) + len(stubs)


def test_variable_nodes_and_keys_do_not_move_when_prior_factors_are_shown(
    hierarchical_factor_graph,
):
    hidden = EPGraphSpec.from_factor_graph(hierarchical_factor_graph)
    shown = EPGraphSpec.from_factor_graph(
        hierarchical_factor_graph, show_prior_factors=True
    )
    assert hidden.variables == shown.variables
    assert hidden.plates == shown.plates


def test_no_label_is_a_variable_counter_form(spec, hierarchical_factor_graph):
    """
    ``variable.label`` is a global-counter form -- ``centre1``, ``cls3``,
    ``mean0``. It names nothing a reader of ``graph.info`` can find, so no spec
    label may be one.
    """
    counter_forms = {
        variable.label for variable in hierarchical_factor_graph.all_variables
    }
    assert counter_forms  # the fixture really does carry them

    labels = {variable.label for variable in spec.variables}
    labels |= {incidence.label for incidence in spec.incidences}

    assert labels.isdisjoint(counter_forms)


def test_labels_are_unique_within_a_factor(spec):
    """
    ``_HierarchicalFactor.name_for_variable`` answers with the distribution
    model's name for every one of its variables; the spec has to separate them
    or the figure carries three nodes reading ``HierarchicalFactor0``.
    """
    for factor in spec.factors:
        labels = [
            incidence.label
            for incidence in spec.incidences
            if incidence.factor_key == factor.key
        ]
        assert len(labels) == len(set(labels))


def test_keys_carry_no_ids_or_namer_counters(spec):
    """
    Keys are built from declaration indices alone. A prior's id or a ``namer``
    counter in a key would make the figure a function of import order.
    """
    keys = (
        [factor.key for factor in spec.factors]
        + [variable.key for variable in spec.variables]
        + [plate.key for plate in spec.plates]
    )
    for key in keys:
        assert "AnalysisFactor" not in key
        assert "HierarchicalFactor" not in key
        assert "prior_" not in key


def test_factor_by_key_maps_onto_the_live_factors(hierarchical_factor_graph, spec):
    mapping = factor_by_key(hierarchical_factor_graph)

    assert len(mapping) == len(hierarchical_factor_graph.factors)
    for factor in spec.factors:
        assert mapping[factor.key].name == factor.name


def test_to_dict_is_json_serialisable_and_stable(hierarchical_factor_graph):
    first = EPGraphSpec.from_factor_graph(hierarchical_factor_graph).to_dict()
    second = EPGraphSpec.from_factor_graph(hierarchical_factor_graph).to_dict()

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_two_builds_of_the_same_model_agree():
    """
    The spec is a function of the model, not of the process: a fresh build of
    the same graph -- new priors, new ids -- gives the same dictionary.
    """
    import itertools

    from autofit.tools.namer import namer

    def build():
        af.ModelObject._ids = itertools.count()
        af.Prior._ids = itertools.count()
        namer.reset()
        return EPGraphSpec.from_factor_graph(hierarchical_graph().graph).to_dict()

    assert build() == build()


def test_shared_graph_draws_every_prior_outside_the_plate():
    """
    ``shared_graph`` gives all three datasets one model object, so every prior
    is literally shared: the plate is a plate of three factors with nothing
    inside it.
    """
    spec = EPGraphSpec.from_factor_graph(shared_graph().graph)

    (plate,) = spec.plates
    assert plate.count == 3
    assert [variable.plate_key for variable in spec.variables] == [None, None, None]
    assert [variable.count for variable in spec.variables] == [1, 1, 1]


def test_variable_graph_splits_shared_from_per_dataset_priors():
    """
    ``variable_graph`` shares ``centre`` and gives each dataset its own
    ``normalization`` and ``sigma`` -- the two halves of the plate rule on one
    graph.
    """
    spec = EPGraphSpec.from_factor_graph(variable_graph().graph)

    outside = [variable for variable in spec.variables if variable.plate_key is None]
    inside = [variable for variable in spec.variables if variable.plate_key is not None]

    assert [variable.label for variable in outside] == ["AnalysisFactor0.centre"]
    assert [variable.label for variable in inside] == ["normalization", "sigma"]
    assert [variable.count for variable in inside] == [3, 3]


def test_label_for_variable_separates_a_plain_factors_arguments():
    """
    A plain ``Factor`` answers with its own name for every argument -- the
    shape ``test_factor_failure_recovery``'s doubles take, and the one the
    state overlay has to map a reverted variable through.
    """
    import numpy as np

    from autofit import graphical as graph
    from autofit.mapper.variable import Variable

    x, y = Variable("x"), Variable("y")
    factor = graph.Factor(lambda x, y: np.sum(x + y), x, y, name="like_xy")

    assert label_for_variable(factor, x) == "like_xy.x"
    assert label_for_variable(factor, y) == "like_xy.y"
