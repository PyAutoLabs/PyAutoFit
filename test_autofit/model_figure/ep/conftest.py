import itertools

import pytest

import autofit as af
from autofit.tools.namer import namer


@pytest.fixture(autouse=True)
def reset_ids():
    """
    Model object and prior ids are global counters, and so is ``namer`` -- the
    source of a declarative factor's name (``AnalysisFactor0``,
    ``HierarchicalFactor0``), which the EP spec records.  They are reset before
    every test so that the spec is a function of the factor graph alone, which
    is what the determinism assertions rely on.
    """
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
    namer.reset()


@pytest.fixture
def hierarchical_factor_graph():
    """
    The ``DeclarativeFactorGraph`` ``EPOptimiser`` would be handed for the
    phase-4 hierarchical double: three ``AnalysisFactor``s, one
    ``HierarchicalFactor`` decomposed into three members, and one
    ``PriorFactor`` per prior.
    """
    from test_autofit.graph_spec.graphical_doubles import hierarchical_graph

    return hierarchical_graph().graph


@pytest.fixture
def stalled_plate(hierarchical_factor_graph):
    """
    A plate of three datasets, one of which never updates.

    ``AnalysisFactor``s need a real non-linear search, so a real EP run over
    this graph is far too slow for a test -- and the per-factor optimiser route
    ``test_factor_failure_recovery`` uses needs one optimiser per factor, which
    is the same problem. The ``EPHistory`` is therefore built by hand from the
    same ``Status`` objects a run would record: two members that update on
    every sweep, and one whose projection is rejected on every sweep, which is
    exactly what ``OverWideFit`` produces.

    Returns ``(factor_graph, ep_history, stalled_factor)``.
    """
    from autofit.graphical.expectation_propagation.history import (
        EPHistory,
        FactorHistory,
    )
    from autofit.graphical.utils import Status, StatusFlag
    from autofit.graphical.declarative.factor.analysis import AnalysisFactor

    history = EPHistory(kl_tol=None, evidence_tol=None)
    analysis_factors = [
        factor
        for factor in hierarchical_factor_graph.factors
        if isinstance(factor, AnalysisFactor)
    ]
    stalled = analysis_factors[-1]

    for factor in hierarchical_factor_graph.factors:
        entry = FactorHistory(factor)
        for _ in range(4):
            if factor is stalled:
                status = Status(
                    success=True,
                    updated=False,
                    flag=StatusFlag.BAD_PROJECTION,
                    changed={variable: False for variable in factor.variables},
                )
            else:
                status = Status(success=True, updated=True, flag=StatusFlag.SUCCESS)
            entry.history.append((None, status))
        history.history[factor] = entry

    return hierarchical_factor_graph, history, stalled
