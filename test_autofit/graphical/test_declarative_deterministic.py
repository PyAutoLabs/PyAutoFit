"""
Seam tests for the declarative → graph lowering contract
(`autofit/graphical/README.md` §8), focused on deterministic quantities
expressed as compound priors.

History: PR #1153 introduced this pattern with an illustration test that
was committed commented-out; the accompanying `model.<property>` sugar
was later deliberately reverted (`be6411755`, "undo annoying change").
The *explicit* compound-prior pattern below is the supported one — these
tests pin its lowering behaviour so the seam cannot silently drift again
(EP review Phase 5, PyAutoFit #1336).
"""
import pytest

import autofit as af
from autofit.mock import MockAnalysis

FWHM_FACTOR = 2.354820045


@pytest.fixture(name="factor_graph_pieces")
def make_factor_graph_pieces():
    model_1 = af.Model(af.ex.Gaussian)
    model_2 = af.Model(af.ex.Gaussian)

    # Deterministic quantities built explicitly from other models' priors.
    deterministic_model = af.Collection(
        model_1.sigma * FWHM_FACTOR,
        model_2.sigma * FWHM_FACTOR,
    )

    deterministic_factor = af.AnalysisFactor(
        prior_model=deterministic_model, analysis=MockAnalysis()
    )
    factor_graph = af.FactorGraphModel(
        af.AnalysisFactor(prior_model=model_1, analysis=MockAnalysis()),
        af.AnalysisFactor(prior_model=model_2, analysis=MockAnalysis()),
        deterministic_factor,
    )
    return model_1, model_2, deterministic_model, deterministic_factor, factor_graph


def test_compound_lowering_adds_no_variables(factor_graph_pieces):
    """
    A compound prior lowers to *no* graph variable: only its component
    priors are variables, and they are shared with the source models
    (README §8 — the relation is enforced exactly inside each factor).
    """
    model_1, model_2, deterministic_model, _, factor_graph = factor_graph_pieces

    source_priors = set(model_1.priors) | set(model_2.priors)
    compound_priors = set(deterministic_model.priors)

    assert len(compound_priors) == 2
    assert compound_priors <= source_priors


def test_graph_shape(factor_graph_pieces):
    """
    3 AnalysisFactors + one PriorFactor per free parameter (2 models x 3
    parameters = 6). The deterministic AnalysisFactor contributes no new
    PriorFactors because its priors are the shared components.
    """
    *_, factor_graph = factor_graph_pieces
    approx = factor_graph.mean_field_approximation()

    names = [factor.name for factor in approx.factor_graph.factors]
    n_analysis = sum("AnalysisFactor" in name for name in names)
    n_prior = sum("PriorFactor" in name for name in names)

    assert n_analysis == 3
    assert n_prior == 6

    # Every variable in the mean field carries a message.
    mean_field = approx.mean_field
    assert len(mean_field) == 6


def test_compound_realises_inside_factor(factor_graph_pieces):
    """
    The deterministic relation is evaluated at instance-creation inside
    the factor: realising the deterministic model at the prior medians
    gives exactly sigma_median * FWHM_FACTOR.
    """
    model_1, _, deterministic_model, _, _ = factor_graph_pieces

    instance = deterministic_model.instance_from_prior_medians()
    sigma_median = model_1.sigma.value_for(0.5)

    assert instance[0] == pytest.approx(sigma_median * FWHM_FACTOR)


def test_deterministic_factor_connects_through_shared_variables(
    factor_graph_pieces,
):
    """
    The deterministic AnalysisFactor's graph variables are precisely the
    shared component priors — this is the channel through which its
    likelihood constrains the source models under EP.
    """
    (
        model_1,
        model_2,
        deterministic_model,
        deterministic_factor,
        factor_graph,
    ) = factor_graph_pieces
    approx = factor_graph.mean_field_approximation()

    # Identity-based lookup: AnalysisFactor names come from a global
    # counter, so name matching is test-order-dependent.
    assert deterministic_factor in approx.factor_graph.factors

    factor_variables = set(deterministic_factor.variables)
    assert factor_variables == set(deterministic_model.priors)
    assert factor_variables <= set(model_1.priors) | set(model_2.priors)
