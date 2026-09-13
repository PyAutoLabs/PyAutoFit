"""
Regenerate the committed model-figure evidence renders.

Run from the PyAutoFit repo root, with the worktree environment active::

    python docs/images/model_figures/make_figures.py

Every PNG in this directory is written by this script at ``width=14.0`` inches
and DPI 100, so none exceeds the 1400 px width budget. The figures are the
acceptance evidence for the ``model-figures`` epic's phase 2 and the images the
cookbook docs pages embed.

Most models here are lifted from the PyAutoFit cookbooks (``autofit_workspace``
``scripts/cookbooks/{model,multi_level_model,multiple_datasets}.py``) so the
figure a reader sees in the docs is the model the surrounding code composes.

The three lens figures (``simple_lens``, ``mge_pixelized``, ``group_scale``) are
built from ``test_autofit/graph_spec/lens_doubles.py``. Those are **structural
doubles** -- classes with the same constructor signatures, priors and
composition as the real PyAutoLens components, so PyAutoFit can render them
without depending on PyAutoLens. Phase 3 of the epic renders the real models and
replaces them.
"""

import sys
from pathlib import Path
from typing import List

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import autofit as af
from autofit import graphical as graph
from autofit.graphical.expectation_propagation.factor_optimiser import (
    AbstractFactorOptimiser,
)
from autofit.mapper.variable import Variable
from autofit.messages.normal import NormalMessage
from autofit.tools.namer import namer

OUTPUT_PATH = Path(__file__).resolve().parent
WIDTH = 14.0


def write(model, name: str, **kwargs):
    """Render one model to ``<name>.png`` in this directory."""
    af.ModelPlotter(model).figure(
        path=str(OUTPUT_PATH),
        filename=name,
        format="png",
        width=WIDTH,
        **kwargs,
    )
    print(f"wrote {name}.png")


# -- the cookbook models ----------------------------------------------------


def gaussian():
    """``scripts/cookbooks/model.py`` -- the plain ``af.Model``."""
    return af.Model(af.ex.Gaussian)


def gaussian_customised():
    """
    ``scripts/cookbooks/model.py`` -- "Model Customization (Model)".

    ``centre`` is fixed, ``normalization`` is given a new prior and ``sigma``
    keeps the prior its configuration file supplies.
    """
    model = af.Model(af.ex.Gaussian)
    model.centre = 0.0
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    return model


def gaussian_collection():
    """
    ``scripts/cookbooks/model.py`` -- "Model Customization (Collection)".
    """
    gaussian = af.Model(af.ex.Gaussian)
    gaussian.normalization = 1.0
    gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    exponential = af.Model(af.ex.Exponential)
    exponential.centre = 50.0
    exponential.add_assertion(exponential.rate > 5.0)

    return af.Collection(gaussian=gaussian, exponential=exponential)


class MultiLevelGaussian:
    def __init__(self, normalization=1.0, sigma=5.0):
        self.normalization = normalization
        self.sigma = sigma


class MultiLevelGaussians:
    def __init__(
        self,
        higher_level_centre: float = 50.0,
        gaussian_list: List[MultiLevelGaussian] = None,
    ):
        self.higher_level_centre = higher_level_centre
        self.gaussian_list = gaussian_list


def multi_level():
    """
    ``scripts/cookbooks/multi_level_model.py`` -- the multi-level composition:
    two ``Gaussian``s whose centre is a parameter of the level above them.
    """
    return af.Model(
        MultiLevelGaussians,
        gaussian_list=[af.Model(MultiLevelGaussian), af.Model(MultiLevelGaussian)],
    )


class _Analysis(af.Analysis):
    """
    A do-nothing analysis. ``AnalysisFactor`` only needs an analysis object to
    pair with the model; nothing here is ever fitted.

    It carries a ``data`` attribute because that is what makes the figure draw
    an **observed** node for the dataset: observed data and a fixed model
    constant are different concepts and must not share the grey pill.
    """

    def __init__(self, index):
        self.index = index
        self.data = [0.0, 1.0, 2.0]

    def log_likelihood_function(self, instance):
        return 0.0


def _factor_graph(model_list):
    return af.FactorGraphModel(
        *[
            af.AnalysisFactor(prior_model=model, analysis=_Analysis(index))
            for index, model in enumerate(model_list)
        ]
    )


def _multiple_datasets_model():
    model = af.Model(af.ex.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
    return model


def multiple_datasets_shared():
    """
    ``scripts/cookbooks/multiple_datasets.py`` -- one model shared by all three
    datasets.
    """
    model = _multiple_datasets_model()
    return _factor_graph([model, model, model]).global_prior_model


def multiple_datasets_variable():
    """
    ``scripts/cookbooks/multiple_datasets.py`` -- a shared ``centre`` with a
    ``normalization`` and ``sigma`` free per dataset.
    """
    model = _multiple_datasets_model()

    model_list = []
    for _ in range(3):
        model_analysis = model.copy()
        model_analysis.normalization = af.LogUniformPrior(
            lower_limit=1e-2, upper_limit=1e2
        )
        model_analysis.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
        model_list.append(model_analysis)

    return _factor_graph(model_list).global_prior_model


def multiple_datasets_relational():
    """
    ``scripts/cookbooks/multiple_datasets.py`` -- ``sigma`` follows the relation
    ``sigma_m * x + sigma_c`` across the datasets, so adding datasets adds no
    parameters.
    """
    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

    sigma_m = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    model_list = []
    for x in [1.0, 2.0, 3.0]:
        model_analysis = model.copy()
        model_analysis.gaussian.sigma = (sigma_m * x) + sigma_c
        model_list.append(model_analysis)

    return _factor_graph(model_list).global_prior_model


# -- the graphical models ---------------------------------------------------


def _graphical_model():
    """One dataset's ``Gaussian``, with a centre that a hyper-prior can draw."""
    model = af.Model(af.ex.Gaussian)
    model.centre = af.TruncatedGaussianPrior(
        mean=50.0, sigma=20.0, lower_limit=0.0, upper_limit=100.0
    )
    model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
    return model


def graphical_shared():
    """
    Plate notation, **shared**: one ``centre``, ``normalization`` and ``sigma``
    for all three datasets -- literally the same three numbers.
    """
    model = _graphical_model()
    return _factor_graph([model, model, model]).global_prior_model


def graphical_variable():
    """
    Plate notation, **partly shared**: one ``centre`` for every dataset, with a
    ``normalization`` and ``sigma`` of its own per dataset.
    """
    model = _graphical_model()

    model_list = []
    for _ in range(3):
        model_analysis = model.copy()
        model_analysis.normalization = af.LogUniformPrior(
            lower_limit=1e-2, upper_limit=1e2
        )
        model_analysis.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
        model_list.append(model_analysis)

    return _factor_graph(model_list).global_prior_model


def _hierarchical_factor_graph_model():
    """
    Three *different* centres, each drawn from one parent ``GaussianPrior``
    whose ``mean`` and ``sigma`` are themselves free (HowToFit chapter 3,
    tutorial 4).

    Returned as the ``FactorGraphModel`` rather than its global prior model,
    because the EP figures below need the factor graph an ``EPOptimiser``
    sweeps and the model figure needs the model.
    """
    model_list = [_graphical_model() for _ in range(3)]

    hierarchical_factor = af.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.TruncatedGaussianPrior(
            mean=50.0, sigma=10, lower_limit=0.0, upper_limit=100.0
        ),
        sigma=af.TruncatedGaussianPrior(
            mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0
        ),
    )
    for model in model_list:
        hierarchical_factor.add_drawn_variable(model.centre)

    return af.FactorGraphModel(
        *[
            af.AnalysisFactor(prior_model=model, analysis=_Analysis(index))
            for index, model in enumerate(model_list)
        ],
        hierarchical_factor,
    )


def graphical_hierarchical():
    """
    Plate notation, **hierarchical**: three *different* centres, each drawn from
    one parent ``GaussianPrior`` whose ``mean`` and ``sigma`` are themselves
    free (HowToFit chapter 3, tutorial 4).

    The teaching pair of the epic: put this figure beside ``graphical_shared``
    and the difference between "the same number" and "from a common population"
    is the difference between a blue reference and a violet arrow.
    """
    return _hierarchical_factor_graph_model().global_prior_model


def composite_shared_relation_assertion():
    """
    All three of sharing, relation and assertion at once -- acceptance case 3.
    """
    model = af.Collection(
        a=af.Model(af.ex.Gaussian),
        b=af.Model(af.ex.Gaussian),
    )
    model.b.centre = model.a.centre
    model.b.sigma = model.a.sigma * 2.0
    model.add_assertion(model.a.sigma > 5.0)
    return model


# -- the lens structural doubles --------------------------------------------


def lens_models():
    from test_autofit.graph_spec.lens_doubles import (
        simple_lens_model,
        mge_model,
        group_model,
    )

    return {
        "simple_lens": simple_lens_model(),
        "mge_pixelized": mge_model(),
        "group_scale": group_model(),
    }


# -- the EP figures ---------------------------------------------------------
#
# `af.EPPlotter` draws the factor graph an `EPOptimiser` sweeps, and -- given
# the run's history -- what the run did to it. These three PNGs are the
# phase-5 evidence and the images `docs/features/graphical.md` embeds.
#
# `ep_state_reverted` is a real `EPOptimiser.run`. `ep_state_stale` is not: a
# plate only forms over `AnalysisFactor`s, and an `AnalysisFactor` needs a real
# non-linear search per factor per sweep, which is minutes for a picture whose
# content is entirely determined by the statuses the run records. Its history
# is therefore built from exactly the `Status` objects `OverWideFit` produces
# -- a projection rejected on every sweep, so the factor's message never moves
# -- which is the same thing the real run below writes for its own factor.


def write_ep(plotter, name: str, kind: str):
    """Render one EP view to ``<name>.png`` in this directory."""
    plotter.figure(
        path=str(OUTPUT_PATH),
        filename=name,
        format="png",
        kind=kind,
        width=WIDTH,
    )
    print(f"wrote {name}.png")


class OverWideFit(AbstractFactorOptimiser):
    """
    A factor fit that comes back wider than its cavity in every variable -- the
    shape a near-singular or noisy finite-difference Hessian produces. The
    quotient ``q* / cavity`` then has negative precision, so ``update_invalid``
    reverts every parameter and the factor's message never moves.

    Copied from ``test_autofit/graphical/functionality/test_factor_failure_recovery.py``,
    where the reversion bugs (PyAutoFit #1571, #1575, #1579) were fixed against
    it, so the figures show the behaviour the tests pin.
    """

    def optimise(self, factor_approx, status=graph.Status()):
        model_dist = graph.MeanField(
            {
                v: NormalMessage(float(m.mean), float(m.sigma) * 3.0)
                for v, m in factor_approx.cavity_dist.items()
            }
        )
        return model_dist, graph.Status(success=True, messages=(), updated=True)


class PartialRevertFit(AbstractFactorOptimiser):
    """
    A factor fit that is valid in one variable and over-wide in another, on
    every sweep: the quotient ``q* / cavity`` has positive precision for the
    first (so it updates) and negative precision for the second (so
    ``update_invalid`` reverts every one of its parameters and its message never
    moves). The factor updates; one of its variables never does.
    """

    def __init__(self, reverting: str):
        super().__init__()
        self.reverting = reverting

    def optimise(self, factor_approx, status=graph.Status()):
        model_dist = graph.MeanField(
            {
                v: NormalMessage(
                    float(m.mean),
                    float(m.sigma) * (3.0 if v.name == self.reverting else 0.5),
                )
                for v, m in factor_approx.cavity_dist.items()
            }
        )
        return model_dist, graph.Status(success=True, messages=(), updated=True)


def ep_model():
    """
    The **model** view: the hierarchical factor graph, drawable before the
    first sweep and written once per run as ``graph_model.png``.

    The same model as ``graphical_hierarchical`` -- so the two figures can be
    read side by side -- but drawn as the explicit factor graph EP sweeps:
    square factor nodes, round variable nodes, and the edges between them.
    """
    return af.EPPlotter(_hierarchical_factor_graph_model().graph), "model"


def ep_state_stale():
    """
    The **state** view with a stalled plate member: three datasets, one of
    which never updates.

    The acceptance the epic review asked for by name -- a plate must never
    report "3 datasets, working" when one of its members has done nothing. The
    plate carries the count and names the failure; the failing member is
    expanded beside it in grey, badged with its zero updates.
    """
    from autofit.graphical.declarative.factor.analysis import AnalysisFactor
    from autofit.graphical.expectation_propagation.history import (
        EPHistory,
        FactorHistory,
    )
    from autofit.graphical.utils import StatusFlag

    factor_graph = _hierarchical_factor_graph_model().graph
    analysis_factors = [
        factor for factor in factor_graph.factors if isinstance(factor, AnalysisFactor)
    ]
    stalled = analysis_factors[-1]

    history = EPHistory(kl_tol=None, evidence_tol=None)
    for factor in factor_graph.factors:
        entry = FactorHistory(factor)
        for _ in range(4):
            if factor is stalled:
                status = graph.Status(
                    success=True,
                    updated=False,
                    flag=StatusFlag.BAD_PROJECTION,
                    changed={variable: False for variable in factor.variables},
                )
            else:
                status = graph.Status(
                    success=True, updated=True, flag=StatusFlag.SUCCESS
                )
            entry.history.append((None, status))
        history.history[factor] = entry

    return af.EPPlotter(factor_graph, ep_history=history), "state"


def ep_state_reverted():
    """
    The **state** view of a real four-sweep EP run in which one factor's
    projection is rejected for one of its two variables on every sweep.

    ``like_xy`` updates -- so no factor-level staleness is reported -- while
    ``y``'s message never moves off the one it started with. The figure marks
    the ``(factor, variable)`` pair it happened to, which is the only place
    that fact is visible (PyAutoFit #1575).
    """
    from autofit.graphical.expectation_propagation.factor_optimiser import (
        ExactFactorFit,
    )
    from autofit.graphical.expectation_propagation.history import EPHistory

    x, y = Variable("x"), Variable("y")

    def joint(x, y):
        return -0.5 * (np.sum((x - 3.0) ** 2) + np.sum((y - 2.0) ** 2))

    prior_x = NormalMessage(1.0, 2.0).as_factor(x, name="prior_x")
    prior_y = NormalMessage(1.0, 2.0).as_factor(y, name="prior_y")
    likelihood = graph.Factor(joint, x, y, name="like_xy")

    factor_graph = graph.FactorGraph([prior_x, prior_y, likelihood])
    model_approx = graph.EPMeanField.from_approx_dists(
        factor_graph,
        {x: NormalMessage(0.0, 10.0), y: NormalMessage(0.0, 10.0)},
    )

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={
            prior_x: ExactFactorFit(),
            prior_y: ExactFactorFit(),
            likelihood: PartialRevertFit(reverting="y"),
        },
        # `kl_tol=None` disables the convergence check: this graph is exact and
        # would otherwise be declared converged after one sweep.
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)

    return af.EPPlotter(factor_graph, ep_history=optimiser.ep_history), "state"


EP_FIGURES = {
    "ep_model": ep_model,
    "ep_state_stale": ep_state_stale,
    "ep_state_reverted": ep_state_reverted,
}


FIGURES = {
    "gaussian": gaussian,
    "gaussian_customised": gaussian_customised,
    "gaussian_collection": gaussian_collection,
    "multi_level": multi_level,
    "multiple_datasets_shared": multiple_datasets_shared,
    "multiple_datasets_variable": multiple_datasets_variable,
    "multiple_datasets_relational": multiple_datasets_relational,
    "composite_shared_relation_assertion": composite_shared_relation_assertion,
    "graphical_shared": graphical_shared,
    "graphical_variable": graphical_variable,
    "graphical_hierarchical": graphical_hierarchical,
}


def main():
    for name, builder in FIGURES.items():
        write(builder(), name)

    for name, model in lens_models().items():
        write(model, name)

    ep_main()


def ep_main():
    """
    The EP figures alone -- the phase-5 evidence.

    ``namer`` -- the source of a declarative factor's name
    (``AnalysisFactor0``, ``HierarchicalFactor0``) -- is a global counter, so it
    is reset before each figure. Otherwise the second graph built in a session
    is labelled ``AnalysisFactor3`` and a reader comparing two figures is
    comparing different numbers for the same thing.
    """
    for name, builder in EP_FIGURES.items():
        namer.reset()
        plotter, kind = builder()
        write_ep(plotter, name, kind)


if __name__ == "__main__":
    main()
