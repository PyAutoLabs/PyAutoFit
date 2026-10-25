"""
The 18-construct catalogue.

One test per model construct from the phase-1 prompt
(``PyAutoMind/active/model_figures_1_graph_spec.md``).  Every assertion is on
the extracted :class:`~autofit.graph_spec.GraphSpec` -- never on a picture:
phase 1 draws nothing.
"""

import itertools

import pytest

import autofit as af
from autofit.example.model import PhysicalNFW
from autofit.graph_spec import GraphSpec, graph_spec_from

from .conftest import LatentAnalysis, NullAnalysis


def _reset_ids():
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()


# ---------------------------------------------------------------- 1


def test_01_model_with_default_priors():
    spec = GraphSpec.from_model(af.Model(af.ex.Gaussian))

    assert spec.root.path == ()
    assert spec.root.name == "model"
    assert spec.root.cls_name == "Gaussian"
    assert spec.root.kind == "model"
    assert spec.root.children == ()

    assert [row.name for row in spec.root.rows] == ["centre", "normalization", "sigma"]
    for row in spec.root.rows:
        assert row.sampling == "free"
        assert row.dimensionality == "scalar"
        assert row.provenance.kind == "config-default"
        assert row.prior_cls_name == "UniformPrior"
        assert row.shared is False
        assert row.in_model_info is True

    assert spec.counts["unique_sampled_scalars"] == 3
    assert spec.counts["fixed_leaf_slots"] == 0
    assert spec.counts["components"] == 1
    assert spec.path_index["centre"] == (("centre",),)


# ---------------------------------------------------------------- 2


def _row_fingerprint(row):
    """Everything about a row except the identity of its prior object."""
    data = row.to_dict()
    data.pop("prior_id")
    data.pop("occurrences")
    return data


def test_02_overridden_prior_is_indistinguishable_from_the_default():
    """
    KNOWN GAP: nothing on a ``Prior`` records whether it came from the config or
    from the user, so ``provenance`` is ``config-default`` either way.  This test
    pins the gap rather than papering over it -- ``user-prior`` is a reserved
    kind, not an emitted one.
    """
    default = GraphSpec.from_model(af.Model(af.ex.Gaussian))

    _reset_ids()
    overridden_model = af.Model(af.ex.Gaussian)
    overridden_model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    overridden = GraphSpec.from_model(overridden_model)

    default_row = default.row(("centre",))
    overridden_row = overridden.row(("centre",))

    assert overridden_row.provenance.kind == "config-default"
    assert _row_fingerprint(default_row) == _row_fingerprint(overridden_row)


# ---------------------------------------------------------------- 3


@pytest.mark.parametrize("build", ["assignment", "keyword"])
def test_03_fixed_float(build):
    if build == "assignment":
        model = af.Model(af.ex.Gaussian)
        model.centre = 0.0
    else:
        model = af.Model(af.ex.Gaussian, centre=0.0)

    spec = GraphSpec.from_model(model)
    row = spec.row(("centre",))

    assert row.sampling == "fixed"
    assert row.value == 0.0
    assert row.prior_cls_name == "Constant"
    assert row.prior_id is None
    assert row.dimensionality == "scalar"
    # Absent from the priors, present in `model.info`.
    assert ("centre",) not in [path for path, _ in model.path_priors_tuples]
    assert "centre" in model.info
    assert row.in_model_info is True
    assert spec.path_index["centre"] == (("centre",),)
    assert spec.counts["unique_sampled_scalars"] == 2
    assert spec.counts["fixed_leaf_slots"] == 1


# ---------------------------------------------------------------- 4


def test_04_shared_prior():
    model = af.Collection(
        a=af.Model(af.ex.Gaussian),
        b=af.Model(af.ex.Gaussian),
    )
    model.a.centre = model.b.centre

    # The raw tree: `a` and `b` are plate-mates under rule R1/R2, so the
    # collapsed spec draws them once (see `test_collapse.py`).  This construct
    # is about *extraction*, so it is asserted uncollapsed.
    spec = GraphSpec.from_model(model, collapse=False)

    a_centre = spec.row(("a", "centre"))
    b_centre = spec.row(("b", "centre"))

    assert a_centre.prior_id == b_centre.prior_id
    assert a_centre.occurrences == (("a", "centre"), ("b", "centre"))
    assert a_centre.shared is True
    # Sharing is never a sampling state.
    assert a_centre.sampling == "free"
    assert b_centre.sampling == "free"

    assert len(spec.shared) == 1
    assert spec.shared[0].prior_id == a_centre.prior_id
    assert spec.shared[0].occurrences == (("a", "centre"), ("b", "centre"))
    assert spec.counts["shared_priors"] == 1
    assert spec.counts["unique_sampled_scalars"] == 5


# ---------------------------------------------------------------- 5


def test_05_relation():
    model = af.Model(af.ex.Gaussian)
    model.centre = model.normalization + model.sigma

    spec = GraphSpec.from_model(model)
    row = spec.row(("centre",))

    assert row.prior_id is None
    assert row.prior_cls_name == "SumPrior"
    assert row.provenance.kind == "relation"
    assert row.provenance.expression == "normalization + sigma"
    assert row.provenance.operands == ("normalization", "sigma")
    # Its value varies during sampling, so its sampling status is free.
    assert row.sampling == "free"

    # `normalization` sits at ('normalization',) *and* at ('centre', 'self'),
    # because a compound prior's operands are real attributes of it.  The second
    # path passes through the relation, so it is not a second *use* of the
    # prior: the row is related, not shared, and there is no `SharedEdge`.
    normalization = spec.row(("normalization",))
    assert set(normalization.occurrences) == {("normalization",), ("centre", "self")}
    assert normalization.direct_occurrences == (("normalization",),)
    assert normalization.shared is False
    assert spec.shared == ()
    assert spec.counts["shared_priors"] == 0

    assert len(spec.relations) == 1
    edge = spec.relations[0]
    assert edge.target_path == ("centre",)
    assert edge.expression == "normalization + sigma"
    assert edge.operand_paths == (("normalization",), ("sigma",))
    assert edge.prior_id is None

    # The compound prior has `cls == float` and is not a component.
    assert spec.node(("centre",)) is None
    assert [node.path for node in spec.components()] == [()]


def test_05b_relation_over_constants_is_fixed():
    model = af.Model(af.ex.Gaussian)
    model.centre = model.sigma * 2.0
    spec = GraphSpec.from_model(model)
    assert spec.row(("centre",)).provenance.expression == "sigma * 2.0"

    _reset_ids()
    other = af.Model(af.ex.Gaussian)
    other.centre = -other.sigma
    assert GraphSpec.from_model(other).row(("centre",)).provenance.expression == (
        "-(sigma)"
    )


# ---------------------------------------------------------------- 6


def test_06_assertion():
    model = af.Model(af.ex.Gaussian)
    model.add_assertion(model.sigma > 0.5, name="sigma is large")

    spec = GraphSpec.from_model(model)

    assert len(spec.assertions) == 1
    edge = spec.assertions[0]
    assert edge.op == "<"
    assert edge.left == "0.5"
    assert edge.right == "sigma"
    # The graph carries the user-supplied assertion label.
    assert edge.name == "sigma is large"

    # Assertions are edges: not in the tree, and not in `model.info`.
    assert [row.name for row in spec.root.rows] == ["centre", "normalization", "sigma"]
    assert spec.path_index["assertion/0"] == ()
    assert "assert" not in model.info


# ---------------------------------------------------------------- 7


def test_07_tuple_parameter():
    model = af.Model(PhysicalNFW)
    spec = GraphSpec.from_model(model)

    assert [row.name for row in spec.root.rows] == [
        "centre",
        "ell_comps",
        "log10m",
        "concentration",
    ]

    centre = spec.row(("centre",))
    assert centre.dimensionality == "tuple"
    assert centre.prior_cls_name == "TuplePrior"
    assert centre.sampling == "free"
    assert [component.name for component in centre.components] == [
        "centre_0",
        "centre_1",
    ]
    assert [component.path for component in centre.components] == [
        ("centre", "centre_0"),
        ("centre", "centre_1"),
    ]
    for component in centre.components:
        assert component.sampling == "free"
        assert component.prior_cls_name == "GaussianPrior"
        assert component.prior_id is not None

    assert spec.path_index["centre/centre_0"] == (("centre", "centre_0"),)
    assert spec.counts["unique_sampled_scalars"] == 6


def test_07b_partially_fixed_tuple_is_a_mixed_state():
    model = af.Model(PhysicalNFW)
    model.centre.centre_1 = 0.5

    spec = GraphSpec.from_model(model)
    centre = spec.row(("centre",))

    assert [component.sampling for component in centre.components] == ["free", "fixed"]
    assert centre.components[1].value == 0.5
    assert centre.sampling == "free"
    assert spec.counts["fixed_leaf_slots"] == 1


# ---------------------------------------------------------------- 8


def test_08_named_collection():
    model = af.Collection(a=af.Model(af.ex.Gaussian), b=af.Model(af.ex.Gaussian))
    # Two identical siblings collapse into a plate; the construct under test is
    # the named-attribute children, so assert the uncollapsed tree.
    spec = GraphSpec.from_model(model, collapse=False)

    assert spec.root.kind == "collection"
    assert spec.root.cls_name == "Collection"
    assert [child.name for child in spec.root.children] == ["a", "b"]
    assert [child.path for child in spec.root.children] == [("a",), ("b",)]
    assert spec.root.rows == ()
    assert spec.counts["components"] == 3


# ---------------------------------------------------------------- 9


def test_09_list_collection():
    model = af.Collection(
        [af.Model(af.ex.Gaussian), af.Model(af.ex.Gaussian), af.Model(af.ex.Gaussian)]
    )
    spec = GraphSpec.from_model(model, collapse=False)

    assert [child.name for child in spec.root.children] == ["0", "1", "2"]
    assert [child.path for child in spec.root.children] == [("0",), ("1",), ("2",)]
    # `item_number` is bookkeeping, never a parameter slot.
    assert spec.root.rows == ()
    assert spec.counts["unique_sampled_scalars"] == 9


# ---------------------------------------------------------------- 10


def test_10_nested_collections():
    model = af.Collection(
        outer=af.Collection(inner=af.Collection(g=af.Model(af.ex.Gaussian)))
    )
    spec = GraphSpec.from_model(model)

    assert [node.path for node in spec.components()] == [
        (),
        ("outer",),
        ("outer", "inner"),
        ("outer", "inner", "g"),
    ]
    assert [node.kind for node in spec.components()] == [
        "collection",
        "collection",
        "collection",
        "model",
    ]
    assert spec.row(("outer", "inner", "g", "sigma")).sampling == "free"


# ---------------------------------------------------------------- 11


class MultiLevel:
    def __init__(self, gaussian_list, normalization=1.0):
        self.gaussian_list = gaussian_list
        self.normalization = normalization


def test_11_multi_level_model():
    model = af.Model(
        MultiLevel,
        gaussian_list=[af.Model(af.ex.Gaussian), af.Model(af.ex.Gaussian)],
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    )
    spec = GraphSpec.from_model(model, collapse=False)

    assert spec.root.cls_name == "MultiLevel"
    # The list became a child Collection at ('gaussian_list',).
    assert [child.path for child in spec.root.children] == [("gaussian_list",)]
    collection = spec.node(("gaussian_list",))
    assert collection.kind == "collection"
    assert [child.name for child in collection.children] == ["0", "1"]
    assert [row.name for row in spec.root.rows] == ["normalization"]
    assert spec.counts["unique_sampled_scalars"] == 7


# ---------------------------------------------------------------- 12


def test_12_instance_leaf():
    instance = af.ex.Gaussian(centre=1.0, normalization=2.0, sigma=3.0)
    model = af.Collection(a=af.Model(af.ex.Gaussian), b=instance)

    spec = GraphSpec.from_model(model)

    assert [child.name for child in spec.root.children] == ["a"]
    row = spec.row(("b",))
    assert row is not None
    assert row.is_instance is True
    assert row.sampling == "fixed"
    assert row.dimensionality == "scalar"
    assert row.prior_cls_name == "Gaussian"
    assert row.value is None
    assert row.prior_id is None
    # `model.info` shows it as `Gaussian (N=0)`.
    assert "Gaussian (N=0)" in model.info
    assert spec.counts["unique_sampled_scalars"] == 3


# ---------------------------------------------------------------- 13


def a_function(centre=0.0, normalization=1.0):
    return centre + normalization


def test_13_model_of_a_function():
    """
    Config priors cannot be resolved for a function (separately filed bug), so
    the priors are supplied explicitly.  ``model.cls`` is the function itself and
    ``cls.__name__`` still works.
    """
    model = af.Model(
        a_function,
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    )
    spec = GraphSpec.from_model(model)

    assert spec.root.cls_name == "a_function"
    assert spec.root.kind == "model"
    assert [row.name for row in spec.root.rows] == ["centre", "normalization"]
    assert all(row.sampling == "free" for row in spec.root.rows)
    assert spec.counts["unique_sampled_scalars"] == 2


# ---------------------------------------------------------------- 14


def test_14_array():
    model = af.Array(
        shape=(2, 2), prior=af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    )
    spec = GraphSpec.from_model(model)

    assert spec.root.cls_name == "Array"
    assert [row.name for row in spec.root.rows] == [
        "prior_0_0",
        "prior_0_1",
        "prior_1_0",
        "prior_1_1",
    ]
    # `shape` and `indices` are bookkeeping noise and are skipped.
    assert spec.row(("shape",)) is None
    assert spec.row(("indices",)) is None
    assert spec.counts["unique_sampled_scalars"] == 4


# ---------------------------------------------------------------- 15


def test_15_factor_graph_fully_shared():
    model = af.Model(af.ex.Gaussian)
    factor_graph = af.FactorGraphModel(
        af.AnalysisFactor(prior_model=model, analysis=NullAnalysis()),
        af.AnalysisFactor(prior_model=model, analysis=NullAnalysis()),
        af.AnalysisFactor(prior_model=model, analysis=NullAnalysis()),
    )
    spec = GraphSpec.from_model(factor_graph.global_prior_model, collapse=False)

    assert spec.root.kind == "global"
    assert spec.root.cls_name == "GlobalPriorModel"
    assert [child.name for child in spec.root.children] == ["0", "1", "2"]
    # Every child is the *same* model object.
    assert len({child.obj_id for child in spec.root.children}) == 1
    # Every parameter is one prior appearing at three paths.
    assert spec.counts["unique_sampled_scalars"] == 3
    assert spec.counts["shared_priors"] == 3
    for edge in spec.shared:
        assert len(edge.occurrences) == 3
    # The declarative factor graph itself is not a parameter slot.
    assert spec.row(("factor",)) is None


# ---------------------------------------------------------------- 16


def test_16_factor_graph_per_dataset_free_parameters():
    model_1 = af.Model(af.ex.Gaussian)
    model_2 = model_1.copy()
    model_2.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    factor_graph = af.FactorGraphModel(
        af.AnalysisFactor(prior_model=model_1, analysis=NullAnalysis()),
        af.AnalysisFactor(prior_model=model_2, analysis=NullAnalysis()),
    )
    spec = GraphSpec.from_model(factor_graph.global_prior_model, collapse=False)

    # Shared parameters keep one prior id; the freed one gets a new id.
    assert spec.row(("0", "centre")).prior_id == spec.row(("1", "centre")).prior_id
    assert spec.row(("0", "sigma")).prior_id != spec.row(("1", "sigma")).prior_id
    assert spec.row(("0", "sigma")).shared is False
    assert spec.row(("1", "sigma")).shared is False
    assert spec.counts["shared_priors"] == 2
    assert spec.counts["unique_sampled_scalars"] == 4


# ---------------------------------------------------------------- 17


def test_17_cross_dataset_relation():
    sm = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    sc = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    x = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    model_1 = af.Model(af.ex.Gaussian)
    model_2 = af.Model(af.ex.Gaussian)
    model_2.sigma = sm * x + sc

    factor_graph = af.FactorGraphModel(
        af.AnalysisFactor(prior_model=model_1, analysis=NullAnalysis()),
        af.AnalysisFactor(prior_model=model_2, analysis=NullAnalysis()),
    )
    spec = GraphSpec.from_model(factor_graph.global_prior_model)

    row = spec.row(("1", "sigma"))
    assert row.prior_cls_name == "SumPrior"
    assert row.provenance.kind == "relation"
    assert row.sampling == "free"
    # Nested SumPrior(MultiplePrior(sm, x), sc): the nesting is parenthesised
    # and each operand resolves to its dotted path in the model.
    assert row.provenance.expression == (
        "(1.sigma.self.sm * 1.sigma.self.x) + 1.sigma.sc"
    )
    assert row.provenance.operands == (
        "1.sigma.self.sm",
        "1.sigma.self.x",
        "1.sigma.sc",
    )

    assert len(spec.relations) == 1
    edge = spec.relations[0]
    assert edge.target_path == ("1", "sigma")
    assert edge.operand_paths == (
        ("1", "sigma", "self", "sm"),
        ("1", "sigma", "self", "x"),
        ("1", "sigma", "sc"),
    )
    assert spec.counts["unique_sampled_scalars"] == 8


# ---------------------------------------------------------------- 18


def test_18_latent_variables():
    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

    without = GraphSpec.from_model(model)
    assert [row.name for row in without.node(("gaussian",)).rows] == [
        "centre",
        "normalization",
        "sigma",
    ]
    assert without.row(("gaussian", "fwhm")) is None

    with_analysis = graph_spec_from(model, analysis=LatentAnalysis())
    row = with_analysis.row(("gaussian", "fwhm"))

    assert row is not None
    assert row.sampling == "solved"
    assert row.provenance.kind == "latent"
    # A latent is not part of the model at all.
    assert row.in_model_info is False
    assert with_analysis.path_index["gaussian/fwhm"] == ()
    assert "fwhm" not in model.info
    # Latents do not change the model's own counts.
    assert with_analysis.counts["unique_sampled_scalars"] == 3
