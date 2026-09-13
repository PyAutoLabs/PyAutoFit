"""
Layer 1 of the EP figure -- the structure of the factor graph.

:class:`EPGraphSpec` is a flat, frozen, JSON-serialisable description of the
graph an :class:`~autofit.graphical.EPOptimiser` sweeps: factor nodes, variable
nodes, the edges between them, and the plates that collapse repeated structure.
It holds no live objects, so the picture it describes cannot drift while a fit
runs.

Where the structure comes from
------------------------------

The source of truth is the optimiser's own ``factor_graph`` -- the
:class:`~autofit.graphical.declarative.graph.DeclarativeFactorGraph` handed to
:class:`~autofit.graphical.EPOptimiser`. It is **never** rebuilt from
``AbstractDeclarativeFactor.graph`` or ``model_approx.factor_graph``: that
property constructs a new graph on every access, renaming every
``PriorFactor`` as it goes, and ``EPHistory`` is keyed by the *optimiser's*
factor objects. So ``from_factor_graph`` iterates ``factor_graph.factors``
exactly once, into tuples.

Plates
------

Phase 4's ``_collapse_siblings`` needs a ``GlobalPriorModel`` and is therefore
unreachable here, so repetition is inferred at *factor* level:

* ``AnalysisFactor``s are grouped by **signature** -- the factor's class name,
  the tuple of dotted paths at which its prior model holds priors, and the
  tuple of prior class names at those paths. Three datasets fitted with the
  same ``Gaussian`` share a signature; a fourth with an extra parameter does
  not.
* ``_HierarchicalFactor``s are grouped by ``factor.name``, which is
  ``distribution_model.name`` -- the very key ``EPOptimiser`` itself uses to
  group a hierarchical factor's members when it reports staleness.

A group of two or more becomes a :class:`PlateGroup`, ordered by its first
member's index in ``factor_graph.factors``; members keep declaration order.

A variable incident on **exactly one** member of a plate is drawn *inside* it
as a single :class:`VariableNode` with ``count=N``; a variable incident on
**every** member is genuinely shared and is drawn once *outside* the plate
(``plate_key is None``). A variable that qualifies for two plates -- the drawn
``centre`` of a hierarchical model belongs to one ``AnalysisFactor`` and to one
``_HierarchicalFactor`` -- is placed in the first plate in plate order, and the
other plate's edge simply crosses into it.

Labels
------

Every label comes from :func:`label_for_variable`, i.e. from
``factor.name_for_variable`` -- never from ``variable.label``, which is a
global-counter form (``centre1``, ``cls3``) that means nothing to a reader.
Inside a plate the factor prefix is stripped, so ``AnalysisFactor0.centre``
reads ``centre``.

Keys
----

Every key is built from **declaration indices** -- a factor's position in
``factor_graph.factors`` and a variable's position within its owning factor.
No key contains an ``id()``, a ``Prior`` id or a ``namer`` counter, so two
builds of the same graph produce byte-identical keys.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "FactorNode",
    "VariableNode",
    "Incidence",
    "PlateGroup",
    "EPGraphSpec",
    "label_for_variable",
    "factor_key",
    "factor_by_key",
]

#: ``FactorNode.kind`` values.
FACTOR_KINDS = ("analysis", "hierarchical", "prior", "factor")

#: ``VariableNode.kind`` values. ``drawn`` is a variable a
#: ``HierarchicalFactor`` draws from its parent distribution; ``hyper`` is one
#: of that distribution's own parameters; everything else is ``free``.
VARIABLE_KINDS = ("free", "drawn", "hyper")


def factor_key(index: int) -> str:
    """
    The key of the factor at ``index`` in ``factor_graph.factors``.
    """
    return f"factor-{index}"


def factor_by_key(factor_graph) -> Dict[str, Any]:
    """
    Map every spec factor key to the live factor object it describes.

    :class:`~autofit.model_figure.ep.state.EPState` needs this to look a
    factor's :class:`~autofit.graphical.expectation_propagation.history.FactorHistory`
    up in an ``EPHistory``, which is keyed by the factor objects themselves.
    The spec deliberately holds no live objects, so the mapping is rebuilt here
    from the same declaration indices the keys are built from.

    Keys for collapsed ``PriorFactor`` stubs are absent: a stub stands for a
    whole group of prior factors and so has no single history.
    """
    return {
        factor_key(index): factor for index, factor in enumerate(factor_graph.factors)
    }


def _model_path(factor, variable) -> Optional[str]:
    """
    The dotted path at which ``factor``'s prior model holds ``variable``.

    Used only to separate variables a factor names identically --
    ``_HierarchicalFactor.name_for_variable`` returns the distribution model's
    name for *every* one of its variables. ``distribution_model`` and
    ``drawn_prior``, the two structural segments a ``_HierarchicalFactor``'s
    internal ``Collection`` adds, are dropped, so the parent's ``mean`` reads
    ``HierarchicalFactor0.mean``.

    ``None`` when there is no prior model, no path, or a path with a
    positional segment (``PriorFactor``'s trivial ``Collection(prior)`` holds
    its prior at ``("0",)``) -- a number is no more informative than the
    factor's own name.
    """
    prior_model = getattr(factor, "prior_model", None)
    if prior_model is None:
        return None
    try:
        path = prior_model.path_for_prior(variable)
    except Exception:  # a factor need not implement the model interface
        return None
    if not path:
        return None
    parts = [
        str(part) for part in path if part not in ("distribution_model", "drawn_prior")
    ]
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def label_for_variable(factor, variable, slot: Optional[int] = None) -> Optional[str]:
    """
    The label ``factor`` gives ``variable``, or ``None`` if it no longer holds it.

    ``factor.name_for_variable`` is the only source: an ``AnalysisFactor``
    answers with its dotted path (``AnalysisFactor0.centre``), and the base
    implementation answers with the factor's own name. When that base answer
    comes back -- so every variable of the factor would carry the same label --
    the label is qualified, first by the factor's own prior-model path
    (:func:`_model_path`), then by the variable's declared name, and finally by
    its argument position. Each fallback is declaration-derived, so a label is
    unique within its factor and stable across runs.

    ``variable.label`` is never consulted: it is a global-counter form.

    Parameters
    ----------
    slot
        The variable's position in ``factor.flat_args``. Computed when omitted;
        pass it to avoid re-scanning the arguments.
    """
    name = factor.name_for_variable(variable)
    if name is None:
        return None
    if name != factor.name:
        return name

    path = _model_path(factor, variable)
    if path is not None:
        return f"{name}.{path}"

    from autofit.mapper.prior.abstract import Prior

    if not isinstance(variable, Prior):
        # A plain ``Variable`` carries the name its author gave it.
        return f"{name}.{variable.name}"

    if slot is None:
        slot = _slot_of(factor, variable)
    if slot is None:
        return name
    return f"{name}.arg{slot}"


def _slot_of(factor, variable) -> Optional[int]:
    for index, argument in enumerate(factor.flat_args):
        if argument is variable:
            return index
    return None


def _strip_prefix(label: str, factor_name: str) -> str:
    """
    Drop a factor's own name from the front of a label drawn inside its plate.

    ``AnalysisFactor0.centre`` reads ``centre`` inside the plate -- the plate
    already says which member is which.
    """
    prefix = f"{factor_name}."
    if label.startswith(prefix) and len(label) > len(prefix):
        return label[len(prefix) :]
    return label


def _unique(key: str, used) -> str:
    """
    Guarantee a key is not already taken, deterministically.
    """
    if key not in used:
        return key
    suffix = 2
    while f"{key}-{suffix}" in used:
        suffix += 1
    return f"{key}-{suffix}"


@dataclass(frozen=True)
class FactorNode:
    """
    One factor in the figure.

    Attributes
    ----------
    key
        Declaration-derived identity, e.g. ``factor-3``. A collapsed
        ``PriorFactor`` stub is keyed by the variable it wraps.
    name
        The factor's own name (``AnalysisFactor0``, ``HierarchicalFactor0``).
    kind
        One of :data:`FACTOR_KINDS`.
    member_index
        Position within its plate's members, or ``None`` outside a plate.
    plate_key
        The plate this factor is drawn inside, or ``None``.
    variable_keys
        Keys of the variable nodes this factor is joined to, in label order.
    """

    key: str
    name: str
    kind: str
    member_index: Optional[int]
    plate_key: Optional[str]
    variable_keys: Tuple[str, ...]


@dataclass(frozen=True)
class VariableNode:
    """
    One variable in the figure.

    Attributes
    ----------
    key
        Declaration-derived identity, e.g. ``plate-0/var-0``.
    label
        What the node reads, stripped of its factor prefix inside a plate.
    kind
        One of :data:`VARIABLE_KINDS`.
    plate_key
        The plate the node is drawn inside, or ``None`` for a shared variable.
    count
        How many variables this node stands for: ``N`` inside a plate whose
        ``N`` members each carry their own copy, ``1`` otherwise.
    """

    key: str
    label: str
    kind: str
    plate_key: Optional[str]
    count: int


@dataclass(frozen=True)
class Incidence:
    """
    One edge: a factor is joined to a variable, and calls it ``label``.

    Plate members each keep their own incidence onto the plate's collapsed
    variable node, because the EP state overlay is per (factor, variable) pair
    -- one member reverting a variable is not the plate reverting it.
    """

    factor_key: str
    variable_key: str
    label: str


@dataclass(frozen=True)
class PlateGroup:
    """
    A run of factors with identical structure, drawn once with a ``x N`` badge.

    Attributes
    ----------
    key
        Declaration-derived identity, e.g. ``plate-0``.
    kind
        ``analysis`` or ``hierarchical``.
    count
        Number of members.
    member_keys
        The members' factor keys, in declaration order.
    signature
        What made them a group: the analysis signature, or the shared
        hierarchical factor name.
    representative_key
        The member the plate's labels and kinds are taken from -- the first.
    """

    key: str
    kind: str
    count: int
    member_keys: Tuple[str, ...]
    signature: Tuple[Any, ...]
    representative_key: str


@dataclass(frozen=True)
class EPGraphSpec:
    """
    The whole structure, flat and serialisable.

    ``prior_factor_variable_keys`` names the variables a ``PriorFactor`` wraps
    even when the prior factors themselves are hidden, so the render layer can
    still draw the small stub that says "this variable has a prior" without the
    graph's ``PriorFactor``s doubling its node count.
    """

    factors: Tuple[FactorNode, ...]
    variables: Tuple[VariableNode, ...]
    incidences: Tuple[Incidence, ...]
    plates: Tuple[PlateGroup, ...]
    prior_factor_variable_keys: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """
        A plain ``dict`` of builtins -- the determinism artefact.

        ``json.dumps`` of this is byte-stable for a given factor graph.
        """
        return asdict(self)

    @classmethod
    def from_factor_graph(
        cls, factor_graph, show_prior_factors: bool = False
    ) -> "EPGraphSpec":
        """
        Read a ``DeclarativeFactorGraph`` once, into a spec.

        Parameters
        ----------
        factor_graph
            ``EPOptimiser.factor_graph`` -- the graph the optimiser sweeps and
            the one ``EPHistory`` is keyed by. Never a rebuilt ``.graph``.
        show_prior_factors
            Draw the graph's ``PriorFactor``s as nodes of their own. Off by
            default: there is one per prior, so they treble the node count
            while saying only what a stub on the variable already says.
        """
        return _Reader(factor_graph, show_prior_factors).spec()


class _Reader:
    """
    One pass over ``factor_graph.factors``, turned into an :class:`EPGraphSpec`.
    """

    def __init__(self, factor_graph, show_prior_factors: bool):
        from autofit.graphical.declarative.factor.analysis import AnalysisFactor
        from autofit.graphical.declarative.factor.hierarchical import (
            _HierarchicalFactor,
        )
        from autofit.graphical.declarative.factor.prior import PriorFactor

        self._analysis_cls = AnalysisFactor
        self._hierarchical_cls = _HierarchicalFactor
        self._prior_cls = PriorFactor

        self.show_prior_factors = show_prior_factors

        #: THE read. Everything below works from these tuples.
        self.factors = tuple(factor_graph.factors)

        self.kinds = tuple(map(self._kind, self.factors))
        self.ordered: List[Tuple[Any, ...]] = []
        self.labels: List[Dict[Any, str]] = []
        for factor in self.factors:
            ordered, labels = self._read_variables(factor)
            self.ordered.append(ordered)
            self.labels.append(labels)

        self.incident: Dict[Any, List[int]] = {}
        for index, variables in enumerate(self.ordered):
            for variable in variables:
                self.incident.setdefault(variable, []).append(index)

        self.drawn = {
            factor.drawn_prior
            for factor, kind in zip(self.factors, self.kinds)
            if kind == "hierarchical"
        }
        self.hyper = {
            variable
            for factor, kind, variables in zip(self.factors, self.kinds, self.ordered)
            if kind == "hierarchical"
            for variable in variables
            if variable is not factor.drawn_prior
        }

    # -- reading -----------------------------------------------------------

    def _kind(self, factor) -> str:
        if isinstance(factor, self._analysis_cls):
            return "analysis"
        if isinstance(factor, self._hierarchical_cls):
            return "hierarchical"
        if isinstance(factor, self._prior_cls):
            return "prior"
        return "factor"

    def _read_variables(self, factor):
        """
        A factor's variables, deduplicated, in label order with their labels.

        ``flat_args`` is keyword order, which is not declaration order, so the
        variables are sorted by their (declaration-derived, unique) label --
        stable between runs and independent of how the arguments were bound.
        """
        labels: Dict[Any, str] = {}
        for slot, variable in enumerate(factor.flat_args):
            if variable in labels:
                continue
            label = label_for_variable(factor, variable, slot=slot)
            if label is None:
                # The factor no longer references this variable -- typically a
                # prior replaced by a fixed scalar after the graph was built.
                continue
            labels[variable] = label
        ordered = tuple(sorted(labels, key=lambda variable: labels[variable]))
        return ordered, labels

    # -- plates ------------------------------------------------------------

    def _signature(self, index) -> Optional[Tuple[Any, ...]]:
        from autofit.mapper.prior.abstract import Prior

        factor = self.factors[index]
        kind = self.kinds[index]
        if kind == "hierarchical":
            return "hierarchical", factor.name
        if kind != "analysis":
            return None
        tuples = factor.prior_model.path_instance_tuples_for_class(Prior)
        paths = tuple(".".join(str(part) for part in path) for path, _ in tuples)
        classes = tuple(type(prior).__name__ for _, prior in tuples)
        return "analysis", type(factor).__name__, paths, classes

    def _plates(self):
        groups: Dict[Tuple[Any, ...], List[int]] = {}
        for index in range(len(self.factors)):
            signature = self._signature(index)
            if signature is None:
                continue
            groups.setdefault(signature, []).append(index)

        plates = []
        for signature, members in sorted(groups.items(), key=lambda item: item[1][0]):
            if len(members) < 2:
                continue
            plates.append(
                (
                    PlateGroup(
                        key=f"plate-{members[0]}",
                        kind=signature[0],
                        count=len(members),
                        member_keys=tuple(factor_key(index) for index in members),
                        signature=signature,
                        representative_key=factor_key(members[0]),
                    ),
                    members,
                )
            )
        return plates

    # -- variable nodes ----------------------------------------------------

    def _variable_kind(self, variable) -> str:
        if variable in self.drawn:
            return "drawn"
        if variable in self.hyper:
            return "hyper"
        return "free"

    def _assign(self, plates):
        """
        Decide which variable node every variable belongs to.

        Returns ``(node_key_by_variable, node_by_key)`` where each node carries
        the first declaration index at which it appears, so the nodes can be
        emitted in reading order.
        """
        node_key: Dict[Any, str] = {}
        nodes: Dict[str, VariableNode] = {}
        used = set()

        for plate, members in plates:
            member_set = set(members)
            collapsed: Dict[str, List[Tuple[int, Any, int]]] = {}
            for index in members:
                for slot, variable in enumerate(self.ordered[index]):
                    if variable in node_key:
                        continue
                    inside = [j for j in self.incident[variable] if j in member_set]
                    if len(inside) != 1:
                        # Shared by every member (or by none but this one, via
                        # a factor outside the plate): not the plate's own.
                        continue
                    label = _strip_prefix(
                        self.labels[index][variable], self.factors[index].name
                    )
                    collapsed.setdefault(label, []).append((index, variable, slot))
            for label, entries in collapsed.items():
                key = _unique(f"{plate.key}/var-{entries[0][2]}", used)
                used.add(key)
                nodes[key] = VariableNode(
                    key=key,
                    label=label,
                    kind=self._variable_kind(entries[0][1]),
                    plate_key=plate.key,
                    count=len(entries),
                )
                for _, variable, _ in entries:
                    node_key[variable] = key

        for index, variables in enumerate(self.ordered):
            for slot, variable in enumerate(variables):
                if variable in node_key:
                    continue
                owner = self.incident[variable][0]
                owner_slot = self.ordered[owner].index(variable)
                key = _unique(f"{factor_key(owner)}/var-{owner_slot}", used)
                used.add(key)
                nodes[key] = VariableNode(
                    key=key,
                    label=self.labels[owner][variable],
                    kind=self._variable_kind(variable),
                    plate_key=None,
                    count=1,
                )
                node_key[variable] = key

        return node_key, nodes

    # -- assembly ----------------------------------------------------------

    def spec(self) -> EPGraphSpec:
        plates = self._plates()
        node_key, nodes = self._assign(plates)

        plate_of_factor: Dict[int, Tuple[str, int]] = {}
        for plate, members in plates:
            for position, index in enumerate(members):
                plate_of_factor[index] = (plate.key, position)

        factor_nodes: List[FactorNode] = []
        incidences: List[Incidence] = []
        emitted: List[str] = []
        variable_nodes: List[VariableNode] = []

        prior_groups: Dict[str, List[int]] = {}

        for index, factor in enumerate(self.factors):
            variables = self.ordered[index]
            for variable in variables:
                key = node_key[variable]
                if key not in emitted:
                    emitted.append(key)
                    variable_nodes.append(nodes[key])

            if self.kinds[index] == "prior":
                for variable in variables:
                    prior_groups.setdefault(node_key[variable], []).append(index)
                continue

            plate_key, member_index = plate_of_factor.get(index, (None, None))
            keys = []
            for variable in variables:
                key = node_key[variable]
                if key not in keys:
                    keys.append(key)
                incidences.append(
                    Incidence(
                        factor_key=factor_key(index),
                        variable_key=key,
                        label=self.labels[index][variable],
                    )
                )
            factor_nodes.append(
                FactorNode(
                    key=factor_key(index),
                    name=factor.name,
                    kind=self.kinds[index],
                    member_index=member_index,
                    plate_key=plate_key,
                    variable_keys=tuple(keys),
                )
            )

        if self.show_prior_factors:
            for variable_key, members in prior_groups.items():
                first = members[0]
                stub_key = f"{variable_key}/prior"
                factor_nodes.append(
                    FactorNode(
                        key=stub_key,
                        name=self.factors[first].name,
                        kind="prior",
                        member_index=None,
                        plate_key=nodes[variable_key].plate_key,
                        variable_keys=(variable_key,),
                    )
                )
                incidences.append(
                    Incidence(
                        factor_key=stub_key,
                        variable_key=variable_key,
                        label=nodes[variable_key].label,
                    )
                )

        return EPGraphSpec(
            factors=tuple(factor_nodes),
            variables=tuple(variable_nodes),
            incidences=tuple(incidences),
            plates=tuple(plate for plate, _ in plates),
            prior_factor_variable_keys=tuple(prior_groups),
        )
