"""
Layer 3 of the EP figure -- what is drawn, in words rather than coordinates.

:class:`~autofit.model_figure.ep.spec.EPGraphSpec` says what the graph is and
:class:`~autofit.model_figure.ep.state.EPState` says what the run did to it.
This module decides what a *reader* is shown: which nodes exist after the
plates collapse, what each one is titled and badged, which member of a plate is
exceptional enough to be drawn on its own, and what the legend and footer say.

Nothing here knows about inches or matplotlib. The same presentation is the
model view (``state=None`` -- structure only, drawable before a fit starts) and
the state view (``state`` supplied -- the same structure with the run painted
on).

The rules
---------

**A plate is drawn once.** Every member of a
:class:`~autofit.model_figure.ep.spec.PlateGroup` collapses into one node,
titled by what made the group (the analysis factors' class, or the
hierarchical group's shared name) and badged ``N datasets`` / ``N members``.

**An aggregate is never shown without its exceptions.** With a state, the
plate node carries the group's *modal* status -- and any member that differs
from the rest (a different status, a rejected projection, or an older last
update) is named in the plate's note **and** drawn as a node of its own beside
the plate. A figure that says "3 datasets, working" while one of them has not
updated since sweep 1 is worse than no figure at all.

**A label says what the reader can look up.** Inside a plate the factor prefix
is already stripped by the spec (``centre``, not ``AnalysisFactor0.centre``);
here the same is done for a variable that every member of a plate shares, which
the spec has to leave qualified because it sits outside the plate. A
hierarchical distribution's own parameters keep their qualified names -- there
is exactly one of each, and ``HierarchicalFactor0.mean`` is what
``graph.info`` calls it.

**An edge carries the sharpest true claim.** A stale factor's edges are stale
(the factor never landed an update at all, which subsumes any rejection); an
edge whose (factor, variable) pair the latest projection rejected is reverted;
everything else is a plain incidence. A plate's edges speak for the members the
plate still stands for -- an exceptional member is drawn beside the plate with
edges of its own -- and where several of those members disagree the more
specific claim wins.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from autofit.model_figure.ep.spec import EPGraphSpec, _strip_prefix

__all__ = [
    "EPNode",
    "EPEdge",
    "EPPlate",
    "EPPresentation",
    "build_ep_presentation",
    "FACTOR_NODE_KINDS",
    "VARIABLE_NODE_KINDS",
]

#: Node kinds that are factors. A node's kind is the only thing that says
#: which side of the bipartite graph it is on, so the two sets never overlap.
FACTOR_NODE_KINDS = ("analysis", "hierarchical", "prior", "factor")

#: Node kinds that are variables.
VARIABLE_NODE_KINDS = ("free", "drawn", "hyper")

#: Edge kinds, sharpest claim first -- the order a plate edge merges by.
EDGE_KINDS = ("reverted", "stale", "incidence")

#: Status tie-break when a plate's members are evenly split: the more alarming
#: status is the one the plate reports, because the exceptional members are
#: listed either way and a missed warning is the expensive error.
_STATUS_PRECEDENCE = ("stale", "reverting", "absent", "converged", "working")


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


@dataclass(frozen=True)
class EPNode:
    """
    One node of the figure -- a factor, a plate of factors, or a variable.

    Attributes
    ----------
    key
        The node's identity. A plate node is keyed by its plate, every other
        node by the spec node it stands for.
    title
        What the node reads.
    subtitle
        A second, quieter line -- a plate member says which member it is.
    badges
        Short tags drawn after the title: ``3 datasets``, ``x3``,
        ``2 updates / 4 sweeps``, ``age 3``, ``BAD_PROJECTION``.
    state
        The factor's status from :class:`~autofit.model_figure.ep.state.EPState`
        (``None`` in the model view and on every variable node).
    note
        A full-width line under the node -- the plate's exception list.
    members
        The spec factor keys this node stands for. Non-empty only on a plate
        node.
    kind
        One of :data:`FACTOR_NODE_KINDS` or :data:`VARIABLE_NODE_KINDS`.
    plate_key
        The plate this node belongs to, or ``None``.
    """

    key: str
    title: str
    subtitle: Optional[str] = None
    badges: Tuple[str, ...] = ()
    state: Optional[str] = None
    note: Optional[str] = None
    members: Tuple[str, ...] = ()
    kind: str = "factor"
    plate_key: Optional[str] = None

    @property
    def is_factor(self) -> bool:
        return self.kind in FACTOR_NODE_KINDS

    @property
    def is_variable(self) -> bool:
        return self.kind in VARIABLE_NODE_KINDS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EPEdge:
    """
    One line of the figure: a factor node is joined to a variable node.

    ``kind`` is one of :data:`EDGE_KINDS`. ``label`` is the name the factor
    gives the variable; it is not drawn (the variable node already carries it)
    but it is what a test reads to check *which* pair reverted.
    """

    source_key: str
    target_key: str
    kind: str = "incidence"
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EPPlate:
    """
    A dashed frame and the nodes it encloses.

    ``node_keys`` is what the frame is drawn around: the plate node, the
    variable nodes that live inside it, and any exceptional member expanded
    beside it. An expanded member is still a member -- a frame that cut through
    its box would say it was half in the group.
    """

    key: str
    title: str
    kind: str
    count: int
    badge: str = ""
    member_keys: Tuple[str, ...] = ()
    node_keys: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EPPresentation:
    """Everything the layout engine needs, and nothing about geometry."""

    nodes: Tuple[EPNode, ...] = ()
    edges: Tuple[EPEdge, ...] = ()
    plates: Tuple[EPPlate, ...] = ()
    legend: str = ""
    footer: str = ""
    kind: str = "model"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "plates": [plate.to_dict() for plate in self.plates],
            "legend": self.legend,
            "footer": self.footer,
            "kind": self.kind,
        }

    def node(self, key: str) -> EPNode:
        """The node with ``key`` -- raises rather than returning ``None``."""
        for node in self.nodes:
            if node.key == key:
                return node
        raise KeyError(key)

    def factor_nodes(self) -> Tuple[EPNode, ...]:
        return tuple(node for node in self.nodes if node.is_factor)

    def variable_nodes(self) -> Tuple[EPNode, ...]:
        return tuple(node for node in self.nodes if node.is_variable)


# ----------------------------------------------------------------------------
# the transformation
# ----------------------------------------------------------------------------


@dataclass
class _Group:
    """A plate, its members' states, and what the collapse decided about them."""

    plate: Any
    modal_status: Optional[str] = None
    modal_age: int = 0
    exceptional: Tuple[str, ...] = ()
    reasons: Dict[str, str] = field(default_factory=dict)
    representative_state: Any = None


class _Context:
    """Everything the rules need to see the whole spec at once."""

    def __init__(self, spec: EPGraphSpec, state):
        self.spec = spec
        self.state = state

        self.factor_nodes = {factor.key: factor for factor in spec.factors}
        self.variable_nodes = {variable.key: variable for variable in spec.variables}
        self.plates = {plate.key: plate for plate in spec.plates}

        self.factor_state = (
            {factor.key: factor for factor in state.factors} if state else {}
        )

        #: variable key -> the factor keys incident on it
        self.incident: Dict[str, List[str]] = {}
        for incidence in spec.incidences:
            keys = self.incident.setdefault(incidence.variable_key, [])
            if incidence.factor_key not in keys:
                keys.append(incidence.factor_key)

        self.groups = {
            plate.key: self._group(plate) for plate in spec.plates
        }  # plate key -> _Group

        #: spec factor key -> the node key it collapses into
        self.collapsed: Dict[str, str] = {}
        for factor in spec.factors:
            if factor.plate_key is not None and factor.kind != "prior":
                self.collapsed[factor.key] = factor.plate_key
            else:
                self.collapsed[factor.key] = factor.key

    # -- plates ------------------------------------------------------------

    def _group(self, plate) -> _Group:
        group = _Group(plate=plate)
        if not self.state:
            return group

        states = [self.factor_state[key] for key in plate.member_keys]
        group.modal_status = _mode(
            [state.status for state in states], _STATUS_PRECEDENCE
        )
        group.modal_age = _mode(sorted(state.age for state in states), ())
        group.representative_state = next(
            state for state in states if state.status == group.modal_status
        )

        exceptional = []
        for key, state in zip(plate.member_keys, states):
            reason = self._reason(state, group)
            if reason is not None:
                exceptional.append(key)
                group.reasons[key] = reason
        group.exceptional = tuple(exceptional)
        return group

    @staticmethod
    def _reason(state, group: _Group) -> Optional[str]:
        """
        Why this member is not what the plate says -- or ``None``.

        The three tests are the plan's: a status the rest do not have, a
        projection the latest sweep rejected, or a last update older than the
        rest's.
        """
        if state.status != group.modal_status:
            return state.status
        if state.reverted_variable_keys:
            return "reverting"
        if state.age > group.modal_age:
            return "older"
        return None

    # -- what a node is called ---------------------------------------------

    def plate_title(self, plate) -> str:
        """
        The class the members share (analysis), or the name they share
        (hierarchical) -- both are ``signature[1]``, which is exactly what made
        them a group.
        """
        if len(plate.signature) > 1 and isinstance(plate.signature[1], str):
            return plate.signature[1]
        return self.factor_nodes[plate.representative_key].name

    def plate_badge(self, plate) -> str:
        word = "dataset" if plate.kind == "analysis" else "member"
        return _plural(plate.count, word)

    def note(self, group: _Group) -> Optional[str]:
        """
        ``1 of 3 stale: AnalysisFactor2`` -- the exception list, never omitted.
        """
        if not group.exceptional:
            return None
        reasons = {group.reasons[key] for key in group.exceptional}
        word = reasons.pop() if len(reasons) == 1 else "exceptional"
        names = ", ".join(self.factor_nodes[key].name for key in group.exceptional)
        return f"{len(group.exceptional)} of {group.plate.count} {word}: {names}"

    # -- badges ------------------------------------------------------------

    def factor_badges(self, state) -> Tuple[str, ...]:
        """
        What the run did, in three tags: how much it updated, how long ago, and
        how the last visit ended when it did not end well.
        """
        if state is None:
            return ()
        badges = [f"{_plural(state.updates, 'update')} / {state.sweeps} sweeps"]
        if state.age > 0:
            badges.append(f"age {state.age}")
        if state.last_flag is not None and state.last_flag != "SUCCESS":
            badges.append(state.last_flag)
        return tuple(badges)

    def variable_badges(self, variable) -> Tuple[str, ...]:
        """
        ``hyper`` for a distribution's own parameter, ``shared x N`` for a
        variable every member of a plate holds, ``xN`` for one the plate
        repeats.
        """
        if variable.kind == "hyper":
            return ("hyper",)
        if variable.plate_key is not None and variable.count > 1:
            return (f"x{variable.count}",)
        plate = self._shared_plate(variable)
        if plate is not None:
            return (f"shared x {plate.count}",)
        return ()

    def _shared_plate(self, variable):
        """
        The plate every one of whose members is incident on ``variable``.

        This is the spec's "genuinely shared" case: the variable is drawn
        outside the plate (``plate_key is None``) because it belongs to all of
        them, not to one.
        """
        if variable.plate_key is not None:
            return None
        incident = set(self.incident.get(variable.key, ()))
        for plate in self.spec.plates:
            if incident.issuperset(plate.member_keys):
                return plate
        return None

    def variable_title(self, variable) -> str:
        """
        The label, with a factor prefix stripped when the plate already says
        which factor it came from.

        The spec strips the prefix inside a plate; a *shared* variable sits
        outside one, so the spec has to leave ``AnalysisFactor0.centre``
        qualified -- but a node joined to all three members is not
        ``AnalysisFactor0``'s, and naming one member is worse than naming none.
        Hyper parameters keep their qualified names: there is one of each and
        the name is what ``graph.info`` prints.
        """
        if variable.kind == "hyper":
            return variable.label
        plate = self._shared_plate(variable)
        if plate is None:
            return variable.label
        label = variable.label
        for key in plate.member_keys:
            label = _strip_prefix(label, self.factor_nodes[key].name)
        return label


def _mode(values, precedence):
    """
    The most common value, ties broken by ``precedence`` then by first
    appearance -- deterministic either way.
    """
    values = list(values)
    if not values:
        return None
    counts: Dict[Any, int] = {}
    order: Dict[Any, int] = {}
    for index, value in enumerate(values):
        counts[value] = counts.get(value, 0) + 1
        order.setdefault(value, index)

    def rank(value):
        try:
            return precedence.index(value)
        except (ValueError, AttributeError):
            return order[value]

    return min(counts, key=lambda value: (-counts[value], rank(value)))


def _edge_kind(state, variable_key: str) -> str:
    """
    The claim one (factor, variable) incidence makes.

    ``stale`` outranks ``reverted`` *within a factor* for the same reason it
    outranks ``reverting`` on the node: a factor that never landed an update is
    making the stronger statement about every one of its variables.
    """
    if state is None:
        return "incidence"
    if state.status == "stale":
        return "stale"
    if variable_key in state.reverted_variable_keys:
        return "reverted"
    return "incidence"


def _sharpest(first: str, second: str) -> str:
    """The more specific of two claims merged onto one plate edge."""
    return min((first, second), key=EDGE_KINDS.index)


def build_ep_presentation(spec: EPGraphSpec, state=None) -> EPPresentation:
    """
    Turn a spec (and optionally a state) into the nodes, edges and plates that
    are drawn.

    Parameters
    ----------
    spec
        The structure, from
        :meth:`~autofit.model_figure.ep.spec.EPGraphSpec.from_factor_graph`.
    state
        The overlay, from
        :meth:`~autofit.model_figure.ep.state.EPState.from_history`. ``None``
        gives the **model view**: the same figure with no run painted on it,
        which is what is drawn before the first sweep.

    Returns
    -------
    An :class:`EPPresentation`.
    """
    context = _Context(spec, state)

    nodes: List[EPNode] = []
    plate_nodes: Dict[str, List[str]] = {plate.key: [] for plate in spec.plates}
    expanded: Dict[str, str] = {}  # spec factor key -> its own node key

    # -- factor nodes, in declaration order, plates in their first member's place
    seen_plates = set()
    for factor in spec.factors:
        if factor.kind == "prior":
            continue  # emitted last, beside their variable

        if factor.plate_key is None:
            state_of = context.factor_state.get(factor.key)
            nodes.append(
                EPNode(
                    key=factor.key,
                    title=factor.name,
                    badges=context.factor_badges(state_of),
                    state=state_of.status if state_of else None,
                    kind=factor.kind,
                )
            )
            continue

        group = context.groups[factor.plate_key]
        if factor.plate_key not in seen_plates:
            seen_plates.add(factor.plate_key)
            plate = context.plates[factor.plate_key]
            nodes.append(
                EPNode(
                    key=plate.key,
                    title=context.plate_title(plate),
                    badges=(context.plate_badge(plate),)
                    + context.factor_badges(group.representative_state),
                    state=group.modal_status,
                    note=context.note(group),
                    members=plate.member_keys,
                    kind=plate.kind,
                    plate_key=plate.key,
                )
            )
            plate_nodes[plate.key].append(plate.key)

        if factor.key in group.exceptional:
            # Never an aggregate without its exceptions: this member is drawn
            # on its own, beside the plate, with its own edges.
            state_of = context.factor_state[factor.key]
            expanded[factor.key] = factor.key
            plate_nodes[factor.plate_key].append(factor.key)
            nodes.append(
                EPNode(
                    key=factor.key,
                    title=factor.name,
                    subtitle=(
                        f"member {factor.member_index + 1} "
                        f"of {context.plates[factor.plate_key].count}"
                    ),
                    badges=context.factor_badges(state_of),
                    state=state_of.status,
                    kind=factor.kind,
                    plate_key=factor.plate_key,
                )
            )

    # -- variable nodes
    for variable in spec.variables:
        nodes.append(
            EPNode(
                key=variable.key,
                title=context.variable_title(variable),
                badges=context.variable_badges(variable),
                kind=variable.kind,
                plate_key=variable.plate_key,
            )
        )
        if variable.plate_key is not None:
            plate_nodes[variable.plate_key].append(variable.key)

    # -- prior stubs, last, so they sit beside the variable they wrap
    for factor in spec.factors:
        if factor.kind != "prior":
            continue
        (variable_key,) = factor.variable_keys
        variable = context.variable_nodes[variable_key]
        nodes.append(
            EPNode(
                key=factor.key,
                title="prior",
                badges=(f"x{variable.count}",) if variable.count > 1 else (),
                kind="prior",
                plate_key=factor.plate_key,
            )
        )
        if factor.plate_key is not None:
            plate_nodes[factor.plate_key].append(factor.key)

    # -- edges, one per incidence after the collapse
    #
    # A plate's edge speaks for the members the plate still stands for: an
    # exceptional member is drawn beside the plate with edges of its own, so
    # its claim is already on the figure and does not also colour the
    # aggregate. When *every* member is exceptional the plate would otherwise
    # be left floating, so then they all contribute.
    contributions: Dict[Tuple[str, str], List[Tuple[str, bool, str]]] = {}
    order: List[Tuple[str, str]] = []

    def _contribute(source, variable_key, kind, exceptional, label):
        edge_key = (source, variable_key)
        if edge_key not in contributions:
            contributions[edge_key] = []
            order.append(edge_key)
        contributions[edge_key].append((kind, exceptional, label))

    for incidence in spec.incidences:
        state_of = context.factor_state.get(incidence.factor_key)
        kind = _edge_kind(state_of, incidence.variable_key)
        is_exceptional = incidence.factor_key in expanded

        _contribute(
            context.collapsed[incidence.factor_key],
            incidence.variable_key,
            kind,
            is_exceptional,
            incidence.label,
        )
        if is_exceptional:
            _contribute(
                incidence.factor_key,
                incidence.variable_key,
                kind,
                False,
                incidence.label,
            )

    edges = []
    for source, target in order:
        entries = contributions[(source, target)]
        speaking = [entry for entry in entries if not entry[1]] or entries
        kind = speaking[0][0]
        for other in speaking[1:]:
            kind = _sharpest(kind, other[0])
        edges.append(
            EPEdge(
                source_key=source,
                target_key=target,
                kind=kind,
                label=speaking[0][2],
            )
        )
    edges = tuple(edges)

    plates = tuple(
        EPPlate(
            key=plate.key,
            title=context.plate_title(plate),
            kind=plate.kind,
            count=plate.count,
            badge=context.plate_badge(plate),
            member_keys=plate.member_keys,
            node_keys=tuple(plate_nodes[plate.key]),
        )
        for plate in spec.plates
    )

    return EPPresentation(
        nodes=tuple(nodes),
        edges=edges,
        plates=plates,
        legend=_legend(state),
        footer=_footer(spec, state),
        kind="state" if state else "model",
    )


def _legend(state) -> str:
    lines = ["box = factor", "pill = variable", "dashed frame = repeated structure"]
    if state is not None:
        lines += [
            "grey = stale (never updated)",
            "red dashed = reverted update",
            "green = converged",
        ]
    return "   ".join(lines)


def _footer(spec: EPGraphSpec, state) -> str:
    """
    The counts a reader checks the picture against.

    Prior stubs are excluded from the factor count: a stub stands for a group
    of ``PriorFactor``s and counting it would make the footer disagree with
    ``graph.info``.
    """
    factors = [factor for factor in spec.factors if factor.kind != "prior"]
    parts = [
        _plural(len(factors), "factor"),
        _plural(len(spec.variables), "variable"),
        _plural(len(spec.plates), "plate"),
    ]
    if state is not None:
        keys = {factor.key for factor in factors}
        statuses = [factor.status for factor in state.factors if factor.key in keys]
        for status in ("stale", "reverting", "converged", "working"):
            parts.append(f"{statuses.count(status)} {status}")
        absent = statuses.count("absent")
        if absent:
            parts.append(f"{absent} absent")
        parts.append(f"sweep {state.step}")
    return "   ".join(parts)
