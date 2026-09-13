"""
Layer 2 of the EP figure -- what the run did to the structure.

:class:`EPGraphSpec` says what the graph *is*; :class:`EPState` says what
happened to it. The two are kept apart so that the same figure can be drawn
before a fit starts (structure only) and after every sweep (structure plus
overlay), and so that the overlay can be asserted on without rendering
anything.

Everything here is read from :class:`~autofit.graphical.expectation_propagation.history.EPHistory`
-- the optimiser's own record, keyed by its own factor objects. Nothing is
recomputed from the mean field, and nothing is stored that the history does not
already know.

What a factor's status means
----------------------------

``stale``
    The factor has been visited and has never landed an update. This is the
    complement of ``EPOptimiser``'s own end-of-run stale check
    (``(raised | skipped) - updated``): the reported posterior for this
    factor's variables is still the message it started with.
``reverting``
    The factor's latest projection was rejected for at least one of its
    variables -- ``update_invalid`` put that variable's message back. The
    factor as a whole updated, so no factor-level warning fires, but the
    reverted variable's marginal did not move.
``converged``
    ``EPHistory``'s own convergence test passes for this factor.
``working``
    Updating, nothing rejected, not yet converged.
``absent``
    The factor is in the graph but has no history: the sweep has not reached
    it yet. Every count is zero.

``stale`` outranks ``reverting``, which outranks ``converged``. A factor that
reverts everything on every projection is both stale and reverting and the
stronger claim wins; a factor that reverted a variable on its last projection
is not reported as converged, because that rejection is precisely what the
diagnostic figure exists to show.

Not in scope
------------

Posterior values -- mean, std, precision, KL sparklines -- are a separate
overlay (the epic's prompt items 4-6) and are deliberately absent here.
``model_approx`` is accepted by :meth:`EPState.from_history` so that the
signature does not have to change when they arrive, and is ignored.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from autofit.model_figure.ep.spec import EPGraphSpec, label_for_variable

logger = logging.getLogger(__name__)

__all__ = [
    "FactorState",
    "VariableState",
    "EPState",
]

#: ``FactorState.status`` values, strongest claim first.
FACTOR_STATUSES = ("absent", "stale", "reverting", "converged", "working")


def _reverted_variables(status) -> Tuple[Any, ...]:
    """
    The variables whose projection this update *rejected*.

    ``Status.changed`` is a per-variable mask of whether the projection was
    accepted, and it is written only on the ``BAD_PROJECTION`` branch of
    ``MeanField.update_factor_mean_field``. ``None`` means the update never
    reached the projection, and an all-``True`` mask means a valid projection
    -- neither is a reversion. Attributes are read directly: ``Status.__iter__``
    yields four fields and ``changed`` is deliberately not one of them.
    """
    changed = status.changed
    if changed is None:
        return ()
    return tuple(variable for variable, accepted in changed.items() if not accepted)


def _reverted_names(status) -> Tuple[str, ...]:
    """
    The names of the reverted variables, sorted.

    This is the ``reverted_variables`` column of ``ep_history.csv``
    (``EPDiagnostics.snapshot``) in tuple form, and a test pins it to a real
    CSV written by a real run. It is not lifted out of ``diagnostics.py``: the
    CSV is a published artefact with its own tests and the two must be able to
    fail independently.
    """
    return tuple(sorted(variable.name for variable in _reverted_variables(status)))


@dataclass(frozen=True)
class FactorState:
    """
    What the EP run has done to one factor.

    Attributes
    ----------
    key
        The :class:`~autofit.model_figure.ep.spec.FactorNode` key this
        describes.
    status
        One of :data:`FACTOR_STATUSES`.
    updates
        How many of the factor's visits landed an update.
    sweeps
        How many times the factor has been visited.
    age
        Sweeps since the factor last updated -- ``sweeps`` when it never has.
    reverted_variable_keys
        Variable node keys whose projection the *latest* visit rejected.
    last_flag
        The latest visit's ``StatusFlag`` name, or ``None`` when absent.
    """

    key: str
    status: str
    updates: int
    sweeps: int
    age: int
    reverted_variable_keys: Tuple[str, ...]
    last_flag: Optional[str]


@dataclass(frozen=True)
class VariableState:
    """
    What the EP run has done to one variable.

    ``reverted_by`` names the factors whose latest projection rejected this
    variable. It is per-factor on purpose: one member of a hierarchical group
    reverting a shared variable is not the group reverting it, and only the
    per-(factor, variable) resolution can say which.
    """

    key: str
    reverted_by: Tuple[str, ...]


@dataclass(frozen=True)
class EPState:
    """
    The whole overlay, flat and serialisable.

    ``step`` is the sweep the state describes: the greatest number of history
    entries any one factor has. A factor with fewer has not been visited on
    every sweep.
    """

    factors: Tuple[FactorState, ...]
    variables: Tuple[VariableState, ...]
    step: int

    def to_dict(self) -> Dict[str, Any]:
        """
        A plain ``dict`` of builtins, byte-stable for a given history.
        """
        return asdict(self)

    @classmethod
    def from_history(
        cls,
        spec: EPGraphSpec,
        ep_history,
        factor_by_key: Dict[str, Any],
        model_approx=None,
    ) -> "EPState":
        """
        Read an ``EPHistory`` onto a spec.

        Parameters
        ----------
        spec
            The structure the overlay is painted on.
        ep_history
            ``EPOptimiser.ep_history``. Only ``EPHistory.history`` is read --
            never ``ep_history[factor]``, which *creates* an empty history for
            any factor it is asked about and so would report a factor as
            visited zero times rather than as absent.
        factor_by_key
            Spec factor key to live factor, from
            :func:`~autofit.model_figure.ep.spec.factor_by_key`. A key that is
            missing (a collapsed ``PriorFactor`` stub stands for a group and
            has no single history) is reported ``absent``.
        model_approx
            Accepted and ignored. Posterior values are a later overlay; see
            the module docstring.
        """
        del model_approx

        histories = getattr(ep_history, "history", {})
        variable_key = _variable_key_lookup(spec)

        factor_states = []
        reverted_by: Dict[str, list] = {variable.key: [] for variable in spec.variables}

        for node in spec.factors:
            factor = factor_by_key.get(node.key)
            history = histories.get(factor) if factor is not None else None
            entries = list(history.history) if history is not None else []

            if not entries:
                factor_states.append(
                    FactorState(
                        key=node.key,
                        status="absent",
                        updates=0,
                        sweeps=0,
                        age=0,
                        reverted_variable_keys=(),
                        last_flag=None,
                    )
                )
                continue

            sweeps = len(entries)
            updated_at = [
                index for index, (_, status) in enumerate(entries) if status.updated
            ]
            updates = len(updated_at)
            age = sweeps if not updated_at else sweeps - 1 - updated_at[-1]

            _, latest = entries[-1]
            reverted = tuple(
                key
                for key in (
                    variable_key.get((node.key, label_for_variable(factor, variable)))
                    for variable in _reverted_variables(latest)
                )
                if key is not None
            )
            for key in reverted:
                reverted_by.setdefault(key, []).append(node.key)

            factor_states.append(
                FactorState(
                    key=node.key,
                    status=_status(ep_history, factor, sweeps, updates, reverted),
                    updates=updates,
                    sweeps=sweeps,
                    age=age,
                    reverted_variable_keys=reverted,
                    last_flag=getattr(latest.flag, "name", None),
                )
            )

        return cls(
            factors=tuple(factor_states),
            variables=tuple(
                VariableState(
                    key=variable.key, reverted_by=tuple(reverted_by[variable.key])
                )
                for variable in spec.variables
            ),
            step=max((state.sweeps for state in factor_states), default=0),
        )


def _variable_key_lookup(spec: EPGraphSpec) -> Dict[Tuple[str, str], str]:
    """
    ``(factor key, label)`` to variable node key, from the spec's own edges.

    The spec holds no live variables, so a variable coming back on a ``Status``
    is matched by the label its factor gives it -- the same
    ``name_for_variable`` route the spec built its edges from, which is why
    :func:`~autofit.model_figure.ep.spec.label_for_variable` guarantees labels
    are unique within a factor.
    """
    return {
        (incidence.factor_key, incidence.label): incidence.variable_key
        for incidence in spec.incidences
    }


def _status(ep_history, factor, sweeps: int, updates: int, reverted) -> str:
    if sweeps >= 1 and updates == 0:
        return "stale"
    if reverted:
        return "reverting"
    if _is_converged(ep_history, factor):
        return "converged"
    return "working"


def _is_converged(ep_history, factor) -> bool:
    """
    ``EPHistory``'s own convergence test, which is the only one EP acts on.

    A figure must never kill a fit, so a convergence test that raises on an
    exotic message family reports "not converged" and says so in the log.
    """
    try:
        return bool(ep_history.is_converged(factor))
    except Exception as e:
        logger.debug(f"EP figure: convergence test failed for {factor}: {e}")
        return False
