"""
Static capability declarations of the non-linear searches.

Every search declares what it is as plain class attributes, defined on
``NonLinearSearch`` (the defaults) and overridden by each concrete search:

========================  ==============================================================
Attribute                 Meaning
========================  ==============================================================
``jax_use``               ``none`` (the backend never needs JAX), ``optional`` (it has a
                          jitted or batched fast path when the analysis is JAX) or
                          ``required`` (the backend cannot run without a traceable JAX
                          likelihood).
``gradient``              ``none`` or ``uses``: whether the backend differentiates the
                          likelihood.
``batched``               Whether the backend evaluates a vectorised (``jax.vmap``)
                          likelihood over many points at once.
``honours_gradient_mode`` Whether the backend reads ``Analysis.gradient_mode``.
``posterior_kind``        ``chain`` (equally weighted MCMC draws), ``weighted`` (weighted
                          samples, e.g. nested sampling) or ``point`` (a point estimate).
``produces_evidence``     Whether the backend estimates the Bayesian evidence.
``resumable``             Whether an interrupted run resumes from its own checkpoint.
``warm_start``            ``provider`` (starts cold and can seed another search),
                          ``consumer`` (wants a warm start) or ``neutral``.
``install_extra``         The pip extra that installs the backend (``""`` = base).
``upstream_url``          The backend project's home page.
``citation_keys``         Keys into ``files/citations.bib``.
``status``                ``stable``, ``experimental`` or ``archived``.
``test_mode_budget``      The reduced budget ``PYAUTO_TEST_MODE=1`` applies.
``objective_target``      What the backend's objective returns and in which coordinates.
``invalid_value``         The sentinel the objective returns for an invalid model.
========================  ==============================================================

``invalid_value`` records what the backend actually sees and is not yet normalised. For
the ``neg2_log_posterior`` minimizers (``BFGS``, ``LBFGS`` and the ``MultiStart*``
family) it is conditional: a NaN or infinite likelihood is replaced by ``-inf`` and then
multiplied by -2 into ``+inf`` (the declared value), while a ``FitException`` early
return and a failed traced assertion return ``-inf``.

These are **static** capabilities, a property of the class. The capability a given run
actually has can be narrower: ``SMC`` estimates the evidence only when it starts from a
prior-sampling initializer (a warm start skips the tempering from the prior). Effective
run capabilities are not declared here.

None of these attributes is ever in a search's ``__identifier_fields__``, so declaring
or changing one never changes an output path. The declarative registry
(``autofit.non_linear.search.registry``) mirrors them for every public search, and a
test asserts the two agree.
"""

import enum
import math
from typing import Any, Dict, NamedTuple


class JaxUse(str, enum.Enum):
    """
    How a search's backend relates to JAX.
    """

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"

    def __str__(self):
        return self.value


class Gradient(str, enum.Enum):
    """
    Whether a search's backend differentiates the likelihood.
    """

    NONE = "none"
    USES = "uses"

    def __str__(self):
        return self.value


class PosteriorKind(str, enum.Enum):
    """
    The representation of a search's posterior samples.
    """

    CHAIN = "chain"
    WEIGHTED = "weighted"
    POINT = "point"

    def __str__(self):
        return self.value


class WarmStart(str, enum.Enum):
    """
    A search's role when searches are chained.
    """

    PROVIDER = "provider"
    CONSUMER = "consumer"
    NEUTRAL = "neutral"

    def __str__(self):
        return self.value


class Status(str, enum.Enum):
    """
    The maturity of a search.
    """

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"

    def __str__(self):
        return self.value


class ObjectiveQuantity(str, enum.Enum):
    """
    The statistical quantity a search's objective returns.
    """

    LOG_LIKELIHOOD = "log_likelihood"
    LOG_POSTERIOR = "log_posterior"
    NEG2_LOG_POSTERIOR = "neg2_log_posterior"

    def __str__(self):
        return self.value


class CoordinateSpace(str, enum.Enum):
    """
    The coordinates a search's backend proposes points in. ``Fitness`` always receives
    physical parameters; a ``unit_cube`` backend maps through the prior transform first.
    """

    PHYSICAL = "physical"
    UNIT_CUBE = "unit_cube"

    def __str__(self):
        return self.value


class ObjectiveTarget(NamedTuple):
    """
    What a search's objective returns (``quantity``) and the coordinates its backend
    proposes points in (``space``).
    """

    quantity: ObjectiveQuantity
    space: CoordinateSpace

    def to_dict(self) -> Dict[str, str]:
        return {"quantity": str(self.quantity), "space": str(self.space)}


CAPABILITY_ATTRIBUTES = (
    "jax_use",
    "gradient",
    "batched",
    "honours_gradient_mode",
    "posterior_kind",
    "produces_evidence",
    "resumable",
    "warm_start",
    "install_extra",
    "upstream_url",
    "citation_keys",
    "status",
    "test_mode_budget",
    "objective_target",
    "invalid_value",
)
"""
The names of the static capability attributes, in manifest order.
"""


def invalid_value_to_str(value: float) -> str:
    """
    The JSON-safe spelling of an ``invalid_value`` sentinel (JSON has no ``-inf``).
    """
    value = float(value)
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return format(value, "g")


def capability_value(name: str, value: Any) -> Any:
    """
    Normalise one capability attribute value into plain JSON data.
    """
    if name == "objective_target":
        return value.to_dict() if value is not None else None
    if name == "invalid_value":
        return invalid_value_to_str(value)
    if name == "citation_keys":
        return list(value)
    if name == "test_mode_budget":
        return dict(value)
    if isinstance(value, enum.Enum):
        return value.value
    return value


def capabilities_from(cls) -> Dict[str, Any]:
    """
    The static capabilities of a search class, as plain JSON data in manifest order.
    """
    return {
        name: capability_value(name, getattr(cls, name))
        for name in CAPABILITY_ATTRIBUTES
    }


JAX_REQUIRED_MESSAGE = (
    "{search} requires a JAX analysis: its backend is JAX-native (jax_use='required') "
    "and cannot run on a numpy likelihood, but the analysis passed to fit has "
    "is_jax=False. Construct the analysis with use_jax=True (and do not set "
    "PYAUTO_DISABLE_JAX=1 outside test mode), or choose a search whose jax_use is "
    "'none' or 'optional' (see the capability matrix in the PyAutoFit docs)."
)
"""
The one message raised when a ``jax_use='required'`` search is given a numpy analysis.
"""


def check_jax_required(search, analysis):
    """
    Raise ``SearchException`` when ``search`` requires JAX and ``analysis`` is not JAX.

    This is the single fail-fast gate every ``jax_use='required'`` search goes through:
    ``NonLinearSearch.start_resume_fit`` calls it after the test-mode bypass return, and
    the JAX-native searches call it again at the top of ``_fit`` for direct callers.
    """
    from autofit import exc

    if search.jax_use == JaxUse.REQUIRED and not analysis.is_jax:
        raise exc.SearchException(
            JAX_REQUIRED_MESSAGE.format(search=type(search).__name__)
        )


def capability_summary_from(search) -> str:
    """
    A short human-readable block naming a search and its main capabilities, written as
    the header of ``model.info`` and as a section of ``search.summary``.
    """
    cls = type(search)
    target = cls.objective_target
    target_text = (
        f"{target.quantity} ({target.space} coordinates)"
        if target is not None
        else "undeclared"
    )
    return (
        f"Search: {cls.__name__} ({cls.__module__}.{cls.__qualname__})\n"
        f"JAX use = {cls.jax_use}, gradient = {cls.gradient}, batched = {cls.batched}\n"
        f"Posterior kind = {cls.posterior_kind}, produces evidence = "
        f"{cls.produces_evidence}, resumable = {cls.resumable}\n"
        f"Objective = {target_text}, invalid value = "
        f"{invalid_value_to_str(cls.invalid_value)}\n"
        f"Status = {cls.status}\n"
    )


def run_summary_from(search) -> str:
    """
    The run-time lines ``search.summary`` adds below the static capabilities: the
    requested and effective worker counts of the last fit (the JAX fork rule of
    ``autofit.non_linear.parallel.pool``). Empty before a fit has resolved its pools.
    """
    return getattr(search, "_parallel_summary", None) or ""
