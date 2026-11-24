"""
The JAX checks ``NonLinearSearch.fit`` runs before a search's backend builds any state
(search-extensibility phase A3b; survey 02 §5.3, decisions D2 and D8).

They run in ``NonLinearSearch.start_resume_fit`` after the test-mode bypass return (so a
``PYAUTO_TEST_MODE>=2`` smoke run, which never runs a backend, is unaffected) and after
the ``jax_use='required'`` gate (``capabilities.check_jax_required``), and only when the
analysis is JAX (``Analysis.is_jax``):

- ``check_x64``: JAX defaults to 32-bit floats, and x64 is switched on only through an
  environment variable read at ``autonerves`` import, so importing ``jax`` first runs
  every likelihood in fp32 without a word. Warn once per process; raise for a search
  that declares ``requires_fp64 = True`` (none does today).
- ``trace_preflight``: one ``jax.eval_shape`` of the objective kind the search actually
  runs (``"batched"`` for a batched execution, else ``"scalar"``) at the prior-median
  vector, plus one of its gradient kind (in the gradient mode the search resolves) when
  the search declares ``gradient == "uses"``. The kinds and modes come from the search's
  ``ExecutionSettings`` (``NonLinearSearch.execution_settings``), the same resolved object
  the backend builds its ``Fitness`` from, so an eager (``compile=False``) scalar
  execution is not traced at all and a search's ``gradient_mode`` override is honoured. That is
  one trace and no compile or execution, so it costs seconds at most where the backend's
  own first call would compile for minutes, and a non-traceable likelihood (an
  ``np.asarray`` or ``float()`` on a parameter-dependent value, a Python ``if`` on an
  array) fails here, naming the search and chaining the original tracer error, instead
  of deep inside the backend after its state exists. It is a **trace check, not a
  validity certificate**: a likelihood that traces can still return NaN.
- the optional numerical probe (``PYAUTO_JAX_PREFLIGHT_PROBE=1``, off by default):
  evaluate the scalar objective once, eagerly, at the prior medians and warn when it
  returns the search's invalid value. It executes the likelihood, so it is never on by
  default.

``PYAUTO_JAX_PREFLIGHT=0`` skips the trace preflight (the escape hatch for a likelihood
deliberately run eagerly, e.g. ``compile=False`` debugging of a function that
does not trace).

JAX is imported inside the functions, so importing this module loads no optional
backend.
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

from autofit import exc
from autofit.non_linear.objective import (
    BATCHED,
    BATCHED_VALUE_AND_GRAD,
    SCALAR,
    VALUE_AND_GRAD,
)
from autofit.non_linear.search import capabilities as cap

logger = logging.getLogger(__name__)

PREFLIGHT_ENV = "PYAUTO_JAX_PREFLIGHT"
"""
Set to ``0`` (or ``false`` / ``no`` / ``off``) to skip the trace preflight.
"""

PROBE_ENV = "PYAUTO_JAX_PREFLIGHT_PROBE"
"""
Set to ``1`` (or ``true`` / ``yes`` / ``on``) to run the numerical probe.
"""

X64_WARNING = (
    "JAX is running in 32-bit precision (jax_enable_x64 is False), so the likelihood "
    "of this JAX analysis is evaluated in float32. PyAutoFit enables x64 through an "
    "environment variable read when autonerves is imported; importing jax before "
    "autofit (or setting JAX_ENABLE_X64=0) leaves it off. Call "
    "jax.config.update('jax_enable_x64', True) before building the analysis, or import "
    "autofit before jax, to fit in float64."
)
"""
The warning ``check_x64`` emits once per process.
"""

X64_REQUIRED_MESSAGE = (
    "{search} requires float64 (requires_fp64=True), but JAX is running in 32-bit "
    "precision (jax_enable_x64 is False). Call jax.config.update('jax_enable_x64', "
    "True) before building the analysis, or import autofit before jax."
)
"""
The error ``check_x64`` raises for a search that declares ``requires_fp64``.
"""

PREFLIGHT_MESSAGE = (
    "{search} cannot run on this JAX analysis: tracing its {kind!r} objective at the "
    "prior-median parameters failed with {error_type}: {error}\n\n"
    "This is the trace preflight, run before the search builds any state: the "
    "likelihood of {analysis} (use_jax=True) must be traceable by jax.jit. Replace numpy "
    "calls on parameter-dependent values (np.asarray, np.array, float(), a Python if on "
    "an array) with jax.numpy / xp equivalents, or construct the analysis with "
    "use_jax=False and choose a search whose jax_use is not 'required'. The original "
    "error is chained above. Set {env}=0 to skip this check."
)
"""
The error ``trace_preflight`` raises, chained to the tracer error.
"""

_x64_warning_emitted = False


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def check_x64(search, analysis):
    """
    Warn once per process when a JAX analysis runs with ``jax_enable_x64`` off, or
    raise ``SearchException`` when ``search`` declares ``requires_fp64``.
    """
    global _x64_warning_emitted

    if not analysis.is_jax:
        return

    import jax

    if jax.config.jax_enable_x64:
        return

    if type(search).requires_fp64:
        raise exc.SearchException(
            X64_REQUIRED_MESSAGE.format(search=type(search).__name__)
        )

    if _x64_warning_emitted:
        return

    _x64_warning_emitted = True
    logger.warning(X64_WARNING)
    warnings.warn(X64_WARNING, UserWarning, stacklevel=3)


@dataclass(frozen=True)
class ExecutionSettings:
    """
    How a search will execute its objective on one analysis, resolved once from the
    search's capability declarations, its runtime options and the analysis: the one
    object the trace preflight validates and the backend builds its ``Fitness`` from
    (``NonLinearSearch.execution_settings``).

    Attributes
    ----------
    batched
        Whether the backend evaluates the batched (``jax.vmap``) objective. A search may
        resolve this at runtime (``Nautilus(use_jax_vmap=False)`` runs the scalar one).
    compile
        Whether the scalar objective is jitted. ``False`` is eager execution (the
        deprecated ``use_jax_jit=False``), which never traces the scalar likelihood.
    uses_gradient
        Whether the backend differentiates the objective (``gradient == "uses"``).
    gradient_mode
        The resolved AD mode: the search's ``gradient_mode`` override when it honours
        one, else the analysis's declaration (``autofit.jax.gradient``).
    fitness_overrides
        The ``Fitness`` keyword arguments the backend passes to ``make_fitness``,
        including the resolved ``compile`` / ``batched`` / ``gradient_mode`` where they
        differ from the defaults.
    """

    batched: bool
    compile: bool
    uses_gradient: bool
    gradient_mode: str
    fitness_overrides: dict = field(default_factory=dict)

    @property
    def kinds(self) -> tuple:
        """
        The objective kinds the backend evaluates: its execution kind, plus the matching
        gradient kind when it differentiates.
        """
        kinds = [BATCHED if self.batched else SCALAR]
        if self.uses_gradient:
            kinds.append(BATCHED_VALUE_AND_GRAD if self.batched else VALUE_AND_GRAD)
        return tuple(kinds)

    @classmethod
    def resolve(cls, search, analysis) -> "ExecutionSettings":
        """
        Resolve ``search``'s execution on ``analysis``: its ``fitness_overrides``
        (runtime batching, eager execution, batch size, ...) over its class
        declarations, and its ``gradient_mode`` override (when the search honours one)
        over the analysis's declared mode.
        """
        from autofit.jax.gradient import resolve_gradient_mode

        search_cls = type(search)
        overrides = dict(search.fitness_overrides(analysis))

        override_mode: Optional[str] = overrides.get("gradient_mode")
        if override_mode is None and search_cls.honours_gradient_mode:
            override_mode = getattr(search, "gradient_mode", None)
            if override_mode is not None:
                overrides["gradient_mode"] = override_mode

        return cls(
            batched=bool(overrides.get("batched", search_cls.batched)),
            compile=bool(overrides.get("compile", True)),
            uses_gradient=search_cls.gradient == cap.Gradient.USES,
            gradient_mode=resolve_gradient_mode(analysis, override=override_mode),
            fitness_overrides=overrides,
        )


def preflight_kinds(search, analysis=None) -> tuple:
    """
    The objective kinds the trace preflight traces for ``search``: the kinds of its
    resolved ``ExecutionSettings`` on ``analysis``, or of its class declarations when
    no analysis is given.
    """
    if analysis is not None:
        return search.execution_settings(analysis).kinds

    cls = type(search)
    return ExecutionSettings(
        batched=bool(cls.batched),
        compile=True,
        uses_gradient=cls.gradient == cap.Gradient.USES,
        gradient_mode="reverse",
    ).kinds


def _preflight_fitness(search, analysis, model, settings: ExecutionSettings):
    """
    A throwaway ``Fitness`` built from the search's resolved execution settings, without
    paths (so no resume sanity check evaluates the likelihood) and without quick
    updates.
    """
    return search.make_fitness(
        analysis=analysis,
        model=model,
        **{
            **settings.fitness_overrides,
            "paths": None,
            "iterations_per_quick_update": None,
            "background_quick_update": False,
            "live_visual_update": False,
        },
    )


def trace_preflight(search, analysis, model):
    """
    Trace the objective kind(s) the search resolves for ``analysis``
    (``NonLinearSearch.execution_settings``) once with ``jax.eval_shape`` at the
    prior-median vector, raising a ``SearchException`` that names the search and chains
    the tracer error when the likelihood cannot be traced. Then run the numerical probe
    if ``PYAUTO_JAX_PREFLIGHT_PROBE`` asks for it.

    An eager scalar execution (``compile=False``) is never traced by the backend, so its
    scalar kind is not traced here either; a gradient kind is, since differentiation
    traces even an eager objective.

    A no-op for a numpy analysis and when ``PYAUTO_JAX_PREFLIGHT=0``.
    """
    if not analysis.is_jax or not _env_flag(PREFLIGHT_ENV, default=True):
        return

    import jax
    import jax.numpy as jnp

    settings = search.execution_settings(analysis)

    fitness = _preflight_fitness(search, analysis, model, settings)

    vector = jnp.asarray(model.physical_values_from_prior_medians)

    for kind in settings.kinds:
        if kind == SCALAR and not settings.compile:
            continue
        objective = fitness.objective(kind, compile=False)
        argument = vector[None, :] if kind in (BATCHED, BATCHED_VALUE_AND_GRAD) else vector
        try:
            jax.eval_shape(objective, argument)
        except Exception as error:
            raise exc.SearchException(
                PREFLIGHT_MESSAGE.format(
                    search=type(search).__name__,
                    kind=kind,
                    error_type=type(error).__name__,
                    error=error,
                    analysis=type(analysis).__name__,
                    env=PREFLIGHT_ENV,
                )
            ) from error

    if _env_flag(PROBE_ENV, default=False):
        numerical_probe(search, fitness, vector)


def numerical_probe(search, fitness, vector) -> float:
    """
    Evaluate the scalar objective once, eagerly, at ``vector`` and warn when it returns
    the search's invalid value (a NaN, infinite, over-ceiling or assertion-violating
    likelihood at the prior medians). Executes the likelihood; off by default.

    Returns
    -------
    The figure of merit at ``vector``.
    """
    import numpy as np

    value = float(np.asarray(fitness.objective(SCALAR, compile=False)(vector)))

    invalid = float(type(search).invalid_value)

    if not np.isfinite(value) or value == invalid:
        logger.warning(
            f"{type(search).__name__} numerical probe: the objective at the prior-median "
            f"parameters is {value} (the search's invalid value is {invalid}). The "
            f"likelihood traces, but returns an invalid value there; check for NaNs, an "
            f"assertion that excludes the prior medians, or a log likelihood over the "
            f"configured ceiling."
        )

    return value


def check_runtime_gates(search, analysis, model):
    """
    The fail-fast gates ``NonLinearSearch.start_resume_fit`` runs after the test-mode
    bypass return and before any backend state exists:

    - whole-graph backend and compute-setup agreement
      (``FactorGraphModel.check_backend_agreement`` / ``check_setup_agreement``),
      through the analysis wrappers that forward their wrapped analysis's likelihood
      (``ModelAnalysis``, ``AnalysisFactor``), so a wrapped graph is still checked as
      one whole graph;
    - the REQUIRED gate: a ``jax_use='required'`` search given a numpy analysis raises
      the one shared ``SearchException``;
    - the x64 check (warn once, raise for ``requires_fp64``) and one ``jax.eval_shape``
      trace of the declared objective kind(s), so a non-traceable likelihood fails
      here, naming the search, rather than inside the backend.
    """
    from autofit.graphical.declarative.collection import FactorGraphModel
    from autofit.graphical.declarative.factor.analysis import AnalysisFactor
    from autofit.non_linear.analysis.model_analysis import ModelAnalysis

    inner = analysis
    while isinstance(inner, (ModelAnalysis, AnalysisFactor)) and not isinstance(
        inner, FactorGraphModel
    ):
        inner = inner.__dict__.get("analysis")
    if isinstance(inner, FactorGraphModel):
        inner.check_backend_agreement()
        inner.check_setup_agreement()

    cap.check_jax_required(search, analysis)

    check_x64(search, analysis)
    trace_preflight(search, analysis, model)
