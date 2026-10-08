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
- ``trace_preflight``: one ``jax.eval_shape`` of the search's declared objective kind
  (``"batched"`` for a ``batched`` search, else ``"scalar"``) at the prior-median vector,
  plus one of its gradient kind when the search declares ``gradient == "uses"``. That is
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


def preflight_kinds(search) -> tuple:
    """
    The objective kinds the trace preflight traces for ``search``: its declared
    execution kind, plus the matching gradient kind when it declares
    ``gradient == "uses"``.
    """
    cls = type(search)
    kinds = [BATCHED if cls.batched else SCALAR]
    if cls.gradient == cap.Gradient.USES:
        kinds.append(BATCHED_VALUE_AND_GRAD if cls.batched else VALUE_AND_GRAD)
    return tuple(kinds)


def _preflight_fitness(search, analysis, model):
    """
    A throwaway ``Fitness`` in the search's declared convention, without paths (so no
    resume sanity check evaluates the likelihood) and without quick updates.
    """
    return search.make_fitness(
        analysis=analysis,
        model=model,
        paths=None,
        iterations_per_quick_update=None,
        background_quick_update=False,
        live_visual_update=False,
    )


def trace_preflight(search, analysis, model):
    """
    Trace the search's declared objective kind(s) once with ``jax.eval_shape`` at the
    prior-median vector, raising a ``SearchException`` that names the search and chains
    the tracer error when the likelihood cannot be traced. Then run the numerical probe
    if ``PYAUTO_JAX_PREFLIGHT_PROBE`` asks for it.

    A no-op for a numpy analysis and when ``PYAUTO_JAX_PREFLIGHT=0``.
    """
    if not analysis.is_jax or not _env_flag(PREFLIGHT_ENV, default=True):
        return

    import jax
    import jax.numpy as jnp

    fitness = _preflight_fitness(search, analysis, model)

    vector = jnp.asarray(model.physical_values_from_prior_medians)

    for kind in preflight_kinds(search):
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
