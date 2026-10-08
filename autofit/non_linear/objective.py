"""
The objective factory shared by every non-linear search.

A search evaluates its likelihood through one of four objective *kinds*:

==========================  =============================================================
Kind                        What the callable takes and returns
==========================  =============================================================
``scalar``                  One parameter vector ``(n_dim,)`` -> one figure of merit.
``batched``                 A batch ``(n, n_dim)`` -> ``(n,)`` figures of merit.
``value_and_grad``          One vector -> ``(figure_of_merit, gradient)``.
``batched_value_and_grad``  A batch -> ``((n,), (n, n_dim))``.
==========================  =============================================================

``kind`` is execution only. What the objective returns (log likelihood, log posterior or
-2 x log posterior), its coordinates and the value it returns for an invalid model are
the search's declared ``objective_target`` and ``invalid_value``
(``autofit.non_linear.search.capabilities``), which ``NonLinearSearch.make_fitness``
turns into the ``Fitness`` the objective is built from.

``Fitness.objective(kind)`` is the entry point; this module holds the pieces it is built
from so a search that composes its own coordinate transform on top of the shared
objective (``MultiStartGradient``, ``SMC``) builds it the same way:

- ``jax_objective(func, kind, ...)`` turns one unjitted, composable scalar function into
  the callable for ``kind``: ``jax.vmap`` for the batched kinds, the declared gradient
  mode's ``value_and_grad`` for the grad kinds, and, unless ``compile=False``, one outer
  ``jax.jit`` wrapped in ``log_on_first_compile``. Building it compiles nothing: tracing
  and XLA compilation happen on the first call (lazy jit).
- ``chunk_slices`` / ``evaluate_in_chunks`` sweep a batched callable over a batch in
  fixed-width chunks, padding the ragged final chunk, so the compiled program is keyed
  on one ``(chunk, n_dim)`` shape however many rows are evaluated (lifted from
  ``MultiStartGradient``).

JAX is imported lazily inside the builders, so importing this module (and ``autofit``)
loads no optional backend.
"""
from typing import Callable, List, Optional, Tuple

SCALAR = "scalar"
BATCHED = "batched"
VALUE_AND_GRAD = "value_and_grad"
BATCHED_VALUE_AND_GRAD = "batched_value_and_grad"

#: Every objective kind, in documentation order.
OBJECTIVE_KINDS = (SCALAR, BATCHED, VALUE_AND_GRAD, BATCHED_VALUE_AND_GRAD)

#: The kinds that differentiate the likelihood, and therefore need JAX.
GRAD_KINDS = (VALUE_AND_GRAD, BATCHED_VALUE_AND_GRAD)

#: The kinds that evaluate a batch of parameter vectors at once.
BATCHED_KINDS = (BATCHED, BATCHED_VALUE_AND_GRAD)

_DESCRIPTIONS = {
    SCALAR: "likelihood function",
    BATCHED: "vectorized (vmap) likelihood function",
    VALUE_AND_GRAD: "likelihood function value and gradient",
    BATCHED_VALUE_AND_GRAD: "vectorized (vmap) likelihood function value and gradient",
}


def check_kind(kind: str) -> str:
    """
    Return ``kind`` unchanged if it is an objective kind, else raise ``ValueError``.
    """
    if kind not in OBJECTIVE_KINDS:
        raise ValueError(
            f"Unknown objective kind {kind!r}. Valid kinds are "
            f"{', '.join(repr(k) for k in OBJECTIVE_KINDS)}."
        )
    return kind


def jax_objective(
    func: Callable,
    kind: str,
    gradient_mode: str = "reverse",
    compile: bool = True,
    description: Optional[str] = None,
) -> Callable:
    """
    The JAX callable of objective ``kind`` built on the unjitted scalar ``func``.

    The transforms compose as ``jit(vmap(value_and_grad(func)))``: one outer jit, so a
    batch is a single XLA program cached on its shape (the ``Fitness._vmap`` ordering of
    PyAutoFit#1636), and a jit nested inside a jit would only add a call boundary.

    Parameters
    ----------
    func
        A scalar function of one flat parameter vector, traceable by JAX.
    kind
        One of ``OBJECTIVE_KINDS``.
    gradient_mode
        ``"reverse"`` or ``"forward"`` (``autofit.jax.gradient``); read by the grad kinds
        only.
    compile
        ``False`` returns the transformed callable unjitted (the debugging escape hatch:
        every call then runs eagerly, op by op).
    description
        How the compile notice names the callable; defaults to one per kind.

    Returns
    -------
    The callable. With ``compile=True`` its first call traces, lowers and compiles (and
    logs that it did), and the jitted function is reachable on ``__wrapped__``, whose
    ``_cache_size()`` is the compile-count probe.
    """
    check_kind(kind)

    import jax

    transformed = func

    if kind in GRAD_KINDS:
        from autofit.jax.gradient import value_and_grad_from

        transformed = value_and_grad_from(transformed, gradient_mode)

    if kind in BATCHED_KINDS:
        transformed = jax.vmap(transformed)

    if not compile:
        return transformed

    from autofit.non_linear.jax_compile import log_on_first_compile

    return log_on_first_compile(
        jax.jit(transformed),
        description or _DESCRIPTIONS[kind],
    )


def chunk_slices(n_rows: int, batch_size: int) -> List[Tuple[int, int, int]]:
    """
    The ``(lo, hi, pad)`` chunk bounds of a sweep over ``n_rows`` rows in
    ``batch_size``-wide chunks.

    Rows ``lo:hi`` of the batch form each chunk, padded by ``pad`` repeats of its final
    row so every chunk presents the same ``(batch_size, n_dim)`` shape to the compiled
    function (one XLA compile for the whole sweep). Only the final chunk can be ragged
    (``pad > 0``).
    """
    return [
        (lo, min(lo + batch_size, n_rows), max(0, lo + batch_size - n_rows))
        for lo in range(0, n_rows, batch_size)
    ]


def evaluate_in_chunks(func: Callable, parameters, batch_size: int, xp=None):
    """
    Evaluate the batched callable ``func`` over ``parameters`` in fixed-width chunks.

    The ragged final chunk is padded with repeats of its last row and the padded rows are
    discarded from every output, so the result is numerically identical to one call on
    the whole batch; only the allocation and the number of compiled shapes change.

    Parameters
    ----------
    func
        A batched callable returning one array, or a tuple of arrays, each with a leading
        batch axis.
    parameters
        The ``(n, n_dim)`` batch.
    batch_size
        The chunk width.
    xp
        The array module the chunks are concatenated with (``jax.numpy`` by default).

    Returns
    -------
    The outputs of ``func`` over the whole batch, in the same structure ``func`` returns.
    """
    if xp is None:
        import jax.numpy as xp

    # One array type for every chunk, so the padded final chunk and the unpadded ones
    # present identical inputs to the compiled function.
    parameters = xp.asarray(parameters)

    chunks = []

    for lo, hi, pad in chunk_slices(parameters.shape[0], batch_size):
        chunk = parameters[lo:hi]
        if pad:
            chunk = xp.concatenate([chunk, xp.tile(chunk[-1:], (pad, 1))])

        outputs = func(chunk)
        is_tuple = isinstance(outputs, tuple)
        outputs = outputs if is_tuple else (outputs,)

        if pad:
            outputs = tuple(output[:-pad] for output in outputs)

        chunks.append(outputs)

    combined = tuple(
        xp.concatenate([chunk[index] for chunk in chunks])
        for index in range(len(chunks[0]))
    )

    return combined if is_tuple else combined[0]
