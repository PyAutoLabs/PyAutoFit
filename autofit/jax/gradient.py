"""
Analysis-declared gradient mode.

An ``af.Analysis`` declares how its likelihood is best differentiated through the class attribute
``gradient_mode``: ``"reverse"`` (the default, ``jax.value_and_grad``) or ``"forward"``
(``jax.jacfwd`` over the flat parameter vector). A gradient search may override the declaration
(e.g. ``af.MultiStartAdam(gradient_mode="reverse")``).

Both modes compute the same gradient; they differ only in cost. Reverse mode costs a small constant
multiple of one likelihood evaluation whatever the number of parameters, so it is the right default.
Forward mode costs one tangent per free parameter, but wins when the likelihood itself contains an
inner forward-mode derivative (reverse mode must then run reverse-over-forward through it) and the
parameter count is modest -- e.g. the source-plane point-source likelihood, whose lensing Hessian is
``jax.jacfwd``-built (autolens_profiling #327/#331: 2-4.5x faster, up to 8x faster to compile).

JAX is imported lazily inside the builders so that ``autofit`` imports without JAX.
"""

from typing import Callable, Optional

GRADIENT_MODES = ("reverse", "forward")


def _check_mode(mode: str, source: str) -> str:
    if mode not in GRADIENT_MODES:
        raise ValueError(
            f"Unknown gradient_mode {mode!r} ({source}). Valid modes are "
            f"{', '.join(repr(m) for m in GRADIENT_MODES)}."
        )
    return mode


def validate_gradient_mode(mode: Optional[str]) -> Optional[str]:
    """
    Return ``mode`` unchanged if it is ``None`` or a valid mode, else raise ``ValueError``.

    Used by constructors (``Fitness``, ``MultiStartGradient``) that accept an optional override,
    so a mistyped mode fails at construction rather than at the first gradient evaluation.
    """
    if mode is None:
        return None
    return _check_mode(mode, "override")


def resolve_gradient_mode(analysis, override: Optional[str] = None) -> str:
    """
    The gradient mode to use for ``analysis``.

    ``override`` (a search- or ``Fitness``-level setting) wins when it is not ``None``; otherwise
    the analysis's ``gradient_mode`` class attribute is used, defaulting to ``"reverse"`` for
    objects that do not declare one. An unknown value from either source raises ``ValueError``.
    """
    if override is not None:
        return _check_mode(override, "override")

    mode = getattr(analysis, "gradient_mode", "reverse")
    return _check_mode(mode, f"declared by {type(analysis).__name__}")


def _forward_value_and_grad(func: Callable) -> Callable:
    import jax
    import jax.numpy as jnp

    def value_twice(vector):
        value = func(vector)
        if jnp.ndim(value) != 0:
            raise ValueError(
                "gradient_mode='forward' requires a scalar objective; the function returned "
                f"an array of shape {jnp.shape(value)}."
            )
        return value, value

    jac = jax.jacfwd(value_twice, has_aux=True)

    def value_and_grad(vector):
        grad, value = jac(vector)
        return value, grad

    return value_and_grad


def value_and_grad_from(func: Callable, mode: str) -> Callable:
    """
    A callable returning ``(value, grad)`` of the scalar ``func`` of a flat parameter vector,
    with the same contract as ``jax.value_and_grad(func)``.

    - ``"reverse"``: ``jax.value_and_grad(func)``;
    - ``"forward"``: ``jax.jacfwd`` over the vector, with the value carried as auxiliary output so
      the likelihood is traced once.
    """
    _check_mode(mode, "value_and_grad_from")

    if mode == "reverse":
        import jax

        return jax.value_and_grad(func)

    return _forward_value_and_grad(func)


def grad_from(func: Callable, mode: str) -> Callable:
    """
    The gradient-only counterpart of ``value_and_grad_from``: ``jax.grad(func)`` in reverse mode,
    the ``jax.jacfwd`` gradient of the scalar ``func`` in forward mode.
    """
    _check_mode(mode, "grad_from")

    if mode == "reverse":
        import jax

        return jax.grad(func)

    value_and_grad = _forward_value_and_grad(func)

    def grad(vector):
        return value_and_grad(vector)[1]

    return grad
