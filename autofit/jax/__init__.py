"""Opt-in JAX integration helpers for PyAutoFit.

The library defines pytree ``tree_flatten`` / ``tree_unflatten`` methods on
its model and prior classes but, by design, does NOT register them with JAX
at import time. Eager registration would force ``jax.tree_util`` to load on
every ``import autofit`` and reintroduce the heavy JAX import that the
2025-11 cleanup removed.

Call :func:`enable_pytrees` once before crossing a ``jax.jit`` or
``jax.vmap`` boundary with PyAutoFit objects. The function is a no-op if JAX
is not installed.
"""

from .pytrees import enable_pytrees, register_model

__all__ = ["enable_pytrees", "register_model"]
