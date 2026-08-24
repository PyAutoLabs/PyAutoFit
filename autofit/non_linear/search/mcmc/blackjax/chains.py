"""
Sampler-agnostic helpers for multi-chain BlackJAX-family searches.

Kept separate from ``blackjax/nuts/search.py`` so a future second BlackJAX
sampler (e.g. HMC) can share the same warm-start / multi-chain / inverse
mass matrix plumbing without importing NUTS-specific code. Following the
existing module convention in this package (see ``_ess_per_param_from`` in
``nuts/search.py``), JAX/blackjax imports live *inside* the functions that
need them so this module -- and anything that merely imports it for the
pure-numpy helpers -- stays importable without the ``[optional]`` extras
installed.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

# The `inverse_mass_matrix` kwarg on `BlackJAXNUTS` accepts any of these.
InverseMassMatrixSpec = Union[None, str, np.ndarray, object]

_VALID_STRING_KINDS = ("diagonal", "dense")


def inverse_mass_matrix_kind_from(spec: InverseMassMatrixSpec) -> str:
    """
    Classify an ``inverse_mass_matrix`` specification into the small set of string kinds recorded in
    ``BlackJAXNUTS.__identifier_fields__`` and persisted onto ``search_internal`` / ``samples_info``.

    Parameters
    ----------
    spec
        ``None``, ``"diagonal"``, ``"dense"``, a numpy array (seed matrix), or a `Result`/`Samples`-like
        object (anything exposing ``.covariance_matrix`` directly, or ``.samples.covariance_matrix``).

    Returns
    -------
    One of ``"none"``, ``"diagonal"``, ``"dense"``, ``"array"``, ``"result"``.
    """
    if spec is None:
        return "none"

    if isinstance(spec, str):
        if spec not in _VALID_STRING_KINDS:
            raise ValueError(
                f"inverse_mass_matrix string must be one of {_VALID_STRING_KINDS}, got {spec!r}"
            )
        return spec

    if isinstance(spec, np.ndarray):
        return "array"

    if hasattr(spec, "covariance_matrix") or hasattr(spec, "samples"):
        return "result"

    raise ValueError(
        "Unsupported inverse_mass_matrix specification: expected None, 'diagonal', 'dense', "
        f"a numpy array, or a Result/Samples object, got {spec!r}"
    )


def resolve_inverse_mass_matrix(
    spec: InverseMassMatrixSpec, n_dim: int
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Resolve an ``inverse_mass_matrix`` specification into the ``(is_mass_matrix_diagonal,
    initial_inverse_mass_matrix)`` pair consumed by ``blackjax.window_adaptation`` /
    ``blackjax.adaptation.staged_adaptation``.

    Parameters
    ----------
    spec
        See `inverse_mass_matrix_kind_from`.
    n_dim
        The number of free model parameters. Used to validate array/covariance shapes.

    Returns
    -------
    is_mass_matrix_diagonal
        Whether ``blackjax.window_adaptation`` should adapt a diagonal (``True``) or dense (``False``)
        inverse mass matrix.
    seed
        ``None`` for a fresh (identity-seeded) adaptation, a 1-D array of shape ``(n_dim,)`` for a diagonal
        seed, or a 2-D array of shape ``(n_dim, n_dim)`` for a dense seed -- passed straight through as
        ``initial_inverse_mass_matrix``.

    Raises
    ------
    ValueError
        If a string is not ``"diagonal"``/``"dense"``, an array has the wrong shape, or a `Result`/`Samples`
        source's covariance is unusable (MLE-only: too few samples, or non-finite/degenerate) -- in which
        case an explicit seed array (e.g. from a Laplace approximation) should be passed instead.
    """
    if spec is None:
        return True, None

    if isinstance(spec, str):
        if spec == "diagonal":
            return True, None
        if spec == "dense":
            return False, None
        raise ValueError(
            f"inverse_mass_matrix string must be one of {_VALID_STRING_KINDS}, got {spec!r}"
        )

    if isinstance(spec, np.ndarray):
        if spec.ndim == 1:
            if spec.shape[0] != n_dim:
                raise ValueError(
                    "inverse_mass_matrix: a 1-D (diagonal) seed array must have shape "
                    f"({n_dim},), got {spec.shape}"
                )
            return True, np.asarray(spec, dtype=float)
        if spec.ndim == 2:
            if spec.shape != (n_dim, n_dim):
                raise ValueError(
                    "inverse_mass_matrix: a 2-D (dense) seed array must have shape "
                    f"({n_dim}, {n_dim}), got {spec.shape}"
                )
            return False, np.asarray(spec, dtype=float)
        raise ValueError(
            f"inverse_mass_matrix: array must be 1-D or 2-D, got ndim={spec.ndim}"
        )

    # Result / Samples: duck-typed, in preference order, to avoid a circular import on
    # `autofit.non_linear.result.Result` / `autofit.non_linear.samples.Samples`.
    if hasattr(spec, "covariance_matrix"):
        samples = spec
    elif hasattr(spec, "samples"):
        samples = spec.samples
    else:
        raise ValueError(
            "Unsupported inverse_mass_matrix specification: expected None, 'diagonal', 'dense', "
            f"a numpy array, or a Result/Samples object, got {spec!r}"
        )

    if samples is None:
        raise ValueError(
            "inverse_mass_matrix: the given Result has no samples (`result.samples` is None) -- "
            "cannot derive a covariance seed from it."
        )

    n_samples = len(samples)
    if n_samples < 2 * n_dim:
        raise ValueError(
            f"inverse_mass_matrix: the given Result/Samples has only {n_samples} samples for "
            f"{n_dim} free parameters (< 2 * n_dim = {2 * n_dim}), so its covariance is unreliable "
            "(this typically means the previous fit was an MLE-style optimizer, not a sampler that "
            "explores the posterior). Pass an explicit inverse_mass_matrix array instead (e.g. from "
            "a Laplace approximation)."
        )

    covariance = np.asarray(samples.covariance_matrix, dtype=float)

    if covariance.shape != (n_dim, n_dim):
        raise ValueError(
            "inverse_mass_matrix: the given Result/Samples covariance_matrix has shape "
            f"{covariance.shape}, expected ({n_dim}, {n_dim})"
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "inverse_mass_matrix: the given Result/Samples covariance_matrix contains non-finite "
            "values -- cannot use it as a warm-start seed. Pass an explicit array instead."
        )

    if np.allclose(covariance, np.eye(n_dim)):
        raise ValueError(
            "inverse_mass_matrix: the given Result/Samples covariance_matrix is the identity matrix, "
            "which `Samples.covariance_matrix` falls back to when it has too few samples to compute a "
            "real covariance (this typically means the previous fit was an MLE-style optimizer). Pass "
            "an explicit inverse_mass_matrix array instead (e.g. from a Laplace approximation)."
        )

    return False, covariance


def stack_initial_positions(parameter_lists) -> np.ndarray:
    """
    Stack the per-chain starting vectors generated by an initializer (a length-``num_chains`` list of
    length-``n_dim`` lists/vectors) into a single ``(num_chains, n_dim)`` array, the shape
    ``BlackJAXNUTS._fit`` vmaps warmup and sampling over.
    """
    return np.asarray(parameter_lists, dtype=float)


def split_chain_diagnostics(positions: np.ndarray) -> np.ndarray:
    """
    Reshape ``(n_samples, n_chains, n_dim)`` sampler-order positions (the persisted ``search_internal``
    layout) into ``(n_chains, n_samples, n_dim)`` -- the ``chain_axis=0``, ``sample_axis=1`` convention
    required by every function in ``blackjax.diagnostics``.
    """
    return np.moveaxis(positions, 0, 1)
