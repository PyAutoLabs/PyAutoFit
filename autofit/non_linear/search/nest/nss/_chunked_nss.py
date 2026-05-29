"""Local replica of ``blackjax.ns.nss.as_top_level_api`` with chunked init.

PyAutoFit#1303 added ``make_chunked_update_strategy`` for the per-iteration
MCMC step path inside ``blackjax.ns.from_mcmc.update_with_mcmc_take_last``.
That covers the inner vmap over ``num_delete`` particles, but **not** the
separate hardcoded ``jax.vmap(init_state_fn)`` inside
``blackjax.ns.nss.as_top_level_api``'s ``init_fn``
(``blackjax/ns/nss.py:223-230`` in the handley-lab fork) — and that's where
inversion-heavy lensing cells (PyAutoLens pixelization / Delaunay at HST
scale) OOM on A100 80 GB before the sampling loop even starts.

This module replaces ``_blackjax.nss(...)`` with a local builder so we
control both seams:

- The chunked update_strategy goes through ``make_chunked_update_strategy``
  (PyAutoFit#1303). Behaviour unchanged from that PR.
- The chunked init swaps ``jax.vmap(init_state_fn)`` for
  ``jax.lax.map(init_state_fn, positions, batch_size=chunk_size)`` so peak
  GPU memory during ``algo.init`` becomes ``chunk_size × per_particle_state``
  instead of ``n_live × per_particle_state``.

When ``chunk_size`` is None the builder still uses ``jax.vmap`` and is
bit-identical to upstream. The builder otherwise produces a
``blackjax.SamplingAlgorithm`` with the same shape ``_blackjax.nss(...)``
returns, so ``af.NSS._fit`` is a one-line switch.

See PyAutoFit#1304 for the diagnosis and A100 evidence (jobs 322605 /
322606 OOM at the same byte counts as before #1303 landed, because the
crash is in ``algo.init`` not ``algo.step``).
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional


def build_chunked_nss_algorithm(
    *,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int,
    chunk_size: Optional[int],
):
    """Return a ``blackjax.SamplingAlgorithm`` with chunked init + step paths.

    Replicates the body of ``blackjax.ns.nss.as_top_level_api`` (the
    handley-lab fork) so we can plug in chunked variants of the two
    vmap sites — the inner MCMC step (via the existing
    ``make_chunked_update_strategy``) and the n_live-wide init.

    Parameters
    ----------
    logprior_fn
        Log-prior callable, ``positions -> scalar log-prior``.
    loglikelihood_fn
        Log-likelihood callable, ``positions -> scalar log-L``.
    num_inner_steps
        Number of HRSS steps per particle replacement (matches
        ``af.NSS.num_mcmc_steps``).
    num_delete
        Number of particles replaced per outer iteration (matches
        ``af.NSS.num_delete``).
    chunk_size
        Optional GPU-memory knob. When None, both vmap sites use plain
        ``jax.vmap`` and the result is bit-identical to upstream
        ``blackjax.nss(...)``. When set, peak memory in each site becomes
        ``chunk_size × per_particle_state``.
    """
    # Local imports keep this module cheap to import when ``af.NSS`` is
    # never used (blackjax + jax are optional deps gated by the
    # ``[nss]`` extra; see PyAutoFit pyproject.toml).
    import jax
    from blackjax import SamplingAlgorithm
    from blackjax.ns.adaptive import init as ns_init
    from blackjax.ns.base import init_state_strategy
    from blackjax.ns.nss import build_kernel, update_inner_kernel_params

    from autofit.non_linear.search.nest.nss._chunked_update import (
        make_chunked_update_strategy,
    )

    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )

    kernel = build_kernel(
        init_state_fn,
        num_inner_steps,
        num_delete,
        update_strategy=make_chunked_update_strategy(chunk_size),
    )

    def init_fn(position, rng_key=None):
        # Mirror the upstream signature (rng_key is unused but accepted for
        # API parity with ``blackjax.ns.nss.as_top_level_api``).
        if chunk_size is None:
            init_batcher = jax.vmap(init_state_fn)
        else:
            init_batcher = lambda p: jax.lax.map(
                init_state_fn, p, batch_size=chunk_size
            )
        return ns_init(
            position,
            init_state_fn=init_batcher,
            update_inner_kernel_params_fn=update_inner_kernel_params,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
