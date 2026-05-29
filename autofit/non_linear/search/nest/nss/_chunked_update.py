"""Chunked replacement for ``blackjax.ns.from_mcmc.update_with_mcmc_take_last``.

Upstream blackjax fans out ``num_delete`` particles through ``jax.vmap``
with no chunking:

    sample_keys = jax.random.split(sample_key, num_delete)
    return jax.vmap(mcmc_kernel)(sample_keys, start_state)

On inversion-heavy likelihoods (e.g. PyAutoLens pixelization / Delaunay
source models) the per-particle MCMC state plus scatter temp buffers
exceeds A100 80 GB even at ``num_delete=16``. See PyAutoFit#1301 for the
full diagnosis and per-cell evidence from ``autolens_profiling``.

``chunked_update_with_mcmc_take_last`` accepts a ``chunk_size`` kwarg and
swaps the vmap for ``jax.lax.map(..., batch_size=chunk_size)`` when
``chunk_size < num_delete`` — same vmap parallelism within a chunk, sequential
chunks across. Peak memory becomes ``chunk_size × per_particle_state``
instead of ``num_delete × per_particle_state``.

When ``chunk_size`` is None or ``>= num_delete`` the function is
bit-identical to upstream.

``blackjax.nss(...)`` already exposes ``update_strategy`` as a kwarg
(see ``blackjax/ns/nss.py:157``), so ``af.NSS._fit`` only needs to pass
this builder to opt in:

    algo = _blackjax.nss(
        ...,
        update_strategy=make_chunked_update_strategy(chunk_size),
    )
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional


def make_chunked_update_strategy(chunk_size: Optional[int]) -> Callable:
    """Return an ``update_strategy`` callable for ``blackjax.nss(...)``.

    Signature matches ``blackjax.ns.from_mcmc.update_with_mcmc_take_last``
    so it can be passed through the ``update_strategy=`` kwarg unmodified.

    Parameters
    ----------
    chunk_size
        Number of particles to vmap-batch per chunk. When None or
        ``>= num_delete`` the chunked path is skipped and the function
        falls through to a plain ``jax.vmap`` (matching upstream
        behaviour bit-for-bit).
    """

    def chunked_update_with_mcmc_take_last(
        constrained_mcmc_step_fn,
        num_mcmc_steps,
        num_delete,
    ):
        """Drop-in for ``blackjax.ns.from_mcmc.update_with_mcmc_take_last``.

        Identical to upstream except the inner
        ``jax.vmap(mcmc_kernel)(sample_keys, start_state)`` is replaced
        with ``jax.lax.map(..., batch_size=chunk_size)`` when
        ``chunk_size`` is set and smaller than ``num_delete``.
        """
        import jax
        import jax.numpy as jnp

        def update_function(rng_key, state, loglikelihood_0, **step_parameters):
            choice_key, sample_key = jax.random.split(rng_key)
            particles = state.particles

            # Select start particles from survivors (verbatim from upstream).
            weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
            weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
            start_idx = jax.random.choice(
                choice_key,
                len(weights),
                shape=(num_delete,),
                p=weights / weights.sum(),
                replace=True,
            )
            start_state = jax.tree.map(lambda x: x[start_idx], particles)

            shared_mcmc_step_fn = partial(
                constrained_mcmc_step_fn,
                loglikelihood_0=loglikelihood_0,
                **step_parameters,
            )

            def mcmc_kernel(rng_key, state):
                keys = jax.random.split(rng_key, num_mcmc_steps)

                def body_fn(state, rng_key):
                    new_state, info = shared_mcmc_step_fn(rng_key, state)
                    return new_state, info

                final_state, infos = jax.lax.scan(body_fn, state, keys)
                return final_state, infos

            sample_keys = jax.random.split(sample_key, num_delete)

            # Fall through to bit-identical upstream behaviour when the
            # user hasn't asked for chunking, or when the requested chunk
            # already covers every particle.
            if chunk_size is None or chunk_size >= num_delete:
                return jax.vmap(mcmc_kernel)(sample_keys, start_state)

            # Chunked path: jax.lax.map(batch_size=k) vmaps within each
            # chunk-of-k particles and loops across chunks.
            return jax.lax.map(
                lambda xs: mcmc_kernel(xs[0], xs[1]),
                (sample_keys, start_state),
                batch_size=chunk_size,
            )

        return update_function

    return chunked_update_with_mcmc_take_last
