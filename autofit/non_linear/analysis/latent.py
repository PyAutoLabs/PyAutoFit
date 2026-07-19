"""
First-class latent-variable extension point and computation engine.

This module holds the two halves of PyAutoFit's latent-variable machinery,
extracted from ``Analysis`` so the concern is self-contained (mirroring how
``Visualizer`` and ``Result`` are declared on an ``Analysis``):

- :class:`Latent` — the user/library extension point. Declare it on an analysis
  as ``Latent = MyLatent`` (just like ``Visualizer = MyVisualizer``) and override
  :meth:`Latent.keys` / :meth:`Latent.variables` to define latent variables.
  Methods are ``@staticmethod`` taking ``analysis`` first, matching the
  ``Visualizer`` convention.

- :func:`latent_samples_from` — the batched evaluation engine (JAX vmap/jit or
  NumPy dispatch, global NaN masking + greedy salvage). ``Analysis.compute_latent_samples``
  is now a thin wrapper around it.

Backwards compatibility: the default :class:`Latent` delegates to the (now
legacy) ``Analysis.LATENT_KEYS`` / ``Analysis.compute_latent_variables`` /
``Analysis.LATENT_BATCH_MODE``, so analyses that still override those keep
working unchanged. New code should subclass :class:`Latent` instead.
"""
import functools
import logging
import time
from typing import Optional

import numpy as np

from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.samples.util import simple_model_for_kwargs

logger = logging.getLogger(__name__)


class Latent:
    """
    Latent-variable extension point, declared on an analysis as
    ``Analysis.Latent = MyLatent`` (mirrors ``Visualizer`` / ``Result``).

    Subclass and override :meth:`keys` and :meth:`variables` to define a
    catalogue of latent variables. All methods are ``@staticmethod`` and take
    the ``analysis`` as their first argument, so per-fit state (e.g.
    ``analysis.kwargs["magzero"]``) is reachable without an instance lifecycle.

    The base implementation is a **backwards-compatibility shim**: it reads the
    legacy ``Analysis.LATENT_KEYS`` and ``Analysis.compute_latent_variables``,
    so existing analyses that override those continue to work without declaring
    a ``Latent``. New code should subclass this class instead.
    """

    # JAX batch strategy: "vmap" or "jit". ``None`` means "defer to
    # ``analysis.LATENT_BATCH_MODE``" — used by the back-compat shim so that an
    # analysis which set ``LATENT_BATCH_MODE = "jit"`` (e.g. autogalaxy's
    # ``AnalysisDataset``) keeps that choice without declaring a ``Latent``.
    BATCH_MODE = None

    @staticmethod
    def keys(analysis) -> list:
        """The ordered list of enabled latent names (back-compat: ``LATENT_KEYS``)."""
        return list(analysis.LATENT_KEYS)

    @staticmethod
    def variables(analysis, parameters, model):
        """
        Compute the latent values for one sample, returned as a tuple/dict
        positionally aligned with :meth:`keys`. Back-compat default delegates
        to ``analysis.compute_latent_variables``.
        """
        return analysis.compute_latent_variables(parameters, model)


def latent_samples_from(
    analysis, samples: Samples, batch_size: Optional[int] = None
) -> Optional[Samples]:
    """
    Compute latent-variable samples for every posterior sample.

    A latent variable is not a free parameter of the model but can be derived
    from it; its values (with errors) are stored in ``latent/`` alongside
    ``samples.csv``. The latents to compute and how to compute them come from
    ``analysis.Latent`` (:meth:`Latent.keys` / :meth:`Latent.variables`).

    Compatible with NumPy and JAX: the per-sample computation is side-effect
    free, batched via ``jax.vmap`` or per-sample ``jax.jit`` (selected by
    ``Latent.BATCH_MODE``), and any NaN/Inf values are masked out globally
    (degenerate latents dropped; samples carrying a NaN in a surviving latent
    dropped; anti-correlated NaNs salvaged by sacrificing the worst-coverage
    latent rather than discarding everything).

    Returns ``None`` when no finite latent samples remain (or no latents are
    defined).
    """
    batch_size = batch_size or 10

    latent = analysis.Latent
    batch_mode = (
        latent.BATCH_MODE
        if latent.BATCH_MODE is not None
        else analysis.LATENT_BATCH_MODE
    )
    keys = latent.keys(analysis)

    if not keys:
        return None

    try:

        start_latent = time.time()

        compute_latent_for_model = functools.partial(
            latent.variables, analysis, model=samples.model
        )

        if analysis._use_jax:
            import jax
            import jax.numpy as jnp
            start = time.time()
            if batch_mode == "vmap":
                logger.info("JAX: Applying vmap and jit to likelihood function for latent variables -- may take a few seconds.")
                # vmap traces `variables` once for the whole batch, so a
                # per-sample try/except is not possible here — latent functions
                # on the vmap path must express failures as NaN (e.g.
                # `jnp.where`), never by raising. The `jit` and numpy paths
                # below do guard per sample.
                batched_compute_latent = jax.jit(jax.vmap(compute_latent_for_model))
            elif batch_mode == "jit":
                logger.info("JAX: Applying per-sample jit to latent variables (LATENT_BATCH_MODE='jit') -- may take a few seconds on first sample.")
                jitted_compute_latent = jax.jit(compute_latent_for_model)
                n_latents = len(keys)
                nan_tuple = tuple(jnp.nan for _ in range(n_latents))

                def _safe_jitted(p):
                    # A latent that raises (any exception, not just
                    # FitException) becomes a NaN row, which the global mask
                    # below drops — one bad sample must not abort the batch.
                    try:
                        return jitted_compute_latent(p)
                    except Exception:
                        return nan_tuple

                def batched_compute_latent(parameters_batch):
                    # Per-sample jit returns one (l1, l2, ..., lN) tuple per
                    # sample. Transpose to a tuple of N batched arrays so the
                    # downstream `jnp.stack(latent_values_batch, axis=-1)`
                    # works identically to the vmap path.
                    sample_results = [
                        _safe_jitted(p) for p in parameters_batch
                    ]
                    n_latents = len(sample_results[0])
                    return tuple(
                        jnp.stack([s[i] for s in sample_results])
                        for i in range(n_latents)
                    )
            else:
                raise ValueError(
                    f"Unknown LATENT_BATCH_MODE={batch_mode!r}; expected 'vmap' or 'jit'."
                )
            logger.info(f"JAX: {batch_mode} dispatch applied in {time.time() - start} seconds.")
        else:
            n_latents = len(keys)
            nan_row = np.full(n_latents, np.nan)

            def _safe_compute(xx):
                # Any exception (not just FitException) becomes a NaN row,
                # which the global mask below drops — a single failing latent
                # evaluation must not abort the whole post-fit latent pass.
                try:
                    return compute_latent_for_model(xx)
                except Exception:
                    return nan_row

            def batched_compute_latent(x):
                return np.array([_safe_compute(xx) for xx in x])

        from autonerves.test_mode import inject_latent_nans

        parameter_array = np.array(samples.parameter_lists)

        # Compute every batch first and accumulate the raw, UN-masked latent
        # values into one (n_samples, n_latents) array. Masking is then done
        # ONCE, globally, after the loop (see below).
        #
        # Doing the finite mask per batch (the previous behaviour) was a bug:
        # a latent that went NaN for a single sample in one batch had its whole
        # column dropped *for that batch only*, while other batches kept it.
        # The resulting `Sample` objects then carried inconsistent kwargs key
        # sets, and `Samples.summary()` raised `KeyError` building its model
        # from the first sample's keys. Masking globally guarantees every
        # retained sample shares one identical key set.
        all_values = []
        all_samples = []
        for i in range(0, len(parameter_array), batch_size):

            batch = parameter_array[i:i + batch_size]
            batch_samples = samples.sample_list[i:i + batch_size]

            # batched JAX call on this chunk
            latent_values_batch = batched_compute_latent(batch)

            if analysis._use_jax:
                import jax.numpy as jnp
                latent_values_batch = jnp.stack(latent_values_batch, axis=-1)  # (batch, n_latents)

            # Unify to NumPy so the global masking below is a single code path
            # for both backends (latent values are scalars, host transfer is
            # cheap and was already forced by the downstream `float(v)`).
            latent_values_batch = np.asarray(latent_values_batch)

            # Test-only NaN injection (no-op unless PYAUTO_LATENT_NAN_INJECT set).
            latent_values_batch = inject_latent_nans(latent_values_batch, start_index=i)

            all_values.append(latent_values_batch)
            all_samples.extend(batch_samples)

        if all_values:
            all_values = np.concatenate(all_values, axis=0)
        else:
            all_values = np.empty((0, len(keys)))

        # Global masking. Every retained sample must share one identical,
        # fully-finite latent key set (a `Samples` object is a rectangular
        # samples x latents block; NaNs anywhere break `quantile`).
        #
        # 1. Drop a latent column that is non-finite for EVERY sample
        #    (genuinely degenerate, e.g. a µJy latent with no magzero).
        kept_idx = [
            i
            for i in range(all_values.shape[1])
            if np.isfinite(all_values[:, i]).any()
        ]

        # 2. Keep samples that are finite across all currently-kept latents.
        #    Normally at least one sample is finite everywhere and we stop
        #    immediately (behaviour unchanged). But if NaNs are anti-correlated
        #    across latents — every sample NaN in some latent — the rectangular
        #    block is empty. Rather than discard ALL latent output, greedily
        #    sacrifice the worst-coverage latent and retry, retaining the
        #    maximal-coverage latents and their finite samples.
        while kept_idx:
            row_mask = np.isfinite(all_values[:, kept_idx]).all(axis=1)
            if row_mask.any():
                break
            nan_counts = (~np.isfinite(all_values[:, kept_idx])).sum(axis=0)
            kept_idx.pop(int(np.argmax(nan_counts)))

        print(f"Time to compute latent variables: {time.time() - start_latent} seconds for {len(samples)} samples.")

        if not kept_idx:
            logger.warning(
                "compute_latent_samples: no finite latent samples remained "
                "after masking; skipping latent output."
            )
            return None

        kept_keys = [keys[i] for i in kept_idx]
        kept_values = all_values[:, kept_idx]
        row_mask = np.isfinite(kept_values).all(axis=1)
        kept_values = kept_values[row_mask]
        kept_samples = [s for s, keep in zip(all_samples, row_mask) if keep]

        if len(kept_samples) == 0:
            logger.warning(
                "compute_latent_samples: no finite latent samples remained "
                "after masking; skipping latent output."
            )
            return None

        latent_samples = [
            Sample(
                log_likelihood=sample.log_likelihood,
                log_prior=sample.log_prior,
                weight=sample.weight,
                kwargs={k: float(v) for k, v in zip(kept_keys, values)},
            )
            for sample, values in zip(kept_samples, kept_values)
        ]

        return type(samples)(
            sample_list=latent_samples,
            model=simple_model_for_kwargs(latent_samples[0].kwargs),
            samples_info=samples.samples_info,
        )

    except NotImplementedError:
        return None
