from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from autonerves import conf

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mcmc.blackjax.chains import (
    InverseMassMatrixSpec,
    inverse_mass_matrix_kind_from,
    resolve_inverse_mass_matrix,
    split_chain_diagnostics,
    stack_initial_positions,
)
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.samples.mcmc import SamplesMCMC
from autofit.non_linear.samples.sample import Sample

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa

logger = logging.getLogger(__name__)


class BlackJAXNUTS(AbstractMCMC):
    __identifier_fields__ = (
        "num_warmup",
        "num_samples",
        "num_chains",
        "inverse_mass_matrix",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        target_accept: float = 0.8,
        max_num_doublings: int = 10,
        seed: int = 42,
        initializer: Optional[Initializer] = None,
        inverse_mass_matrix: InverseMassMatrixSpec = None,
        mass_matrix_shrinkage: float = 0.0,
        share_adaptation: bool = False,
        auto_correlation_settings: AutoCorrelationsSettings = AutoCorrelationsSettings(
            check_for_convergence=False
        ),
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: Optional[int] = None,
        number_of_cores: int = 1,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A BlackJAX No-U-Turn Sampler (NUTS) non-linear search.

        BlackJAX is a sampling library implemented on top of JAX. NUTS is its
        gradient-based MCMC kernel — an extension of HMC that adapts trajectory
        length on the fly. The autofit ``Analysis`` therefore must be
        constructed with ``use_jax=True`` so the log-likelihood is JAX-traceable
        end-to-end (and ``jax.grad`` of it can be taken). A clear error is
        raised at fit time otherwise.

        For a full description of BlackJAX, see:

        https://github.com/blackjax-devs/blackjax

        The fit runs in two phases, both vmapped over ``num_chains`` independent
        chains:

        1) ``blackjax.window_adaptation`` warmup, which tunes the leapfrog step
           size (dual averaging) and an inverse mass matrix (diagonal or dense,
           see ``inverse_mass_matrix`` below) per chain.
        2) NUTS sampling with each chain's tuned kernel, run inside a
           ``jax.lax.scan`` so the inner step is fully JIT-compiled. The scan is
           broken into ``iterations_per_full_update``-sized chunks so partial
           state is persisted and ``perform_update`` runs periodically
           (samples.csv flush, plotting), mirroring the Emcee chunking pattern.

        **Warm starting.** Both the starting point(s) and the mass matrix can be
        seeded from a previous `Result`, without ever modifying that result's
        model's priors (contrast with ``Result.model_centred*``, which rewrite
        priors to new `GaussianPrior`s). For example, to warm-start a 4-chain
        run from an earlier (e.g. single-chain, or MLE) fit's ``result``::

            search = af.BlackJAXNUTS(
                num_chains=4,
                initializer=result.start_point_from(n_points=4, jitter=0.05),
                inverse_mass_matrix=result,
            )

        ``result.start_point_from(...)`` (equivalently
        ``InitializerParamStartPoints.from_result(result, n_points=4, jitter=0.05)``)
        maps the previous result's best-fit point onto the (possibly different)
        target model by path, then jitters ``n_points`` physical-space starting
        vectors around it — one per chain. Passing ``inverse_mass_matrix=result``
        (or ``result.samples``) seeds warmup's dense inverse mass matrix from
        that result's ``samples.covariance_matrix``; this raises a clear error
        if the result looks MLE-only (too few samples for a reliable
        covariance), in which case an explicit array (e.g. from a Laplace
        approximation) should be passed instead.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are
            output to.
        path_prefix
            The path of folders prefixing the name folder where results are
            output.
        unique_tag
            A unique tag for this model-fit, used as a folder between the
            path prefix and the search name and as the SQLite identifier.
        num_warmup
            Number of warmup steps used by ``blackjax.window_adaptation``
            to tune the leapfrog step size and inverse mass matrix.
        num_samples
            Number of post-warmup samples drawn from the tuned NUTS kernel,
            per chain.
        num_chains
            Number of independent chains sampled in parallel via ``vmap``.
        target_accept
            Target Metropolis acceptance rate for window adaptation
            (default 0.8 — the standard Stan setting).
        max_num_doublings
            NUTS doubling cap. 10 means at most 1024 leapfrog steps per
            sample, which is the standard ceiling.
        seed
            Integer seed passed to ``jax.random.PRNGKey``.
        initializer
            Generates the starting point(s), one per chain. Defaults to
            ``InitializerBall(0.49, 0.51)`` from ``AbstractMCMC`` — small
            ball around the prior median in unit-cube coordinates. Pass
            ``result.start_point_from(n_points=num_chains, jitter=...)`` (or
            an ``InitializerParamStartPoints.from_result(...)``) to warm-start
            from a previous result instead.
        inverse_mass_matrix
            Controls the metric adapted by ``blackjax.window_adaptation``:

            - ``None`` (default): adapt a fresh diagonal inverse mass matrix,
              seeded at the identity (standard behaviour).
            - ``"diagonal"`` / ``"dense"``: adapt a diagonal / dense inverse
              mass matrix, still seeded at the identity.
            - a ``numpy`` array: seed the adaptation with this matrix -- a 1-D
              array of shape ``(n_dim,)`` seeds a diagonal metric, a 2-D array
              of shape ``(n_dim, n_dim)`` seeds a dense metric.
            - a `Result` or `Samples` object: seed a dense metric from its
              ``samples.covariance_matrix``. Raises ``ValueError`` if that
              covariance looks MLE-only (too few samples, or non-finite /
              identity) -- pass an explicit array in that case.
        mass_matrix_shrinkage
            Forwarded to ``blackjax.window_adaptation``'s
            ``imm_shrinkage_to_previous``: shrinkage of each warmup window's
            adapted inverse mass matrix toward the *previous* window's (in
            addition to Stan's existing shrinkage toward the identity).
            ``0.0`` (default) reproduces Stan's standard per-window-reset
            behaviour; a positive value lets a high-confidence
            ``inverse_mass_matrix`` seed persist further into warmup.
        share_adaptation
            If ``True``, after per-chain warmup the ``num_chains`` tuned
            inverse mass matrices are averaged and the tuned step sizes
            combined via their median, and every chain's sampling phase uses
            this single shared, shared kernel. If ``False`` (default) each
            chain samples with its own independently-tuned kernel.
        auto_correlation_settings
            Configures the per-parameter ESS-derived integrated
            auto-correlation diagnostics. ``check_for_convergence`` defaults
            to ``False`` here because NUTS is run to a fixed sample budget
            after warmup.
        iterations_per_full_update
            Sample chunk size between ``perform_update`` calls. Inherited
            from the autonerves config when ``None``.
        number_of_cores
            Currently unused — chains run vmapped on a single device. Kept
            for API parity with the other MCMC searches.
        silence
            If True, the default print output of the non-linear search is
            silenced.
        session
            An SQLalchemy session instance.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            auto_correlation_settings=auto_correlation_settings,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.target_accept = target_accept
        self.max_num_doublings = max_num_doublings
        self.seed = seed

        # The raw specification (used at fit time to resolve an actual seed
        # array) is kept private; the public `inverse_mass_matrix` attribute
        # is the small descriptive string used by `__identifier_fields__` and
        # persisted onto `search_internal` / `samples_info`. Validated eagerly
        # here (`inverse_mass_matrix_kind_from` raises on an unrecognised
        # spec) so a bad value fails at construction, not mid-fit.
        self._inverse_mass_matrix_spec = inverse_mass_matrix
        self.inverse_mass_matrix = inverse_mass_matrix_kind_from(inverse_mass_matrix)
        self.mass_matrix_shrinkage = mass_matrix_shrinkage
        self.share_adaptation = share_adaptation

        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating BlackJAXNUTS Search")

        conf.instance["output"]["search_internal"] = True

    def apply_test_mode(self):
        logger.warning(
            "TEST MODE 1 (reduced iterations): BlackJAXNUTS will run with "
            "num_warmup=20, num_samples=20, num_chains<=2 for faster completion."
        )
        self.num_warmup = 20
        self.num_samples = 20
        self.num_chains = min(self.num_chains, 2)

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using BlackJAX NUTS. The autofit ``Analysis`` must be
        constructed with ``use_jax=True`` so the log-likelihood is
        JAX-traceable and differentiable.

        Returns
        -------
        (search_internal, fitness): tuple
            ``search_internal`` is the dict pickled under
            ``search_internal/``; ``fitness`` is the autofit Fitness wrapper.
        """
        import jax
        import jax.numpy as jnp
        import blackjax

        # JAX is mandatory: NUTS needs gradients of the log-density. Refuse
        # cleanly if the analysis was built without ``use_jax=True``.
        xp = getattr(analysis, "_xp", None)
        if xp is None or not xp.__name__.startswith("jax"):
            raise ValueError(
                "BlackJAXNUTS requires an Analysis built with use_jax=True. "
                "NUTS is a gradient-based sampler and the log-likelihood must "
                "flow through jax.grad. Construct Analysis(..., use_jax=True) "
                "and call enable_pytrees() / register_model(model) before fit. "
                "See autofit_workspace_test/scripts/searches/BlackJAXNUTS.py "
                "for a worked example."
            )

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,  # log-posterior target for NUTS
            resample_figure_of_merit=-jnp.inf,
            iterations_per_quick_update=self.iterations_per_quick_update,
            background_quick_update=self.quick_update_background,
            live_visual_update=self.live_visual_update,
        )

        # Initial position(s): borrow the standard initializer machinery so
        # users can substitute their own (InitializerBall by default for
        # MCMC; InitializerParamStartPoints.from_result(...) to warm-start).
        # One starting point per chain.
        unit_lists, parameter_lists, _ = self.initializer.samples_from_model(
            total_points=self.num_chains,
            model=model,
            fitness=fitness,
            paths=self.paths,
            n_cores=self.number_of_cores,
        )

        self.plot_start_point(
            parameter_vector=parameter_lists[0],
            model=model,
            analysis=analysis,
        )

        n_dim = model.prior_count

        # (num_chains, n_dim)
        initial_positions = jnp.asarray(stack_initial_positions(parameter_lists))

        # Build the JIT'd log-density target. ``fitness.call`` is the pure
        # JAX-traceable path (it routes through model.instance_from_vector and
        # analysis.log_likelihood_function with xp=jnp) — distinct from
        # ``call_wrap``/``__call__``, which add Python-side history tracking
        # and a ``float()`` conversion that would break NUTS gradients.
        @jax.jit
        def log_density(params):
            return fitness.call(params)

        # One-shot trace + compile so warmup timing is honest.
        _ = float(log_density(initial_positions[0]))

        rng_key = jax.random.PRNGKey(self.seed)

        # ---- Inverse mass matrix -----------------------------------------
        is_mass_matrix_diagonal, imm_seed = resolve_inverse_mass_matrix(
            self._inverse_mass_matrix_spec, n_dim=n_dim
        )

        # ---- Warmup (vmapped over chains) --------------------------------
        self.logger.info(
            f"BlackJAXNUTS: window adaptation ({self.num_warmup} steps x "
            f"{self.num_chains} chains, target_accept={self.target_accept}, "
            f"inverse_mass_matrix={self.inverse_mass_matrix})"
        )

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_density,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            initial_inverse_mass_matrix=imm_seed,
            imm_shrinkage_to_previous=self.mass_matrix_shrinkage,
            target_acceptance_rate=self.target_accept,
            max_num_doublings=self.max_num_doublings,
        )

        rng_key, warmup_key = jax.random.split(rng_key)
        warmup_keys = jax.random.split(warmup_key, self.num_chains)

        def run_warmup(key, position):
            return warmup.run(key, position, num_steps=self.num_warmup)

        (last_state, tuned_params), _ = jax.vmap(run_warmup)(
            warmup_keys, initial_positions
        )
        jax.block_until_ready(last_state.position)

        if self.share_adaptation:
            shared_step_size = jnp.median(tuned_params["step_size"])
            shared_inverse_mass_matrix = jnp.mean(
                tuned_params["inverse_mass_matrix"], axis=0
            )
            tuned_params = {
                "step_size": jnp.full_like(
                    tuned_params["step_size"], shared_step_size
                ),
                "inverse_mass_matrix": jnp.broadcast_to(
                    shared_inverse_mass_matrix,
                    tuned_params["inverse_mass_matrix"].shape,
                ),
            }

        # ---- Sampling (vmapped over chains) ------------------------------
        self.logger.info(
            f"BlackJAXNUTS: sampling ({self.num_samples} steps x "
            f"{self.num_chains} chains, chunked "
            f"{self.iterations_per_full_update} per perform_update)"
        )

        # `blackjax.nuts(...)` fixes a single (step_size, inverse_mass_matrix)
        # pair; building the kernel directly and vmapping it over the
        # per-chain tuned params lets each chain keep (or share, see
        # `share_adaptation` above) its own tuned metric.
        nuts_kernel = blackjax.nuts.build_kernel()

        def step_fn(key, state, step_size, inverse_mass_matrix):
            return nuts_kernel(
                key,
                state,
                log_density,
                step_size,
                inverse_mass_matrix,
                self.max_num_doublings,
            )

        vmapped_step = jax.vmap(step_fn, in_axes=(0, 0, 0, 0))

        def one_step(state, rng_key):
            step_keys = jax.random.split(rng_key, self.num_chains)
            new_state, info = vmapped_step(
                step_keys,
                state,
                tuned_params["step_size"],
                tuned_params["inverse_mass_matrix"],
            )
            return new_state, (new_state, info)

        def run_chunk(rng_key, initial_state, n_steps):
            # No outer @jax.jit: ``random.split`` rejects a traced size, and
            # ``lax.scan`` already JIT-compiles its body so the inner per-step
            # kernel still runs as a single fused XLA computation.
            keys = jax.random.split(rng_key, n_steps)
            _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
            return states, infos

        positions_chunks = []
        log_likelihood_chunks = []
        info_chunks = {
            "acceptance_rate": [],
            "num_integration_steps": [],
            "is_divergent": [],
            "num_trajectory_expansions": [],
        }

        state = last_state
        total_done = 0
        iterations_remaining = self.num_samples

        while iterations_remaining > 0:
            # ``run_chunk`` scans ``chunk_n`` steps and the key split below sizes
            # itself from it, so this must be an ``int`` (PyAutoFit#1422).
            chunk_n = self._steps_until_full_update(iterations_remaining)

            rng_key, sample_key = jax.random.split(rng_key)
            states, infos = run_chunk(sample_key, state, chunk_n)
            jax.block_until_ready(states.position)

            # Per-sample log-likelihood for the chunk (NUTS only stores the
            # log-density inside its kernel state — we recompute the
            # log-likelihood explicitly so the resulting SamplesMCMC has a
            # clean log_likelihood / log_prior split). ``states.position`` has
            # shape (chunk_n, num_chains, n_dim); vmap the per-sample
            # log-likelihood over both leading axes.
            chunk_log_l = jax.vmap(jax.vmap(_log_likelihood_only(fitness)))(
                states.position
            )

            positions_chunks.append(np.asarray(states.position))
            log_likelihood_chunks.append(np.asarray(chunk_log_l))
            info_chunks["acceptance_rate"].append(np.asarray(infos.acceptance_rate))
            info_chunks["num_integration_steps"].append(
                np.asarray(infos.num_integration_steps)
            )
            info_chunks["is_divergent"].append(np.asarray(infos.is_divergent))
            info_chunks["num_trajectory_expansions"].append(
                np.asarray(infos.num_trajectory_expansions)
            )

            # Carry forward the last state position for the next chunk.
            state = jax.tree_util.tree_map(lambda x: x[-1], states)

            total_done += chunk_n
            iterations_remaining = self.num_samples - total_done

            search_internal = _build_search_internal(
                positions_chunks=positions_chunks,
                log_likelihood_chunks=log_likelihood_chunks,
                info_chunks=info_chunks,
                tuned_params=tuned_params,
                last_state_position=np.asarray(state.position),
                num_warmup=self.num_warmup,
                num_samples_completed=total_done,
                num_samples_total=self.num_samples,
                num_chains=self.num_chains,
                warm_start_source=type(self.initializer).__name__,
                inverse_mass_matrix_kind=self.inverse_mass_matrix,
            )

            self.output_search_internal(search_internal=search_internal)

            if iterations_remaining > 0:
                self.perform_update(
                    model=model,
                    analysis=analysis,
                    search_internal=search_internal,
                    fitness=fitness,
                    during_analysis=True,
                )

        return search_internal, fitness

    # ------------------------------------------------------------------
    # Persistence + samples
    # ------------------------------------------------------------------

    @property
    def backend_filename(self):
        return self.paths.search_internal_path / "search_internal.pickle"

    @property
    def backend(self) -> dict:
        """Load the pickled search-internal dict written by ``_fit``."""
        if not Path(self.backend_filename).is_file():
            raise FileNotFoundError(
                f"search_internal.pickle does not exist at "
                f"{self.paths.search_internal_path}"
            )
        with open(self.backend_filename, "rb") as f:
            return pickle.load(f)

    def output_search_internal(self, search_internal):
        """
        Pickle the search-internal dict.

        BlackJAX has no native on-disk format (cf. emcee's HDFBackend), so we
        round-trip the chain + diagnostics via pickle. We bypass
        ``self.paths.save_search_internal`` because the autofit dill path
        chokes on a few numpy/jax-backed members; a direct pickle of
        already-numpy data is robust.

        ``NullPaths`` (no ``name``/``path_prefix``) sets
        ``search_internal_path`` to ``None`` to suppress disk output —
        skip silently in that case.
        """
        if self.paths.search_internal_path is None:
            return
        os.makedirs(self.paths.search_internal_path, exist_ok=True)
        with open(self.backend_filename, "wb") as f:
            pickle.dump(search_internal, f)

    def _test_mode_samples_info(self) -> dict:
        return {
            "num_warmup": int(self.num_warmup),
            "num_samples": 0,
            "num_chains": int(self.num_chains),
            "ess_min": float("nan"),
            "ess_per_param": [],
            "ess_bulk_per_param": [],
            "ess_tail_per_param": [],
            "ess_bulk_min": float("nan"),
            "ess_tail_min": float("nan"),
            "rhat_per_param": [],
            "rhat_max": float("nan"),
            "mean_acceptance": float("nan"),
            "n_divergent": 0,
            "divergent_indices": [],
            "tree_depth_histogram": {},
            "n_logl_evals": 0,
            "total_walkers": int(self.num_chains),
            "total_steps": 0,
            "warm_start_source": type(self.initializer).__name__,
            "inverse_mass_matrix_kind": self.inverse_mass_matrix,
        }

    def samples_info_from(self, search_internal=None):
        search_internal = search_internal if search_internal is not None else self.backend

        positions = search_internal["positions"]  # (n_samples, n_chains, n_dim)
        info = search_internal["infos"]

        ess_per_param = _ess_per_param_from(positions)
        n_logl_evals = int(info["num_integration_steps"].sum())
        mean_acceptance = float(info["acceptance_rate"].mean())

        is_divergent = info["is_divergent"]
        n_divergent = int(is_divergent.sum())
        divergent_indices = [
            [int(sample_index), int(chain_index)]
            for sample_index, chain_index in np.argwhere(is_divergent)
        ]

        tree_depths = np.asarray(info["num_trajectory_expansions"])
        tree_depth_values, tree_depth_counts = np.unique(
            tree_depths, return_counts=True
        )
        tree_depth_histogram = {
            int(depth): int(count)
            for depth, count in zip(tree_depth_values, tree_depth_counts)
        }

        ess_bulk_per_param, ess_tail_per_param, rhat_per_param = _chain_diagnostics_from(
            positions
        )

        return {
            "num_warmup": int(search_internal["num_warmup"]),
            "num_samples": int(search_internal["num_samples_completed"]),
            "num_chains": int(search_internal["num_chains"]),
            "ess_min": float(ess_per_param.min()),
            "ess_per_param": ess_per_param.tolist(),
            "ess_bulk_per_param": ess_bulk_per_param.tolist(),
            "ess_tail_per_param": ess_tail_per_param.tolist(),
            "ess_bulk_min": float(ess_bulk_per_param.min()),
            "ess_tail_min": float(ess_tail_per_param.min()),
            "rhat_per_param": rhat_per_param.tolist(),
            "rhat_max": float(rhat_per_param.max()),
            "mean_acceptance": mean_acceptance,
            "n_divergent": n_divergent,
            "divergent_indices": divergent_indices,
            "tree_depth_histogram": tree_depth_histogram,
            "n_logl_evals": n_logl_evals,
            "check_size": self.auto_correlation_settings.check_size,
            "required_length": self.auto_correlation_settings.required_length,
            "change_threshold": self.auto_correlation_settings.change_threshold,
            "total_walkers": int(search_internal["num_chains"]),
            "total_steps": int(search_internal["num_samples_completed"]),
            "warm_start_source": search_internal.get("warm_start_source", "none"),
            "inverse_mass_matrix_kind": search_internal.get(
                "inverse_mass_matrix_kind", "none"
            ),
            "time": self.timer.time if self.timer else None,
        }

    def samples_via_internal_from(self, model, search_internal=None):
        """
        Convert the BlackJAX chain pickled under ``search_internal/`` into a
        standard ``SamplesMCMC``. NUTS samples are unweighted draws from the
        posterior, so weights are 1.0.
        """
        search_internal = search_internal if search_internal is not None else self.backend

        positions = search_internal["positions"]  # (n_samples, n_chains, n_dim)
        log_likelihood_array = search_internal["log_likelihood_history"]  # (n_samples, n_chains)

        # Chain-major flatten: all of chain 0's samples (in draw order), then
        # chain 1's, etc. -- matches `total_walkers = num_chains` below.
        n_samples, n_chains, n_dim = positions.shape
        positions_chain_major = np.moveaxis(positions, 0, 1).reshape(
            n_chains * n_samples, n_dim
        )
        log_likelihood_chain_major = np.moveaxis(log_likelihood_array, 0, 1).reshape(
            n_chains * n_samples
        )

        parameter_lists = positions_chain_major.tolist()
        log_likelihood_list = [float(x) for x in log_likelihood_chain_major]
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        weight_list = [1.0] * len(parameter_lists)

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return SamplesMCMC(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
            auto_correlation_settings=self.auto_correlation_settings,
            auto_correlations=self.auto_correlations_from(
                search_internal=search_internal
            ),
        )

    def auto_correlations_from(self, search_internal=None):
        """
        Synthesise the standard ``AutoCorrelations`` from BlackJAX's per-param
        ESS via the canonical identity ``τ_int = N / ESS``. ``previous_times``
        uses the chain truncated by ``check_size`` so the
        relative-change convergence metric stays meaningful.
        """
        search_internal = search_internal if search_internal is not None else self.backend
        positions = search_internal["positions"]  # (n_samples, n_chains, n_dim)
        n_samples = positions.shape[0]

        check_size = self.auto_correlation_settings.check_size

        times = _times_from_positions(positions)

        # Slice for "previous" — match emcee's ``[:-check_size]`` pattern.
        # If check_size >= n_samples (e.g. early in a long run), fall back
        # to a half-chain split so the comparison still has signal.
        if check_size < n_samples:
            previous_positions = positions[:-check_size]
        else:
            previous_positions = positions[: max(1, n_samples // 2)]

        if previous_positions.shape[0] >= 2:
            previous_times = _times_from_positions(previous_positions)
        else:
            previous_times = times.copy()

        return AutoCorrelations(
            check_size=check_size,
            required_length=self.auto_correlation_settings.required_length,
            change_threshold=self.auto_correlation_settings.change_threshold,
            times=times,
            previous_times=previous_times,
        )


# --------------------------------------------------------------------------
# Module-level helpers (kept outside the class so jax.vmap / jax.jit don't
# capture ``self`` and trip pytree complaints)
# --------------------------------------------------------------------------


def _log_likelihood_only(fitness: Fitness):
    """
    Return a JAX-traceable function ``log_l(params)`` that mirrors
    ``fitness.call`` but strips the prior contribution. Used post-sampling
    to populate the per-sample log_likelihood column independently of the
    log_prior column (which is computed by ``model.log_prior_list_from``).
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def log_l(params):
        instance = fitness.model.instance_from_vector(vector=params, xp=jnp)
        return fitness.analysis.log_likelihood_function(instance=instance)

    return log_l


def _ess_per_param_from(positions: np.ndarray) -> np.ndarray:
    """
    Per-parameter effective sample size via BlackJAX's Geyer-style
    monotone variance estimator.

    ``positions`` has shape ``(num_samples, num_chains, n_dim)`` (the
    ``search_internal`` layout, for any ``num_chains >= 1``); this is
    reshaped to BlackJAX's ``(chain_axis=0, sample_axis=1)`` convention via
    ``split_chain_diagnostics`` before the diagnostic is applied, so a
    single-chain run (``num_chains == 1``) sees the same
    ``(1, num_samples, n_dim)`` shape the original single-chain
    implementation used.
    """
    import jax.numpy as jnp
    from blackjax.diagnostics import effective_sample_size

    chain_major = split_chain_diagnostics(positions)
    ess = effective_sample_size(jnp.asarray(chain_major))
    return np.atleast_1d(np.asarray(ess))


def _chain_diagnostics_from(positions: np.ndarray):
    """
    Bulk/tail ESS and rank-normalised split-R-hat per parameter, via
    ``blackjax.diagnostics``. See ``_ess_per_param_from`` for the
    ``(num_samples, num_chains, n_dim)`` -> ``(chain_axis, sample_axis)``
    reshape. R-hat with a single chain is split-chain only (BlackJAX splits
    each chain in half internally before comparing) -- it is still a useful
    non-stationarity check, just not a genuine multi-chain convergence
    diagnostic in that case.
    """
    import jax.numpy as jnp
    from blackjax.diagnostics import ess_bulk, ess_tail, rhat

    chain_major = jnp.asarray(split_chain_diagnostics(positions))

    ess_bulk_per_param = np.atleast_1d(np.asarray(ess_bulk(chain_major)))
    ess_tail_per_param = np.atleast_1d(np.asarray(ess_tail(chain_major)))
    rhat_per_param = np.atleast_1d(np.asarray(rhat(chain_major)))

    return ess_bulk_per_param, ess_tail_per_param, rhat_per_param


def _times_from_positions(positions: np.ndarray) -> np.ndarray:
    """
    Synthesise integrated auto-correlation times from per-parameter ESS via
    the standard identity ``τ_int = N / ESS``. Clamped: if ESS < 1 (rare,
    e.g. mostly-divergent chain), we floor ``ESS = 1`` so ``times`` does
    not go past ``num_samples`` and ``AutoCorrelations.check_if_converged``
    stays well-defined.

    ``positions`` has shape ``(num_samples, num_chains, n_dim)``; ``N`` in
    the identity above is the *total* sample count across all chains
    (``num_samples * num_chains``), matching what ``effective_sample_size``
    itself pools over.
    """
    n_samples = positions.shape[0] * positions.shape[1]
    ess = _ess_per_param_from(positions)
    ess = np.clip(ess, a_min=1.0, a_max=None)
    return n_samples / ess


def _build_search_internal(
    positions_chunks,
    log_likelihood_chunks,
    info_chunks,
    tuned_params,
    last_state_position,
    num_warmup,
    num_samples_completed,
    num_samples_total,
    num_chains,
    warm_start_source="none",
    inverse_mass_matrix_kind="none",
):
    """
    Glue chunked sampling output into the persistence dict pickled under
    ``search_internal/search_internal.pickle``.
    """
    return {
        "positions": np.concatenate(positions_chunks, axis=0),
        "log_likelihood_history": np.concatenate(log_likelihood_chunks, axis=0),
        "infos": {
            "acceptance_rate": np.concatenate(info_chunks["acceptance_rate"]),
            "num_integration_steps": np.concatenate(
                info_chunks["num_integration_steps"]
            ),
            "is_divergent": np.concatenate(info_chunks["is_divergent"]),
            "num_trajectory_expansions": np.concatenate(
                info_chunks["num_trajectory_expansions"]
            ),
        },
        # tuned_params often contains JAX arrays; convert to numpy so the
        # pickle is portable across JAX versions.
        "tuned_params": {
            k: (np.asarray(v) if hasattr(v, "shape") else v)
            for k, v in tuned_params.items()
        },
        "last_state_position": last_state_position,
        "num_warmup": num_warmup,
        "num_samples_completed": num_samples_completed,
        "num_samples": num_samples_total,
        "num_chains": num_chains,
        "warm_start_source": warm_start_source,
        "inverse_mass_matrix_kind": inverse_mass_matrix_kind,
    }
