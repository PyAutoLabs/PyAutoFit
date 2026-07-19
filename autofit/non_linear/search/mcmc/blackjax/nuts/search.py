import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from autonerves import conf

from autofit.database.sqlalchemy_ import sa
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.samples.mcmc import SamplesMCMC
from autofit.non_linear.samples.sample import Sample

logger = logging.getLogger(__name__)


class BlackJAXNUTS(AbstractMCMC):
    __identifier_fields__ = ("num_warmup", "num_samples", "num_chains")

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

        The fit runs in two phases:

        1) ``blackjax.window_adaptation`` warmup, which tunes the leapfrog step
           size (dual averaging) and a diagonal inverse mass matrix.
        2) NUTS sampling with the tuned kernel, run inside a ``jax.lax.scan``
           so the inner step is fully JIT-compiled. The scan is broken into
           ``iterations_per_full_update``-sized chunks so partial state is
           persisted and ``perform_update`` runs periodically (samples.csv
           flush, plotting), mirroring the Emcee chunking pattern.

        The single-chain default is the natural fit for the autofit MCMC
        plumbing (``SamplesMCMC``, ``AutoCorrelations``). Resume support is
        not implemented in v1; the on-disk pickle layout leaves room for it.

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
            to tune the leapfrog step size and diagonal inverse mass matrix.
        num_samples
            Number of post-warmup samples drawn from the tuned NUTS kernel.
        num_chains
            Number of independent chains. v1 uses 1; values >1 raise an
            error until multi-chain support lands.
        target_accept
            Target Metropolis acceptance rate for window adaptation
            (default 0.8 — the standard Stan setting).
        max_num_doublings
            NUTS doubling cap. 10 means at most 1024 leapfrog steps per
            sample, which is the standard ceiling.
        seed
            Integer seed passed to ``jax.random.PRNGKey``.
        initializer
            Generates the starting point. Defaults to
            ``InitializerBall(0.49, 0.51)`` from ``AbstractMCMC`` — small
            ball around the prior median in unit-cube coordinates.
        auto_correlation_settings
            Configures the per-parameter ESS-derived integrated
            auto-correlation diagnostics. ``check_for_convergence`` defaults
            to ``False`` here because NUTS is run to a fixed sample budget
            after warmup.
        iterations_per_full_update
            Sample chunk size between ``perform_update`` calls. Inherited
            from the autonerves config when ``None``.
        number_of_cores
            Currently unused — single chain runs on a single device. Kept
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

        if num_chains != 1:
            raise NotImplementedError(
                "BlackJAXNUTS currently supports num_chains=1 only. "
                "Multi-chain support will be added in a future revision."
            )

        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.target_accept = target_accept
        self.max_num_doublings = max_num_doublings
        self.seed = seed

        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating BlackJAXNUTS Search")

        conf.instance["output"]["search_internal"] = True

    def apply_test_mode(self):
        logger.warning(
            "TEST MODE 1 (reduced iterations): BlackJAXNUTS will run with "
            "num_warmup=20, num_samples=20 for faster completion."
        )
        self.num_warmup = 20
        self.num_samples = 20

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
        )

        # Initial position: borrow the standard initializer machinery so users
        # can substitute their own (InitializerBall by default for MCMC).
        # Single chain → ask for one starting point.
        unit_lists, parameter_lists, _ = self.initializer.samples_from_model(
            total_points=1,
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

        initial_position = jnp.asarray(parameter_lists[0])

        # Build the JIT'd log-density target. ``fitness.call`` is the pure
        # JAX-traceable path (it routes through model.instance_from_vector and
        # analysis.log_likelihood_function with xp=jnp) — distinct from
        # ``call_wrap``/``__call__``, which add Python-side history tracking
        # and a ``float()`` conversion that would break NUTS gradients.
        @jax.jit
        def log_density(params):
            return fitness.call(params)

        # One-shot trace + compile so warmup timing is honest.
        _ = float(log_density(initial_position))

        rng_key = jax.random.PRNGKey(self.seed)

        # ---- Warmup ----------------------------------------------------
        self.logger.info(
            f"BlackJAXNUTS: window adaptation ({self.num_warmup} steps, "
            f"target_accept={self.target_accept})"
        )

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_density,
            target_acceptance_rate=self.target_accept,
            max_num_doublings=self.max_num_doublings,
        )

        rng_key, warmup_key = jax.random.split(rng_key)
        (last_state, tuned_params), _ = warmup.run(
            warmup_key, initial_position, num_steps=self.num_warmup
        )
        jax.block_until_ready(last_state.position)

        # ---- Sampling --------------------------------------------------
        self.logger.info(
            f"BlackJAXNUTS: sampling ({self.num_samples} steps, "
            f"chunked {self.iterations_per_full_update} per perform_update)"
        )

        nuts_kernel = blackjax.nuts(log_density, **tuned_params)

        def one_step(state, rng_key):
            new_state, info = nuts_kernel.step(rng_key, state)
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
        }

        state = last_state
        total_done = 0
        iterations_remaining = self.num_samples

        while iterations_remaining > 0:
            chunk_n = min(self.iterations_per_full_update, iterations_remaining)

            rng_key, sample_key = jax.random.split(rng_key)
            states, infos = run_chunk(sample_key, state, chunk_n)
            jax.block_until_ready(states.position)

            # Per-sample log-likelihood for the chunk (NUTS only stores the
            # log-density inside its kernel state — we recompute the
            # log-likelihood explicitly so the resulting SamplesMCMC has a
            # clean log_likelihood / log_prior split).
            chunk_log_l = jax.vmap(_log_likelihood_only(fitness))(states.position)

            positions_chunks.append(np.asarray(states.position))
            log_likelihood_chunks.append(np.asarray(chunk_log_l))
            info_chunks["acceptance_rate"].append(np.asarray(infos.acceptance_rate))
            info_chunks["num_integration_steps"].append(
                np.asarray(infos.num_integration_steps)
            )
            info_chunks["is_divergent"].append(np.asarray(infos.is_divergent))

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
            "mean_acceptance": float("nan"),
            "n_divergent": 0,
            "n_logl_evals": 0,
            "total_walkers": int(self.num_chains),
            "total_steps": 0,
        }

    def samples_info_from(self, search_internal=None):
        search_internal = search_internal if search_internal is not None else self.backend

        positions = search_internal["positions"]
        info = search_internal["infos"]

        ess_per_param = _ess_per_param_from(positions)
        n_logl_evals = int(info["num_integration_steps"].sum())
        mean_acceptance = float(info["acceptance_rate"].mean())
        n_divergent = int(info["is_divergent"].sum())

        return {
            "num_warmup": int(search_internal["num_warmup"]),
            "num_samples": int(search_internal["num_samples_completed"]),
            "num_chains": int(search_internal["num_chains"]),
            "ess_min": float(ess_per_param.min()),
            "ess_per_param": ess_per_param.tolist(),
            "mean_acceptance": mean_acceptance,
            "n_divergent": n_divergent,
            "n_logl_evals": n_logl_evals,
            "check_size": self.auto_correlation_settings.check_size,
            "required_length": self.auto_correlation_settings.required_length,
            "change_threshold": self.auto_correlation_settings.change_threshold,
            "total_walkers": int(search_internal["num_chains"]),
            "total_steps": int(search_internal["num_samples_completed"]),
            "time": self.timer.time if self.timer else None,
        }

    def samples_via_internal_from(self, model, search_internal=None):
        """
        Convert the BlackJAX chain pickled under ``search_internal/`` into a
        standard ``SamplesMCMC``. NUTS samples are unweighted draws from the
        posterior, so weights are 1.0.
        """
        search_internal = search_internal if search_internal is not None else self.backend

        positions = search_internal["positions"]  # (num_samples, n_dim)
        log_likelihood_array = search_internal["log_likelihood_history"]

        parameter_lists = positions.tolist()
        log_likelihood_list = [float(x) for x in log_likelihood_array]
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
        positions = search_internal["positions"]
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
    monotone variance estimator. Single-chain shape contract: positions is
    ``(num_samples, n_dim)``; we add a length-1 chain axis so the
    diagnostic sees ``(n_chains=1, num_samples, n_dim)``.
    """
    import jax.numpy as jnp
    from blackjax.diagnostics import effective_sample_size

    ess = effective_sample_size(jnp.asarray(positions)[None, ...])
    return np.asarray(ess)


def _times_from_positions(positions: np.ndarray) -> np.ndarray:
    """
    Synthesise integrated auto-correlation times from per-parameter ESS via
    the standard identity ``τ_int = N / ESS``. Clamped: if ESS < 1 (rare,
    e.g. mostly-divergent chain), we floor ``ESS = 1`` so ``times`` does
    not go past ``num_samples`` and ``AutoCorrelations.check_if_converged``
    stays well-defined.
    """
    n_samples = positions.shape[0]
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
    }
