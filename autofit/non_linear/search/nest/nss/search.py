import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from autofit.database.sqlalchemy_ import sa
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.search.nest.nss.samples import NSSamples
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.test_mode import is_test_mode


try:
    import blackjax as _blackjax
    from nss.ns import (
        log_weights as _nss_log_weights,
        finalise as _nss_finalise,
    )

    _HAS_NSS = True
except ImportError:
    _blackjax = None
    _nss_log_weights = None
    _nss_finalise = None
    _HAS_NSS = False


logger = logging.getLogger(__name__)


_CHECKPOINT_FILENAME = "nss_checkpoint.pkl"


def _save_checkpoint(path, state, dead, run_key, iteration):
    """Atomically pickle the resumable state of an in-flight NSS run.

    The blackjax ``state`` and each entry in ``dead`` are pytrees of JAX arrays.
    JAX arrays do pickle directly, but to keep the on-disk format independent
    of the JAX install (so a checkpoint written on one cluster can be loaded
    on another) we round-trip through NumPy before writing. ``_load_checkpoint``
    reverses the conversion.

    A tmp-and-rename pattern guards against partial writes — a SLURM timeout
    halfway through a pickle dump leaves the previous-good checkpoint intact.
    """
    import jax

    to_numpy = lambda x: jax.tree_util.tree_map(np.asarray, x)
    blob = {
        "state": to_numpy(state),
        "dead": [to_numpy(d) for d in dead],
        "run_key": np.asarray(run_key),
        "iteration": int(iteration),
    }
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(blob, f)
    os.replace(tmp_path, path)


def _load_checkpoint(path):
    """Reverse of ``_save_checkpoint`` — restore a saved blob to JAX pytrees."""
    import jax
    import jax.numpy as jnp

    with open(path, "rb") as f:
        blob = pickle.load(f)
    to_jax = lambda x: jax.tree_util.tree_map(jnp.asarray, x)
    return (
        to_jax(blob["state"]),
        [to_jax(d) for d in blob["dead"]],
        jnp.asarray(blob["run_key"]),
        int(blob["iteration"]),
    )


class _NSSInternal:
    """Container holding the post-run state of ``nss.ns.run_nested_sampling``.

    The ``NonLinearSearch`` pipeline stores this object as ``search_internal``
    (via ``paths.save_search_internal``) and passes it back into
    ``samples_via_internal_from`` to extract the posterior. Keeping it as a
    plain dataclass-style holder rather than the bare ``nss`` return tuple
    means we can unpickle it later without depending on JAX (the
    ``final_state.particles.position`` array is stored as a NumPy view).
    """

    def __init__(
        self,
        positions: np.ndarray,
        loglikelihoods: np.ndarray,
        log_weights: np.ndarray,
        logZs: np.ndarray,
        wall_time: float,
        sampling_time: float,
        evals: int,
        ess: int,
        n_live: int,
        num_mcmc_steps: int,
        num_delete: int,
        termination: float,
        seed: int,
    ):
        self.positions = positions
        self.loglikelihoods = loglikelihoods
        self.log_weights = log_weights
        self.logZs = logZs
        self.wall_time = wall_time
        self.sampling_time = sampling_time
        self.evals = evals
        self.ess = ess
        self.n_live = n_live
        self.num_mcmc_steps = num_mcmc_steps
        self.num_delete = num_delete
        self.termination = termination
        self.seed = seed


class NSS(abstract_nest.AbstractNest):
    __identifier_fields__ = (
        "n_live",
        "num_mcmc_steps",
        "num_delete",
        "termination",
        "seed",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        n_live: int = 200,
        num_mcmc_steps: int = 5,
        num_delete: int = 50,
        termination: float = -3.0,
        checkpoint_interval: int = 100,
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: Optional[int] = None,
        number_of_cores: int = 1,
        seed: int = 42,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A Nested Slice Sampling non-linear search (JAX-native, ``yallup/nss``).

        ``af.NSS`` wraps ``nss.ns.run_nested_sampling`` into a first-class
        ``NonLinearSearch`` so users can drop ``search = af.NSS(...)`` into any
        production autolens / autogalaxy / autofit script alongside
        ``af.Nautilus(...)``. The likelihood and prior closures both run inside
        ``jax.jit``, leveraging the Phase 0 JAX-native priors (PyAutoFit#1262)
        to make ``model.vector_from_unit_vector`` and
        ``model.log_prior_list_from_vector`` traceable.

        Phases 1-3 of the ``nss_first_class_sampler`` roadmap are live:
        - Phase 1: the wrapper itself (this class).
        - Phase 2: checkpoint/resume via ``checkpoint_interval`` — a
          ``nss_checkpoint.pkl`` is written to ``paths.search_internal_path``
          every N outer iterations and reloaded automatically on resume.
        - Phase 3: on-the-fly visualization via ``iterations_per_quick_update``
          — every N outer iterations the current best live particle is fed to
          ``analysis.visualize`` so partial results appear in the image_path
          directory during long fits.

        Phase 4 (``pip install autofit[nss]`` extra) is still pending — for now
        ``af.NSS`` is an optional requirement and must be installed manually
        via ``pip install git+https://github.com/yallup/nss.git``.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            A unique tag for this model-fit, given a unique entry in the
            sqlite database and used as the folder after the path prefix and
            before the search name.
        n_live
            Number of live particles maintained by NSS. Production-tested
            default of 200 corresponds to FINDINGS_v3's ``c3_big_delete``
            config on the HST MGE problem (recovered ``einstein_radius=1.5996``
            in 7 min wall time).
        num_mcmc_steps
            Number of slice-MCMC inner steps per dead-point batch.
        num_delete
            Number of particles killed per outer iteration. Larger
            ``num_delete`` reduces JIT overhead per iteration at the cost of
            slightly worse posterior coverage.
        termination
            Convergence criterion. The fit stops when
            ``logZ_live - logZ < termination``. Default ``-3.0`` corresponds
            to delta-logZ < 1e-3.
        checkpoint_interval
            Outer iterations between checkpoint writes. Default ``100`` writes
            a ``nss_checkpoint.pkl`` (atomic via tmp-and-rename) every ~5000-
            10000 likelihood evaluations at typical ``num_delete=50``. Set to
            a large value to effectively disable checkpointing on short runs.
        iterations_per_quick_update
            Outer iterations between on-the-fly visualizations. When non-None
            the current best live particle is fed to ``analysis.visualize``
            every N iterations so partial results appear in the image_path
            directory during long fits. ``analysis.visualize`` is wrapped in
            try/except so a viz failure logs a warning but does not kill the
            sampler.
        iterations_per_full_update
            Accepted for API parity. NSS does not have a full-update concept
            separate from the outer-iteration cadence.
        number_of_cores
            Accepted for API parity only. NSS runs on whatever device JAX is
            configured for (CPU, GPU, TPU) — multiprocessing parallelism is
            not used. If ``number_of_cores > 1`` a warning is logged.
        seed
            JAX random seed used to initialise the unit-cube draws for the
            ``n_live`` initial particles and the inner slice-sampling RNG.
        silence
            If True, suppresses NSS's progress bar output.
        session
            An SQLAlchemy session instance so the results of the model-fit
            are written to an SQLite database.

        Notes
        -----
        Sampling space: NSS samples in **physical** parameter space (initial
        particles are drawn via ``model.vector_from_unit_vector`` per unit-cube
        sample, then the slice walks proceed in physical coordinates). The
        Nautilus / PocoMC convention is to sample in unit-cube space and
        apply the prior transform inside the likelihood — that is **not** what
        ``af.NSS`` does. The slice walks have the natural metric of the
        problem and there is no per-step remapping cost.
        """

        if not _HAS_NSS:
            raise ImportError(
                "af.NSS requires the optional `nss` package and the matching "
                "`handley-lab/blackjax` fork. Install via:\n"
                "    pip install autofit[nss]\n"
                "The extra pins specific upstream commits — see PyAutoFit's "
                "pyproject.toml `[project.optional-dependencies] nss` entry."
            )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_full_update=iterations_per_full_update,
            iterations_per_quick_update=iterations_per_quick_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.n_live = n_live
        self.num_mcmc_steps = num_mcmc_steps
        self.num_delete = num_delete
        self.termination = termination
        self.checkpoint_interval = checkpoint_interval
        self.seed = seed

        if number_of_cores is not None and number_of_cores > 1:
            logger.warning(
                "af.NSS received number_of_cores=%d. NSS is JAX-native and "
                "runs on whatever device JAX is configured for; the value is "
                "ignored. Set number_of_cores=1 to silence this warning.",
                number_of_cores,
            )

        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating NSS Search")

    def apply_test_mode(self):
        logger.warning(
            "TEST MODE 1 (reduced iterations): NSS will run with a loose "
            "termination criterion for faster completion."
        )
        self.termination = -1.0

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using NSS, with checkpoint/resume + on-the-fly visualization.

        Builds JAX-traceable ``log_likelihood`` and ``prior_logprob`` closures
        threaded through Phase 0's ``xp=jnp`` plumbing, draws ``n_live`` initial
        particles by mapping unit-cube samples through the prior transform,
        and runs the NSS outer loop inline (mirroring the upstream
        ``nss.ns.run_nested_sampling`` pattern). Between outer iterations the
        loop can (a) pickle resumable state to ``nss_checkpoint.pkl`` and
        (b) call ``analysis.visualize`` on the current best live particle.

        On entry, if a checkpoint exists at the expected path the loop resumes
        from the saved ``(state, dead, run_key, iteration)``. On successful
        exit the checkpoint is deleted — mirrors Nautilus's
        ``output_search_internal`` post-success cleanup so the next fresh fit
        doesn't accidentally resume from a stale checkpoint.

        Returns
        -------
        (search_internal, fitness)
            ``search_internal`` is a ``_NSSInternal`` holder (NumPy arrays
            only). ``fitness`` is a ``Fitness`` instance that ``af.NSS`` does
            not use for sampling (inline JAX closures handle that) but is
            required by ``AbstractNest.perform_update`` for post-fit work
            like latent-sample generation, which calls ``fitness.batch_size``.
        """

        import jax
        import jax.numpy as jnp
        import time

        self.logger.info("Starting NSS non-linear search.")

        ndim = model.prior_count

        def log_likelihood(params):
            instance = model.instance_from_vector(vector=params, xp=jnp)
            raw = analysis.log_likelihood_function(instance=instance)
            return jnp.where(jnp.isfinite(raw), raw, -1e30)

        def prior_logprob(params):
            log_priors = model.log_prior_list_from_vector(vector=params, xp=jnp)
            return sum(log_priors)

        rng_key = jax.random.PRNGKey(self.seed)
        rng_key, init_key, run_key = jax.random.split(rng_key, 3)

        unit_cube = jax.random.uniform(init_key, shape=(self.n_live, ndim))
        initial_samples = jnp.stack(
            [
                model.vector_from_unit_vector(unit_cube[i], xp=jnp)
                for i in range(self.n_live)
            ]
        )

        algo = _blackjax.nss(
            logprior_fn=prior_logprob,
            loglikelihood_fn=log_likelihood,
            num_delete=self.num_delete,
            num_inner_steps=self.num_mcmc_steps,
        )

        @jax.jit
        def one_step(carry, _):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = algo.step(subk, state)
            return (state, k), dead_point

        checkpoint_path = self._nss_checkpoint_path
        if checkpoint_path is not None and checkpoint_path.exists():
            state, dead, run_key, iteration = _load_checkpoint(checkpoint_path)
            self.logger.info(
                "Resuming NSS from checkpoint at iteration %d (state file %s).",
                iteration,
                checkpoint_path,
            )
        else:
            state = algo.init(initial_samples)
            dead = []
            iteration = 0
            self.logger.info(
                "NSS configuration: n_live=%d, num_mcmc_steps=%d, num_delete=%d, "
                "termination=%s, ndim=%d, checkpoint_interval=%d. JIT compile on "
                "first iteration may take 25-30 s.",
                self.n_live,
                self.num_mcmc_steps,
                self.num_delete,
                self.termination,
                ndim,
                self.checkpoint_interval,
            )

        t_start = time.time()
        while not state.integrator.logZ_live - state.integrator.logZ < self.termination:
            (state, run_key), dead_info = one_step((state, run_key), None)
            dead.append(dead_info)
            iteration += 1

            if (
                checkpoint_path is not None
                and iteration % self.checkpoint_interval == 0
            ):
                _save_checkpoint(checkpoint_path, state, dead, run_key, iteration)

            if (
                self.iterations_per_quick_update is not None
                and iteration % self.iterations_per_quick_update == 0
            ):
                self._fire_quick_update(state=state, model=model, analysis=analysis)

        wall_time = time.time() - t_start

        final_state = _nss_finalise(state, dead)

        rng_key, weight_key = jax.random.split(rng_key, 2)
        log_w_mc = _nss_log_weights(weight_key, final_state, shape=100)
        log_w_per_particle = log_w_mc.mean(axis=-1)
        logZs = jax.scipy.special.logsumexp(
            jnp.nan_to_num(log_w_mc, nan=jnp.nan_to_num(log_w_mc).min()),
            axis=0,
        )

        def _safe_ess(log_w_mean):
            log_w_mean = log_w_mean - log_w_mean.max()
            weights = jnp.exp(log_w_mean)
            return float(weights.sum() ** 2 / (weights ** 2).sum())

        ess = int(_safe_ess(log_w_mc.mean(axis=-1)))
        evals = int(
            final_state.update_info.num_steps.sum()
            + final_state.update_info.num_shrink.sum()
        )

        search_internal = _NSSInternal(
            positions=np.asarray(final_state.particles.position),
            loglikelihoods=np.asarray(final_state.particles.loglikelihood),
            log_weights=np.asarray(log_w_per_particle),
            logZs=np.asarray(logZs),
            wall_time=float(wall_time),
            sampling_time=float(wall_time),
            evals=evals,
            ess=ess,
            n_live=int(self.n_live),
            num_mcmc_steps=int(self.num_mcmc_steps),
            num_delete=int(self.num_delete),
            termination=float(self.termination),
            seed=int(self.seed),
        )

        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except OSError as exc:
                self.logger.warning(
                    "Failed to delete completed-run checkpoint %s: %s. The "
                    "next fresh af.NSS fit at this path will attempt to resume "
                    "from it — delete manually if that is not desired.",
                    checkpoint_path,
                    exc,
                )

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=-1.0e99,
            batch_size=1,
        )

        return search_internal, fitness

    @property
    def _nss_checkpoint_path(self) -> Optional[Path]:
        """Resolve the checkpoint location, or None when paths is NullPaths."""
        if isinstance(self.paths, NullPaths):
            return None
        try:
            return Path(self.paths.search_internal_path) / _CHECKPOINT_FILENAME
        except TypeError:
            return None

    def _fire_quick_update(self, state, model, analysis):
        """Push the current best live particle through ``analysis.visualize``.

        The Nautilus / Dynesty quick-update path goes through
        ``Fitness.manage_quick_update``; ``af.NSS`` bypasses ``Fitness._call``
        for sampling so we invoke ``analysis.visualize`` directly between
        outer-loop iterations. Wrapped in try/except — a visualization failure
        logs a warning but does not kill a long sampler run.
        """
        try:
            best_idx = int(np.asarray(state.particles.loglikelihood).argmax())
            best_params = np.asarray(state.particles.position[best_idx]).tolist()
            instance = model.instance_from_vector(vector=best_params)
            analysis.visualize(
                paths=self.paths,
                instance=instance,
                during_analysis=True,
            )
        except Exception as exc:
            self.logger.warning(
                "af.NSS quick-update visualization failed: %s. Continuing the "
                "fit — quick-update is best-effort, the final visualization "
                "fires at the end of the run regardless.",
                exc,
            )

    @property
    def checkpoint_file(self):
        """Path to the on-disk checkpoint written between outer-loop iterations.

        Returns the same value as ``_nss_checkpoint_path`` — exposed as a
        public property for symmetry with ``af.Nautilus.checkpoint_file``.
        """
        return self._nss_checkpoint_path

    def samples_info_from(self, search_internal: Optional[_NSSInternal] = None):
        if search_internal is None:
            search_internal = self.paths.load_search_internal()
        return {
            "log_evidence": float(np.asarray(search_internal.logZs).mean()),
            "log_evidence_error": float(np.asarray(search_internal.logZs).std()),
            "total_samples": int(search_internal.evals),
            "total_accepted_samples": int(len(search_internal.positions)),
            "time": float(search_internal.wall_time),
            "sampling_time": float(search_internal.sampling_time),
            "number_live_points": int(search_internal.n_live),
            "num_mcmc_steps": int(search_internal.num_mcmc_steps),
            "num_delete": int(search_internal.num_delete),
            "termination": float(search_internal.termination),
            "ess": int(search_internal.ess),
        }

    def samples_via_internal_from(
        self,
        model: AbstractPriorModel,
        search_internal: Optional[_NSSInternal] = None,
    ):
        """Convert the stored ``_NSSInternal`` holder into an ``NSSamples``."""

        if search_internal is None:
            search_internal = self.paths.load_search_internal()

        parameter_lists = np.asarray(search_internal.positions).tolist()
        log_likelihood_list = np.asarray(search_internal.loglikelihoods).tolist()

        log_w = np.asarray(search_internal.log_weights)
        log_w_norm = log_w - log_w.max()
        weights = np.exp(log_w_norm)
        weight_total = weights.sum()
        if weight_total > 0:
            weights = weights / weight_total
        weight_list = weights.tolist()

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector))
            for vector in parameter_lists
        ]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return NSSamples(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
        )
