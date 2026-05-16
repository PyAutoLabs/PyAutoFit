import logging
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
    from nss.ns import run_nested_sampling, log_weights as _nss_log_weights

    _HAS_NSS = True
except ImportError:
    run_nested_sampling = None
    _nss_log_weights = None
    _HAS_NSS = False


logger = logging.getLogger(__name__)


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

        Phase 1 of the ``nss_first_class_sampler`` roadmap. Checkpointing /
        resumption (Phase 2) and on-the-fly visualization (Phase 3) are stubbed
        — kwargs are accepted but log a warning when set, and a state file at
        ``paths.search_internal_path / state.json`` triggers a warning that
        resume is not yet supported (the fit then proceeds from scratch).

        ``af.NSS`` is an optional requirement and must be installed manually
        via ``pip install git+https://github.com/yallup/nss.git`` (Phase 4 of
        the roadmap will ship a ``pyautofit[nss]`` extra).

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
        iterations_per_quick_update
            Accepted for API parity with other nested samplers. **Not yet
            wired** — quick-update visualization is Phase 3.
        iterations_per_full_update
            Accepted for API parity. NSS performs its own internal output
            cadence and does not honour intermediate full updates in Phase 1.
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
                "af.NSS requires the optional `nss` package. Install via\n"
                "    pip install git+https://github.com/yallup/nss.git\n"
                "(Phase 4 of the nss_first_class_sampler roadmap will ship a\n"
                "`pip install autofit[nss]` extra — track the progress in\n"
                "PyAutoPrompt/autofit/nss_install_simplification.md.)"
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
        self.seed = seed

        if number_of_cores is not None and number_of_cores > 1:
            logger.warning(
                "af.NSS received number_of_cores=%d. NSS is JAX-native and "
                "runs on whatever device JAX is configured for; the value is "
                "ignored. Set number_of_cores=1 to silence this warning.",
                number_of_cores,
            )

        if iterations_per_quick_update is not None:
            logger.info(
                "af.NSS received iterations_per_quick_update=%s. Quick-update "
                "visualization is Phase 3 of the nss_first_class_sampler "
                "roadmap and is not yet wired up; the kwarg is currently a "
                "no-op.",
                iterations_per_quick_update,
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
        Fit a model using NSS.

        Builds JAX-traceable ``log_likelihood`` and ``prior_logprob`` closures
        threaded through Phase 0's ``xp=jnp`` plumbing, draws ``n_live``
        initial particles by mapping unit-cube samples through the prior
        transform, then calls ``nss.ns.run_nested_sampling``. The returned
        ``final_state`` and ``results`` are repackaged into a ``_NSSInternal``
        holder (NumPy arrays only) so the standard PyAutoFit pickled-search
        path keeps working.

        Returns
        -------
        (search_internal, fitness)
            ``search_internal`` is a ``_NSSInternal`` holder. ``fitness`` is a
            ``Fitness`` instance that is **not** used by ``af.NSS`` for
            sampling (the JAX likelihood + prior closures are built inline
            and passed straight to ``nss.ns.run_nested_sampling``) but is
            required by ``AbstractNest.perform_update`` for post-fit work
            like latent-sample generation, which calls ``fitness.batch_size``.
        """

        import jax
        import jax.numpy as jnp
        import time

        if not isinstance(self.paths, NullPaths):
            state_file = Path(self.paths.search_internal_path) / "state.json"
            if state_file.exists():
                self.logger.warning(
                    "Detected %s — resume is Phase 2 of the "
                    "nss_first_class_sampler roadmap and is not yet wired up. "
                    "Proceeding with a fresh fit.",
                    state_file,
                )

        self.logger.info("Starting new NSS non-linear search.")

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

        self.logger.info(
            "NSS configuration: n_live=%d, num_mcmc_steps=%d, num_delete=%d, "
            "termination=%s, ndim=%d. JIT compile on first iteration may "
            "take 25-30 s.",
            self.n_live,
            self.num_mcmc_steps,
            self.num_delete,
            self.termination,
            ndim,
        )

        t_start = time.time()
        final_state, results = run_nested_sampling(
            run_key,
            loglikelihood_fn=log_likelihood,
            prior_logprob=prior_logprob,
            num_mcmc_steps=self.num_mcmc_steps,
            initial_samples=initial_samples,
            num_delete=self.num_delete,
            termination=self.termination,
        )
        wall_time = time.time() - t_start

        rng_key, weight_key = jax.random.split(rng_key, 2)
        log_w_mc = _nss_log_weights(weight_key, final_state, shape=100)
        log_w_per_particle = log_w_mc.mean(axis=-1)

        search_internal = _NSSInternal(
            positions=np.asarray(final_state.particles.position),
            loglikelihoods=np.asarray(final_state.particles.loglikelihood),
            log_weights=np.asarray(log_w_per_particle),
            logZs=np.asarray(results.logZs),
            wall_time=float(wall_time),
            sampling_time=float(results.time),
            evals=int(results.evals),
            ess=int(results.ess),
            n_live=int(self.n_live),
            num_mcmc_steps=int(self.num_mcmc_steps),
            num_delete=int(self.num_delete),
            termination=float(self.termination),
            seed=int(self.seed),
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
    def checkpoint_file(self):
        """Path to the checkpoint file used by Phase 2's resume hook.

        Phase 1 only checks for existence to warn the user that resume is not
        yet supported. Phase 2 will use this path to write incremental state.
        """
        try:
            return self.paths.search_internal_path / "state.json"
        except TypeError:
            return None

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
