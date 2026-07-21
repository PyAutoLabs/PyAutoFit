import inspect
from typing import Optional

import numpy as np

from autofit.database.sqlalchemy_ import sa

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.mle.abstract_mle import AbstractMLE
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.search.mle.multi_start_gradient.convergence import (
    MultiStartGradientConvergence,
)


class AbstractMultiStartGradient(AbstractMLE):

    # Name of the optax update rule, resolved lazily at fit time from ``optax``
    # then ``optax.contrib`` (so ``optax`` is only imported when a fit is
    # actually run — it is a JAX-only optional dependency). Subclasses set this
    # and a default learning rate for the rule; a ``_default_learning_rate`` of
    # ``None`` means the rule is learning-rate-free (e.g. Prodigy) and is built
    # from its own default with no learning rate supplied.
    optax_method = None
    _default_learning_rate = None

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        n_starts: int = 48,
        n_steps: int = 300,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_consecutive_nan: int = 8,
        start_lower_limit: float = 0.15,
        start_upper_limit: float = 0.85,
        resurrect: bool = False,
        convergence: Optional[MultiStartGradientConvergence] = None,
        initializer: Optional[AbstractInitializer] = None,
        iterations_per_full_update: int = None,
        iterations_per_quick_update: int = None,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A multi-start first-order gradient MAP optimizer (the "GIGA-Lens" recipe).

        This search runs ``n_starts`` independent optimizations from broad,
        randomly drawn starting points, all in parallel via ``jax.vmap``, using a
        fixed self-normalised optax update rule (Adam / ADABelief / Lion) on the
        unconstrained (unit-cube) parameterization. A wide population of starts is
        what lets the method escape the many wrong basins that trap every
        single-start, line-search or second-order optimizer on the kinked
        likelihoods this promotes from (see the Phase-3 GPU MAP-optimizer
        benchmark). The best-basin start is returned as the maximum-log-posterior
        (MAP) point, with every start's final point retained as a diagnostic.

        The method is JAX-native: it requires an ``Analysis`` whose
        ``log_likelihood_function`` is JAX-traceable (``use_jax=True``) and the
        optional ``jax`` + ``optax`` dependencies.

        Parameters
        ----------
        n_starts
            The number of independent broad starts run in parallel (vmapped).
        n_steps
            The number of gradient-update steps each start is run for.
        learning_rate
            The optax learning rate. If ``None``, the rule's default is used
            (Adam / ADABelief ``1e-2``; the sign-based Lion ``1e-3``). The
            learning-rate-free rules (e.g. Prodigy) leave this ``None`` and are
            built from their own default, estimating their own step scale.
        max_consecutive_nan
            The per-start rejected-step budget handed to ``optax.apply_if_finite``:
            a non-finite gradient/update zeroes that start's step, and only after
            this many *consecutive* non-finite steps does the guard error out.
            This is the in-step guard for the measure-zero singularities (e.g.
            ell_comps / shear at exactly 0); it does not rescue landscapes with
            broad non-finite regions (that is the Phase-2 restart-on-death layer).
        batch_size
            The number of starts evaluated per vmapped ``value_and_grad`` call,
            via ``jax.lax.map``. ``None`` (default) evaluates all ``n_starts`` in
            a single ``jax.vmap`` — fastest, but it allocates the whole batched
            jvp at once, which for a memory-heavy likelihood (e.g. a pixelized
            source at 16 starts, ~58 GB in float64) exhausts even an 80 GB GPU.
            Setting it trades a little speed for a bounded memory footprint.

            This is purely an **implementation-level tiling**: it is numerically
            inert (identical results, only the allocation changes). That makes it
            unlike ``af.Nautilus``'s ``n_batch``, which is Nautilus's own
            algorithmic knob (how many points it proposes per iteration) that
            autofit merely forwards. Here ``n_starts`` is the algorithm; this
            only decides how many of those starts are evaluated at a time.
        start_lower_limit, start_upper_limit
            The unit-cube bounds broad starts are drawn uniformly from. The
            interior default ``(0.15, 0.85)`` avoids the prior edges where many
            transforms (e.g. ``arctan2`` / ``sqrt`` at exactly 0) are singular.
        resurrect
            Restart-on-death. When ``True``, any start whose objective goes
            non-finite is redrawn each step (fresh params from the start band +
            its per-start optimizer state reinitialised), leaving alive starts
            untouched. Default ``False`` — the parametric (MGE-class) cell has
            only the measure-zero singularity, so the ``apply_if_finite`` guard
            suffices and behaviour/results are unchanged. Turn it on for
            likelihoods with broad non-finite regions (pixelized sources), where
            every trajectory otherwise walks into a wall within ~25–50 steps and
            ``apply_if_finite`` alone latches the start *at* the cliff edge;
            resurrection keeps the population alive and makes the landscape
            searchable at all. (Even so, on such landscapes a nested sampler
            still wins decisively — resurrection makes gradient MAP *viable*
            there, not competitive.)
        convergence
            Auto-convergence (early-stopping) settings. When
            ``check_for_convergence`` is ``True`` (the default) the search stops
            early once the global-best figure-of-merit has plateaued, so users do
            not have to hand-tune ``n_steps`` — which remains a hard ceiling / max
            budget (the search never runs forever). ``None`` builds the default
            ``MultiStartGradientConvergence()``. The check is deliberately skipped
            when ``resurrect=True`` (the pixelized regime, whose best-fom climbs in
            breakthrough jumps that a plateau check would false-stop on); there the
            search leans on the ``n_steps`` ceiling.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.n_starts = n_starts
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.max_consecutive_nan = max_consecutive_nan
        self.learning_rate = (
            learning_rate if learning_rate is not None else self._default_learning_rate
        )
        self.start_lower_limit = start_lower_limit
        self.start_upper_limit = start_upper_limit
        self.resurrect = resurrect
        self.convergence = (
            convergence if convergence is not None else MultiStartGradientConvergence()
        )

        self.logger.debug(f"Creating {self.optax_method} MultiStartGradient Search")

    def _fit(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
    ):
        """
        Fit a model by running ``n_starts`` broad optax optimizations in parallel
        (vmapped) and returning the best-basin (maximum-log-posterior) start.

        The objective minimized is ``Fitness.call`` with
        ``fom_is_log_likelihood=False`` and ``convert_to_chi_squared=True``, which
        returns ``-2 * log_posterior``; invalid / NaN models map to ``+inf`` so
        they are never selected as the best basin.
        """
        try:
            import jax
            import jax.numpy as jnp
            import optax
            import optax.contrib  # noqa: F401  — makes optax.contrib rules resolvable
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} requires the optional `jax` and `optax` "
                "dependencies. Install them with `pip install autofit[jax] optax`."
            ) from e

        if not getattr(analysis, "_use_jax", False):
            raise ValueError(
                f"{type(self).__name__} is a JAX-native gradient search and "
                "requires a JAX-traceable Analysis (e.g. `AnalysisImaging(..., "
                "use_jax=True)`). The supplied analysis is not running on the JAX "
                "backend."
            )

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=True,
        )

        # -2 * log_posterior, to MINIMIZE. value_and_grad over every start.
        #
        # Unbatched we vmap all starts at once — fastest, but it allocates the
        # whole batched jvp, which for a memory-heavy likelihood (e.g. a
        # pixelized source at 16 starts, ~58 GB in float64) exhausts even an
        # 80 GB GPU. When `batch_size` is set we hand the tiling to
        # `jax.lax.map`, which vmaps *within* each chunk and scans across them,
        # handling a ragged final chunk without a second compile. It is
        # numerically identical to the vmap; `batch_size` never changes results,
        # it only bounds memory.
        #
        # `lax.map` is only used when batching is requested: with
        # `batch_size=None` it degrades to a sequential scan, which would throw
        # away the parallelism of the default path.
        _value_and_grad = jax.value_and_grad(fitness.call)
        _vmapped = jax.jit(jax.vmap(_value_and_grad))

        if self.batch_size is None:
            batched_value_and_grad = _vmapped
        else:
            batch_size = self.batch_size

            @jax.jit
            def batched_value_and_grad(params):
                return jax.lax.map(_value_and_grad, params, batch_size=batch_size)

        # The optax rule (resolved from optax / optax.contrib), guarded by
        # apply_if_finite, with a jitted per-start (vmapped) update step. Built
        # once, identically, on both the fresh and resume paths so a loaded
        # opt_state (itself a vmapped pytree) round-trips.
        optimizer, step_update = self._build_optimizer(optax=optax, jax=jax)

        try:
            search_internal = self.paths.load_search_internal()

            params = jnp.asarray(search_internal["params"])
            opt_state = optax.tree_utils.tree_get(search_internal, "opt_state")
            best_params = np.asarray(search_internal["best_params"])
            best_fom = float(search_internal["best_fom"])
            fom_history = list(search_internal["fom_history"])
            total_steps = int(search_internal["total_steps"])
            n_resurrections = int(search_internal.get("n_resurrections", 0))
            stop_reason = search_internal.get("stop_reason", None)

            self.logger.info(
                "Resuming MultiStartGradient search (previous samples found)."
            )

        except (FileNotFoundError, TypeError, KeyError):

            params = self._broad_starts(
                model=model,
                fitness=fitness,
                batched_value_and_grad=batched_value_and_grad,
                jnp=jnp,
            )

            # Per-start optimizer state: one independent state per start, so
            # learning-rate-free rules never share a global scalar estimate.
            opt_state = jax.vmap(optimizer.init)(params)

            best_params = np.asarray(params[0])
            best_fom = np.inf
            fom_history = []
            total_steps = 0
            n_resurrections = 0
            stop_reason = None

            self.logger.info(
                f"Starting new {self.optax_method} MultiStartGradient search "
                f"({self.n_starts} starts, no previous samples found)."
            )

        # Deterministic RNG for redrawing dead starts (only used when
        # ``resurrect`` is on); seeded independently of the broad-start draw.
        resurrect_rng = np.random.default_rng(1)

        # ``n_steps`` is the hard ceiling / max budget; ``stop_reason`` becomes
        # ``"converged"`` if the auto-convergence check stops the search early
        # (this also short-circuits the loop on a resumed, already-converged run).
        while total_steps < self.n_steps and stop_reason != "converged":

            steps_remaining = self.n_steps - total_steps
            iterations = min(
                self.iterations_per_full_update or self.n_steps, steps_remaining
            )

            # Convergence is assessed every step (``fom_history`` updates every
            # step) rather than only at the ``iterations_per_full_update`` boundary,
            # so early-stopping works even with the default single-chunk update
            # cadence (``iterations_per_full_update=None``). Checkpointing and the
            # expensive ``perform_update`` still happen only at the boundary (or on
            # convergence), so the extra work per step is a cheap NumPy plateau
            # check. The check is skipped when ``resurrect`` is on (pixelized
            # regime), where a plateau does not mean converged.
            converged = False

            for _ in range(iterations):
                foms, grads = batched_value_and_grad(params)

                alive = np.isfinite(np.asarray(foms))
                foms_np = np.where(alive, np.asarray(foms), np.inf)
                best_index = int(np.argmin(foms_np))
                if foms_np[best_index] < best_fom:
                    best_fom = float(foms_np[best_index])
                    best_params = np.asarray(params[best_index])

                fom_history.append(best_fom)

                # Restart-on-death: redraw any start whose objective went
                # non-finite (fresh params + reinitialised per-start optimizer
                # state), leaving alive starts untouched. best_* is captured
                # above, from the pre-redraw alive population. The (old,
                # non-finite) grads of a dead start are zeroed for one step by
                # apply_if_finite on its fresh state, so a redrawn start simply
                # waits a step before descending.
                if self.resurrect and not alive.all():
                    dead_idx = np.flatnonzero(~alive)
                    n_resurrections += int(dead_idx.size)
                    params, opt_state = self._reinit_dead_starts(
                        params=params,
                        opt_state=opt_state,
                        dead_idx=dead_idx,
                        model=model,
                        optimizer=optimizer,
                        jax=jax,
                        jnp=jnp,
                        rng=resurrect_rng,
                    )

                updates, opt_state = step_update(grads, opt_state, params, foms)
                params = optax.apply_updates(params, updates)

                total_steps += 1

                if not self.resurrect and self.convergence.check_if_converged(
                    fom_history
                ):
                    converged = True
                    break

            if converged:
                stop_reason = "converged"
            elif total_steps >= self.n_steps:
                stop_reason = "max_steps"

            search_internal = {
                "params": np.asarray(params),
                "opt_state": opt_state,
                "best_params": best_params,
                "best_fom": best_fom,
                "fom_history": np.asarray(fom_history),
                "total_steps": total_steps,
                "n_resurrections": n_resurrections,
                "stop_reason": stop_reason,
            }
            self.paths.save_search_internal(obj=search_internal)

            # A converged (or ceiling-reached) boundary is the final update, so it
            # runs with ``during_analysis=False``; intermediate boundaries do not.
            is_final = converged or total_steps >= self.n_steps
            self.perform_update(
                model=model,
                analysis=analysis,
                during_analysis=not is_final,
                fitness=fitness,
                search_internal=search_internal,
            )

            if converged:
                self.logger.info(
                    f"{self.optax_method} MultiStartGradient converged early at "
                    f"step {total_steps} (best figure-of-merit plateaued)."
                )

        self.logger.info(f"{self.optax_method} MultiStartGradient sampling complete.")

        return search_internal, fitness

    def _resolve_optax_rule(self, optax):
        """
        Resolve ``self.optax_method`` to an optax update-rule factory, looked up
        first in ``optax`` (the built-in Adam family) and then in
        ``optax.contrib`` (the learning-rate-free / experimental rules such as
        ``prodigy``). Raises if neither module provides it.
        """
        rule = getattr(optax, self.optax_method, None)
        if rule is None:
            rule = getattr(optax.contrib, self.optax_method, None)
        if rule is None:
            raise ValueError(
                f"{type(self).__name__}: optax update rule "
                f"'{self.optax_method}' was not found in `optax` or "
                "`optax.contrib`. Check the `optax_method` name and that the "
                "installed optax version provides it."
            )
        return rule

    def _build_optimizer(self, optax, jax):
        """
        Build the guarded optimizer and its jitted per-start update step.

        Per-start ``jax.vmap`` over ``init`` / ``update`` is load-bearing: the
        learning-rate-free rules estimate a *global scalar* step scale from
        whole-tree norms (Prodigy / D-Adapt ``d``, DoG ``max_dist``, Mechanic's
        scale, MoMo's Polyak step). The stacked-``(n_starts, ndim)`` state that
        is safe for the elementwise Adam family would silently couple every
        start into one shared estimate, so each start carries its own state.

        ``optax.apply_if_finite`` is the per-start in-step guard: a non-finite
        gradient/update zeroes that start's step rather than NaN-poisoning the
        whole population. MoMo-family rules additionally consume the loss each
        step via ``value=``; this is detected on the *unwrapped* rule, since
        ``apply_if_finite`` hides ``value`` behind ``**extra_args`` (it does
        forward it, for optax >= 0.2.5).
        """
        rule = self._resolve_optax_rule(optax)
        base = rule() if self.learning_rate is None else rule(self.learning_rate)

        needs_value = "value" in inspect.signature(base.update).parameters

        optimizer = optax.apply_if_finite(
            base, max_consecutive_errors=self.max_consecutive_nan
        )

        if needs_value:

            @jax.jit
            def step_update(grads, opt_state, params, values):
                return jax.vmap(lambda g, s, p, v: optimizer.update(g, s, p, value=v))(
                    grads, opt_state, params, values
                )

        else:

            @jax.jit
            def step_update(grads, opt_state, params, values):
                del values
                return jax.vmap(optimizer.update)(grads, opt_state, params)

        return optimizer, step_update

    def _reinit_dead_starts(
        self, params, opt_state, dead_idx, model, optimizer, jax, jnp, rng
    ):
        """
        Redraw the dead starts (``dead_idx``) and reinitialise their per-start
        optimizer state, leaving the alive starts untouched — the restart-on-
        death recovery step.

        Each dead row's params are redrawn uniformly from the start band and
        mapped to physical parameters; a fresh vmapped optimizer state is built
        and merged into the live state pytree with a boolean mask (``jnp.where``
        per leaf). ``np.asarray`` of a JAX array is read-only, so the redraw
        happens in an ``np.array`` copy.
        """
        n = params.shape[0]

        params_np = np.array(params)  # writable copy (jax arrays are read-only)
        for k in dead_idx:
            unit_vector = rng.uniform(
                self.start_lower_limit, self.start_upper_limit, size=model.prior_count
            )
            params_np[k] = np.asarray(
                model.vector_from_unit_vector(unit_vector=list(unit_vector), xp=jnp)
            )
        params = jnp.asarray(params_np)

        fresh_state = jax.vmap(optimizer.init)(params)

        mask = np.zeros(n, dtype=bool)
        mask[dead_idx] = True
        mask_j = jnp.asarray(mask)

        def merge(old, fresh):
            m = mask_j.reshape((n,) + (1,) * (fresh.ndim - 1))
            return jnp.where(m, fresh, old)

        opt_state = jax.tree.map(merge, opt_state, fresh_state)

        return params, opt_state

    def _broad_starts(self, model, fitness, batched_value_and_grad, jnp):
        """
        Draw ``n_starts`` broad starting points in the unit cube, map them to
        physical parameters, and keep only those with a finite objective and a
        finite gradient (degenerate points such as ell_comps / shear at exactly 0
        have NaN gradients and must be filtered out).
        """
        rng = np.random.default_rng(0)

        starts = []
        max_tries = self.n_starts * 30
        tries = 0
        while len(starts) < self.n_starts and tries < max_tries:
            tries += 1
            unit_vector = rng.uniform(
                self.start_lower_limit, self.start_upper_limit, size=model.prior_count
            )
            vector = jnp.asarray(
                model.vector_from_unit_vector(unit_vector=list(unit_vector), xp=jnp)
            )
            fom, grad = jax_value_and_grad_single(fitness, vector)
            if np.isfinite(float(fom)) and np.all(np.isfinite(np.asarray(grad))):
                starts.append(vector)

        if len(starts) == 0:
            raise ValueError(
                f"{type(self).__name__} could not draw any finite-gradient starting "
                f"points in {tries} attempts. Check the model / analysis are "
                "JAX-traceable and the prior ranges are not everywhere singular."
            )

        if len(starts) < self.n_starts:
            self.logger.warning(
                f"Only collected {len(starts)}/{self.n_starts} finite-gradient "
                f"starts (from {tries} draws)."
            )

        return jnp.stack(starts)

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        """
        Returns a `Samples` object from the MultiStartGradient internal results.

        The best-basin (maximum-log-posterior) start is the first sample; every
        start's final point is retained as a diagnostic sample so per-start basin
        spread can be inspected downstream.
        """
        if search_internal is None:
            search_internal = self.paths.load_search_internal()

        best_params = np.asarray(search_internal["best_params"])
        per_start_params = np.asarray(search_internal["params"])
        total_steps = int(search_internal["total_steps"])

        parameter_lists = [list(best_params)] + [list(p) for p in per_start_params]

        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)

        # Fitness.call returns -2 * log_posterior, so log_posterior = -0.5 * fom.
        best_log_posterior = -0.5 * float(search_internal["best_fom"])
        log_likelihood_list = [best_log_posterior - log_prior_list[0]]
        log_likelihood_list += [np.nan for _ in range(len(parameter_lists) - 1)]

        weight_list = [1.0] + [0.0] * (len(parameter_lists) - 1)

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        samples_info = {
            "n_starts": self.n_starts,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "total_steps": total_steps,
            "optax_method": self.optax_method,
            "learning_rate": self.learning_rate,
            "max_consecutive_nan": self.max_consecutive_nan,
            "resurrect": self.resurrect,
            "n_resurrections": int(search_internal.get("n_resurrections", 0)),
            "time": self.timer.time if self.timer else None,
        }

        return Samples(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )


def jax_value_and_grad_single(fitness, vector):
    """
    Single-point ``value_and_grad`` of the fitness objective, used to filter
    broad starts down to those with a finite value and gradient before the
    batched loop begins.
    """
    import jax

    return jax.value_and_grad(fitness.call)(vector)


class MultiStartAdam(AbstractMultiStartGradient):
    """
    Multi-start gradient MAP search using the Adam optax update rule.

    Adam was the certified best method in the Phase-3 GPU MAP-optimizer
    benchmark — no line-search or second-order optimizer beat wide multi-start
    Adam on the kinked (NNLS active-set) lens likelihood.
    """

    optax_method = "adam"
    _default_learning_rate = 1.0e-2


class MultiStartADABelief(AbstractMultiStartGradient):
    """
    Multi-start gradient MAP search using the ADABelief optax update rule, which
    tied Adam for best in the Phase-3 benchmark.
    """

    optax_method = "adabelief"
    _default_learning_rate = 1.0e-2


class MultiStartLion(AbstractMultiStartGradient):
    """
    Multi-start gradient MAP search using the Lion optax update rule. Lion is
    sign-based, so it wants a ~10x smaller learning rate than Adam / ADABelief.
    """

    optax_method = "lion"
    _default_learning_rate = 1.0e-3


class MultiStartProdigy(AbstractMultiStartGradient):
    """
    Multi-start gradient MAP search using the learning-rate-free Prodigy update
    rule (``optax.contrib.prodigy``).

    Prodigy estimates its own step scale, so it takes **no** learning rate
    (``_default_learning_rate = None``, resolved from ``optax.contrib``). On the
    parametric MGE cell it is bit-identical to hand-tuned Adam (best log
    posterior +31787.84, same winning start) with the ``learning_rate``
    hyperparameter deleted at zero cost — the headline learning-rate-free result
    from the #101 experiment. Its global distance estimate ``d`` is per-start
    (the base class vmaps optimizer state over starts), so starts do not share
    one estimate.
    """

    optax_method = "prodigy"
    _default_learning_rate = None
