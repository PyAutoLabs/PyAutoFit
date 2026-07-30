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
        iterations_per_log: int = 10,
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
            The number of starts evaluated per compiled ``value_and_grad``
            call. ``None`` (default) evaluates all ``n_starts`` in a single
            ``jax.vmap`` — fastest, but it allocates the whole batched jvp at
            once, which for a memory-heavy likelihood (e.g. a pixelized source
            at 16 starts, ~58 GB in float64) exhausts even an 80 GB GPU.
            Setting it sweeps the starts in ``batch_size``-wide vmapped chunks
            from a Python loop, bounding both the memory footprint (one
            chunk's jvp allocated at a time) and the XLA compile (one
            chunk-shaped program, ever). The compile bound is why the sweep is
            a Python loop and not an in-XLA scan: ``jax.lax.map`` welds the
            whole sweep into one program, which for a multi-band
            ``FactorGraphModel`` objective is intractable to compile (>1 hour
            cold on CPU), while the same-width chunk alone compiles in
            minutes.

            This is purely an **implementation-level tiling**: it is numerically
            inert (identical results, only the allocation and dispatch change).
            That makes it unlike ``af.Nautilus``'s ``n_batch``, which is Nautilus's own
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
        iterations_per_log
            How many steps pass between progress lines on the search log. The
            step loop is otherwise entirely silent between "Starting new ..." and
            "... sampling complete", which on a long fit leaves a user unable to
            tell a live search from a hung one.

            Neither of the framework's usual progress channels reaches this
            search, which is why it needs its own: ``iterations_per_full_update``
            defaults to the never-sentinel, so the whole run is a single chunk
            whose lone ``perform_update`` is (correctly) suppressed as duplicated
            work; and ``Fitness``'s quick-update counter is Python state mutated
            inside ``fitness.call``, which is traced under ``jax.jit(jax.vmap(…))``
            here — it runs once at trace time and never again. The samplers dodge
            the problem entirely by delegating to their own library's progress bar
            (emcee ``progress=True``, dynesty ``print_progress``); a search that
            owns its step loop has to report for itself.

            The first step always logs regardless of this cadence, because that
            line is what tells the user the XLA compile finished and stepping has
            begun — the longest unexplained wait in a real run. A log line rather
            than a progress bar: ``n_steps`` is a ceiling that auto-convergence
            routinely stops short of, so a bar's ETA would mislead, and lines
            survive SLURM/HPC log capture where bars do not.
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

        # Rejected here rather than clamped, and at construction rather than at
        # the first step: a sub-1 cadence makes ``total_steps % cadence`` either
        # raise (0) or log every step (negative, since the modulo is never 0),
        # and a user who mistyped the knob is better served by an immediate error
        # than by a schedule they never asked for. Mirrors the reasoning on
        # ``AbstractSearch._check_step_count``, which it reuses.
        self._check_step_count(iterations_per_log, "iterations_per_log")
        self.iterations_per_log = int(iterations_per_log)

        self.logger.debug(f"Creating {self.optax_method} MultiStartGradient Search")

    def _is_final_boundary(self, converged: bool, total_steps: int) -> bool:
        """
        Whether the chunk boundary just reached ends the search — either the
        auto-convergence check fired, or the ``n_steps`` ceiling was hit.

        ``_fit`` must not emit a ``perform_update`` at a terminal boundary:
        ``start_resume_fit`` performs the final update unconditionally the
        moment ``_fit`` returns, so one here is duplicated work. Flipping
        ``during_analysis`` does not avoid that — ``SearchUpdater.update``
        rebuilds the samples, recomputes the summary and re-runs likelihood
        profiling on every call regardless of the flag.

        A named predicate rather than an inline expression so the rule can be
        tested directly; ``_fit`` itself needs jax + optax + a JAX-traceable
        ``Analysis`` and cannot be driven from the NumPy-only library suite.
        """
        return bool(converged) or total_steps >= self.n_steps

    @staticmethod
    def _stop_reason_on_resume(stop_reason):
        """
        The ``stop_reason`` a resumed run should start from, given the one
        restored from its previous ``search_internal``.

        ``"converged"`` survives: the step loop's guard uses it to refuse
        resuming a converged search into further steps. Every other reason
        describes a run that is now over — keeping it would stamp a stale
        ``"max_steps"`` on every intermediate checkpoint of a search that is in
        fact still running, which is exactly what raising ``n_steps`` (the
        documented way to extend a budget) does. It is re-derived at each chunk
        boundary, so it starts clear.
        """
        return stop_reason if stop_reason == "converged" else None

    def _should_log_progress(self, total_steps: int, converged: bool) -> bool:
        """
        Whether the step just completed should emit a progress line.

        Three ways to qualify: the first step (the compile is over and the search
        is genuinely moving — the single most useful line in the run), every
        ``iterations_per_log``-th step thereafter, and the step that converged
        (which lands off-cadence far more often than not, and is the one step a
        user most wants to see).

        ``silence`` suppresses all of them, matching how the samplers gate their
        own progress output (e.g. Dynesty's ``print_progress=not self.silence``).

        A named predicate rather than an inline condition so the cadence rule can
        be tested directly; ``_fit`` needs jax + optax + a JAX-traceable
        ``Analysis`` and cannot be driven from the NumPy-only library suite.
        """
        if self.silence:
            return False

        return (
            bool(converged)
            or total_steps <= 1
            or total_steps % self.iterations_per_log == 0
        )

    def _compile_message(self, batched: bool) -> str:
        """
        The notice logged immediately before a step that will trigger an XLA
        compile, so the ensuing wait is explained rather than silent.

        Two distinct compiles block a fresh run, hence the ``batched`` switch:
        the single-point ``value_and_grad`` that ``_broad_starts`` calls per draw,
        and the vmapped (or ``batch_size``-chunked) one the step loop calls. Both
        can run to minutes on a multi-band objective, and neither is reachable by
        the ``Fitness._jit`` / ``_vmap`` / ``_grad`` compile notices — this search
        builds its transforms straight off ``fitness.call``.

        Phrased to match the sampler-wide compile message rather than inventing a
        second dialect for the same event.
        """
        if not batched:
            return (
                "JAX: jit compiling the single-point objective used to draw "
                f"{self.n_starts} broad starts, could take seconds or minutes..."
            )

        over = (
            f"batches of {self.batch_size} starts"
            if self.batch_size is not None
            else f"all {self.n_starts} starts at once"
        )

        return (
            f"JAX: jit compiling the vmapped objective over {over}, "
            "could take seconds or minutes..."
        )

    def _progress_message(
        self,
        total_steps: int,
        best_fom: float,
        previous_fom: Optional[float],
        n_alive: int,
        estim_lr=None,
    ) -> str:
        """
        The one-line progress report for a single step.

        Compact and pipe-delimited rather than prose: unlike the one-shot notices
        this line repeats tens of times per run, and sentences would bury the
        numbers that carry the signal.

        Reports the log posterior, not the raw figure of merit. ``best_fom`` is
        ``-2 * log_posterior`` (what the search minimises), so the sign-and-halve
        conversion here is the same one ``samples_via_internal_from`` applies —
        a user comparing the log to their results should not have to do it in
        their head.

        Parameters
        ----------
        total_steps
            Steps completed, reported against the ``n_steps`` ceiling.
        best_fom
            The global best figure-of-merit. Non-finite until some start finds a
            finite basin, which is reported plainly rather than as ``inf``.
        previous_fom
            The global best at the previous progress line, used for the gain
            since. ``None`` on the first line, and skipped while either end of
            the comparison is non-finite (a first finite basin is an arrival, not
            an improvement, and ``inf - x`` would print as ``inf``).
        n_alive
            Starts whose objective is currently finite, against ``n_starts`` —
            the population-health signal that a falling best-fom alone hides.
        estim_lr
            Per-start self-estimated step scale, for the learning-rate-free rules
            that carry one (Prodigy's ``d``). ``None`` for the fixed-rate Adam
            family, whose optimizer state has no such field; the field is then
            omitted rather than rendered empty.
        """
        parts = [f"{self.optax_method} step {total_steps}/{self.n_steps}"]

        if np.isfinite(best_fom):
            parts.append(f"best log_post {-0.5 * best_fom:.4f}")

            if previous_fom is not None and np.isfinite(previous_fom):
                # fom is -2*log_posterior, so a fom drop of d is a log-posterior
                # gain of d/2.
                parts.append(f"gained {0.5 * (previous_fom - best_fom):.4f}")
        else:
            parts.append("best log_post — (no finite basin yet)")

        parts.append(f"alive {n_alive}/{self.n_starts}")

        if estim_lr is not None:
            lr = np.asarray(estim_lr, dtype=float)
            parts.append(
                f"d {np.nanmin(lr):.2e} / {np.nanmedian(lr):.2e} / {np.nanmax(lr):.2e}"
            )

        return " | ".join(parts)

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
        # 80 GB GPU. When `batch_size` is set we sweep the starts in
        # `batch_size`-wide vmapped chunks from a Python loop. Only one
        # chunk-shaped function is ever XLA-compiled (the jit cache keys on the
        # `(batch_size, ndim)` shape; the ragged final chunk is padded to that
        # shape and the padded rows discarded), so the compile cost is that of
        # a single chunk regardless of `n_starts`. An in-XLA scan over the
        # chunks (`jax.lax.map`) instead welds the whole sweep into one
        # program, which for a multi-band `FactorGraphModel` objective is
        # intractable to compile (>1 hour cold on CPU, and memory-explosive to
        # compile) while the chunk alone compiles in minutes. The tiling is
        # numerically identical to the vmap; `batch_size` never changes
        # results, it only bounds memory and compile.
        _value_and_grad = jax.value_and_grad(fitness.call)
        _vmapped = jax.jit(jax.vmap(_value_and_grad))

        if self.batch_size is None:
            batched_value_and_grad = _vmapped
        else:
            batch_size = self.batch_size

            def batched_value_and_grad(params):
                foms_chunks = []
                grads_chunks = []
                for lo, hi, pad in _chunk_slices(params.shape[0], batch_size):
                    chunk = params[lo:hi]
                    if pad:
                        chunk = jnp.concatenate(
                            [chunk, jnp.tile(chunk[-1:], (pad, 1))]
                        )
                    foms, grads = _vmapped(chunk)
                    if pad:
                        foms = foms[:-pad]
                        grads = grads[:-pad]
                    foms_chunks.append(foms)
                    grads_chunks.append(grads)
                return (
                    jnp.concatenate(foms_chunks),
                    jnp.concatenate(grads_chunks),
                )

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

            if not self.silence:
                self.logger.info(self._compile_message(batched=False))

            params = self._broad_starts(
                model=model,
                value_and_grad_single=jax.jit(_value_and_grad),
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

        stop_reason = self._stop_reason_on_resume(stop_reason)

        # The vmapped objective compiles on its first call below, not here, so
        # the notice is emitted at the call site the first time round. Resumed
        # runs pay it too (a fresh process has no live trace cache), which is why
        # the flag is set here rather than only on the fresh path.
        awaiting_compile = True

        # The global best at the previous progress line, so each line can report
        # the gain since the last one rather than since the previous *step* —
        # over a cadence of ``iterations_per_log`` steps that is the number a
        # user can actually act on.
        previous_logged_fom = None

        # ``n_steps`` is the hard ceiling / max budget; ``stop_reason`` becomes
        # ``"converged"`` if the auto-convergence check stops the search early
        # (this also short-circuits the loop on a resumed, already-converged run).
        while total_steps < self.n_steps and stop_reason != "converged":

            steps_remaining = self.n_steps - total_steps
            iterations = self._steps_until_full_update(steps_remaining)

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
                if awaiting_compile:
                    if not self.silence:
                        self.logger.info(self._compile_message(batched=True))
                    awaiting_compile = False

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

                # Logged before the ``converged`` break so the terminating step
                # reports itself; it is the one step a user most wants to see and
                # it lands off-cadence more often than not.
                if self._should_log_progress(
                    total_steps=total_steps, converged=converged
                ):
                    # Prodigy et al. carry a self-estimated step scale; the Adam
                    # family's state has no such field and ``tree_get`` returns
                    # None, which the message renders by omission. Read only on a
                    # logging step: it is an (n_starts,) device->host copy, which
                    # merely pulls forward the sync the next iteration's
                    # ``np.asarray(foms)`` would force anyway.
                    estim_lr = optax.tree_utils.tree_get(opt_state, "estim_lr")

                    self.logger.info(
                        self._progress_message(
                            total_steps=total_steps,
                            best_fom=best_fom,
                            previous_fom=previous_logged_fom,
                            n_alive=int(alive.sum()),
                            estim_lr=estim_lr,
                        )
                    )
                    previous_logged_fom = best_fom

                if converged:
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

            # Update only at *intermediate* boundaries. ``start_resume_fit``
            # performs the final update unconditionally the moment ``_fit``
            # returns, so any update emitted here at a terminal boundary is pure
            # duplicated work — and flipping ``during_analysis`` does not avoid
            # it: ``SearchUpdater.update`` rebuilds the samples, recomputes the
            # summary and re-runs likelihood profiling on *every* call,
            # regardless of the flag. The previous ``during_analysis=not
            # is_final`` was worse still, running the whole final pass twice
            # including final visualization and latents.
            #
            # Nothing is lost by skipping: the checkpoint is
            # ``save_search_internal`` above, and the ``search_internal`` dict
            # this loop just built (carrying ``stop_reason``, ``converged`` and
            # ``fom_history``) is what ``_fit`` returns and what that final
            # update is computed from.
            if not self._is_final_boundary(converged=converged, total_steps=total_steps):
                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
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

    def _broad_starts(self, model, value_and_grad_single, jnp):
        """
        Draw ``n_starts`` broad starting points in the unit cube, map them to
        physical parameters, and keep only those with a finite objective and a
        finite gradient (degenerate points such as ell_comps / shear at exactly 0
        have NaN gradients and must be filtered out).

        ``value_and_grad_single`` is the jitted single-point objective built
        once in ``_fit``: one XLA compile (persistently cached) serves every
        draw. Evaluating the draws eagerly instead re-pays the un-compiled AD
        cost per draw, which on a multi-band ``FactorGraphModel`` objective
        dominated the whole fit (~13 minutes for 16 draws, cache or no cache).
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
            fom, grad = value_and_grad_single(vector)
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
        # ``stop_reason`` is persisted by the fit loop ("converged" | "max_steps");
        # older search_internal files (pre-auto-convergence) have neither it nor a
        # ``fom_history``, so both are read defensively.
        stop_reason = search_internal.get("stop_reason")
        fom_history = search_internal.get("fom_history")

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
            # Auto-convergence outcome: whether the run stopped on the plateau
            # check ("converged") or exhausted the ``n_steps`` ceiling
            # ("max_steps"), the settings that produced it, and the global-best
            # figure-of-merit trace so the plateau can be inspected downstream.
            "stop_reason": stop_reason,
            "converged": stop_reason == "converged",
            "convergence": {
                "check_for_convergence": self.convergence.check_for_convergence,
                "window": self.convergence.window,
                "rtol": self.convergence.rtol,
                "atol": self.convergence.atol,
                "min_steps": self.convergence.min_steps,
            },
            "fom_history": (
                [float(x) for x in np.asarray(fom_history)]
                if fom_history is not None
                else None
            ),
            "time": self.timer.time if self.timer else None,
        }

        return Samples(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )


def _chunk_slices(n_rows, batch_size):
    """
    The ``(lo, hi, pad)`` chunk bounds the batched ``value_and_grad`` sweep
    iterates over: rows ``lo:hi`` of the params array, padded by ``pad`` repeats
    of the final row so every chunk presents the same ``(batch_size, ndim)``
    shape to the compiled function (one XLA compile for the whole sweep). Only
    the final chunk can be ragged (``pad > 0``) — the broad-start collection may
    return fewer than ``n_starts`` rows, so raggedness is not restricted to
    ``n_starts % batch_size``.
    """
    return [
        (lo, min(lo + batch_size, n_rows), max(0, lo + batch_size - n_rows))
        for lo in range(0, n_rows, batch_size)
    ]


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
