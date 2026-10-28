"""
The minimal ``run(ctx)`` bridge (search-extensibility phase A2, decision D7).

``docs/design/run_ctx.md`` (frozen) fixes the ``run(ctx)`` signature and the members of
``FitContext``. A search that implements ``run(ctx)`` (and ``raw_samples_from``) instead
of ``_fit`` is driven through a ``FitContext`` by ``NonLinearSearch._fit``, the bridge.
``Drawer`` and ``Nautilus`` were the first two searches on it (A2); since A5 every
PyAutoFit search is. An external search that still overrides ``_fit`` keeps working (it
takes its pools from ``self._pools(analysis)``).

Members filled by later phases:

- ``ctx.checkpointer`` is the search's archive strategy (``NonLinearSearch.checkpointer``)
  and ``ctx.resume`` its resume state (``NonLinearSearch.resume_state``) when the
  interrupted run's file exists, else ``None`` (phase A3, ``autofit.non_linear.checkpoint``);
  ``ctx.resume`` is the ``Checkpointer`` handle, not the loaded state, because some
  backends reopen their own native file (``Nautilus``);
- ``ctx.rng`` is the ``numpy.random.SeedSequence`` of the search-level ``seed`` (phase
  A4, ``autofit.non_linear.search.seed``): stream 0 is already spawned for the
  initializer, so ``ctx.rng.spawn(k)`` gives the backend streams 1..k.

``RawSamples`` and ``samples_from_raw`` are phase A3's samples adapter
(``autofit/non_linear/samples/adapter.py``), re-exported here for ``run(ctx)`` searches.
"""
from typing import Any, Iterator

import numpy as np

from autofit.non_linear.samples.adapter import RawSamples, samples_from_raw
from autofit.non_linear.search.seed import seed_sequence
from autofit.non_linear.status import status_notify


class UpdateSchedule:
    def __init__(self, search):
        """
        How many iterations a ``run(ctx)`` backend runs before its next ``ctx.update``,
        from the search's ``iterations_per_full_update``.

        The backend's total budget is a search setting (``nsteps``, ``n_like_max``, ...),
        so it is passed in: ``next_budget(done, total)`` and ``chunks(total)``.
        """
        self._search = search

    def next_budget(self, done: int, total: int) -> int:
        """
        The next chunk's length, given ``done`` of ``total`` iterations already run.
        """
        return self._search._steps_until_full_update(total - done)

    def chunks(self, total: int) -> Iterator[int]:
        """
        The chunk lengths that run ``total`` iterations, one update between chunks.
        """
        done = 0
        while done < total:
            budget = self.next_budget(done, total)
            yield budget
            done += budget


class FitContext:
    def __init__(
        self,
        search,
        model,
        analysis,
        fitness,
        pool,
        test_mode_level: int,
    ):
        """
        Everything one ``fit`` hands a ``run(ctx)`` backend (``docs/design/run_ctx.md``).

        Created by the bridge once per fit, after the test-mode bypass and the fail-fast
        gates; never stored on the search, never pickled, never copied with it; closed on
        every exit.
        """
        self._search = search
        self._analysis = analysis

        self.model = model
        self.paths = search.paths
        self.fitness = fitness
        self.test_mode_level = test_mode_level
        self.pool = pool
        # Stream 0 is the initializer's (`ctx.start_points`), spawned here so the
        # backend's `ctx.rng.spawn(k)` hands out streams 1..k (search-level seed, A4).
        self.rng = seed_sequence(search)
        self.rng.spawn(1)
        self.checkpointer = search.checkpointer
        resume_state = search.resume_state
        self.resume = (
            resume_state
            if resume_state is not None and resume_state.exists(search.paths)
            else None
        )
        self.schedule = UpdateSchedule(search)

    def objective(self, kind: str):
        """
        The objective of ``kind`` (``Fitness.objective``): what it returns, in which
        coordinates and for an invalid model are the search's declared
        ``objective_target`` and ``invalid_value``.
        """
        return self.fitness.objective(kind)

    def start_points(self, n: int):
        """
        ``n`` initial points from the search's initializer, with the start-point plot.

        Returns
        -------
        The ``(n, n_dim)`` parameters and the ``(n,)`` figures of merit, as arrays.
        """
        _, parameter_lists, figure_of_merit_list = self._search.start_points(
            model=self.model,
            fitness=self.fitness,
            n=n,
            plot=self._search._plots_start_point,
            test_mode_samples=self._search._start_points_in_test_mode,
        )
        return np.asarray(parameter_lists), np.asarray(figure_of_merit_list)

    def update(self, internal: Any):
        """
        Write the samples, visuals and summary of the run so far
        (``perform_update(during_analysis=True)``).

        The backend is paused for the duration of the call (the update is synchronous),
        so ``internal`` is read as it stands at this point of the run.

        The checkpoint is saved first: the search's resume state (``ctx.resume``'s
        strategy, ``NonLinearSearch.resume_state``) gets ``save(internal)``, so an
        interrupted run resumes from the last update. A native resume state (one the
        backend writes itself) keeps its no-op ``save``; a search without a resume
        state is not resumable and saves nothing until its archive is finalized at
        completion. ``finalize`` is never called here, since that is the completion step
        that may discard the resume state.
        """
        resume_state = self._search.resume_state
        if resume_state is not None:
            resume_state.save(self.paths, internal)

        # The fit's status record takes the search's `progress` and keeps `internal`
        # for a SIGUSR1 / SIGTERM checkpoint flush (`autofit.non_linear.status`).
        status_notify(self._search, "update", internal)

        return self._search.perform_update(
            model=self.model,
            analysis=self._analysis,
            during_analysis=True,
            fitness=self.fitness,
            search_internal=internal,
        )

    def close(self, failed: bool):
        """
        Clean up after ``run``, on every exit. The pools the context built are shut
        down on both paths (closed and joined on success, terminated and joined on
        failure; ``PoolFactory.close``). On failure the quick-update threads are also
        shut down and the compiled objectives released; on success the fitness is still
        needed for the final update, which shuts its threads down.
        """
        close_pools = getattr(self.pool, "close", None)
        if close_pools is not None:
            close_pools(failed=failed)

        if not failed:
            return

        shutdown = getattr(self.fitness, "shutdown_quick_update", None)
        if shutdown is not None:
            shutdown()

        objectives = getattr(self.fitness, "_objectives", None)
        if objectives is not None:
            objectives.clear()
