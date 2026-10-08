"""
The minimal ``run(ctx)`` bridge (search-extensibility phase A2, decision D7).

``docs/design/run_ctx.md`` (frozen) fixes the ``run(ctx)`` signature and the members of
``FitContext``. A search that implements ``run(ctx)`` (and ``raw_samples_from``) instead
of ``_fit`` is driven through a ``FitContext`` by ``NonLinearSearch._fit``, the bridge;
a search that overrides ``_fit`` keeps working unchanged. ``Drawer`` and ``Nautilus`` are
the first two searches on the bridge.

Members filled by later phases are present with their A2 value:

- ``ctx.resume`` is ``None`` and ``ctx.checkpointer`` is ``None`` (phase A3 fills both;
  until then a backend resumes from its own native checkpoint, as ``Nautilus`` does);
- ``ctx.rng`` is a ``numpy.random.SeedSequence`` from the search's ``seed`` attribute
  when it has one (phase A4 adds one search-level seed and feeds the initializer).

``RawSamples`` and ``samples_from_raw`` are phase A3's samples adapter
(``autofit/non_linear/samples/adapter.py``), re-exported here for ``run(ctx)`` searches.
"""
from typing import Any, Iterator

import numpy as np

from autofit.non_linear.samples.adapter import RawSamples, samples_from_raw


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
        self.rng = np.random.SeedSequence(getattr(search, "seed", None))
        self.resume = None
        self.checkpointer = None
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
        )
        return np.asarray(parameter_lists), np.asarray(figure_of_merit_list)

    def update(self, internal: Any):
        """
        Write the samples, visuals and summary of the run so far
        (``perform_update(during_analysis=True)``).

        The backend is paused for the duration of the call (the update is synchronous),
        so ``internal`` is read as it stands at this point of the run.
        """
        return self._search.perform_update(
            model=self.model,
            analysis=self._analysis,
            during_analysis=True,
            fitness=self.fitness,
            search_internal=internal,
        )

    def close(self, failed: bool):
        """
        Clean up after ``run``. On failure: close the pools the context built, shut down
        the quick-update threads and release the compiled objectives. On success the
        fitness is still needed for the final update, which shuts its threads down.
        """
        if not failed:
            return

        self.pool.close()

        shutdown = getattr(self.fitness, "shutdown_quick_update", None)
        if shutdown is not None:
            shutdown()

        objectives = getattr(self.fitness, "_objectives", None)
        if objectives is not None:
            objectives.clear()
