import numpy as np

from autofit.non_linear.test_mode import is_test_mode


class MultiStartGradientConvergence:
    def __init__(
        self,
        check_for_convergence: bool = True,
        window: int = 50,
        rtol: float = 1.0e-4,
        atol: float = 1.0e-3,
        min_steps: int = 100,
    ):
        """
        Settings for the auto-convergence (early-stopping) check of the multi-start
        gradient searches (``MultiStartAdam`` / ``MultiStartADABelief`` /
        ``MultiStartLion`` / ``MultiStartProdigy``).

        This mirrors the ``AutoCorrelationsSettings`` precedent that lets the
        ensemble MCMC samplers (Emcee / Zeus) terminate before their full step
        budget: the search still has a hard ``n_steps`` ceiling (it never runs
        forever), but when ``check_for_convergence`` is ``True`` it stops early once
        the global-best figure-of-merit has plateaued.

        The search minimises ``-2 * log_posterior`` (a chi-squared-like quantity),
        so the tracked global best figure-of-merit ``best_fom`` is monotonically
        non-increasing. Convergence is declared when the improvement over the
        trailing ``window`` of the best-fom history is within tolerance::

            improvement = fom_history[-window] - fom_history[-1]   # >= 0
            converged   = improvement <= atol + rtol * abs(fom_history[-window])

        evaluated only once at least ``max(min_steps, window)`` steps have been
        taken, so early transient plateaus (before the population has descended)
        cannot trigger a false stop.

        Scope: this is the parametric-source (MGE / Sersic) regime, where a
        global-best plateau genuinely means converged. It is deliberately **not**
        applied when ``resurrect=True`` (the pixelized regime), whose best-fom
        climbs in long plateaus punctuated by breakthrough jumps that plateau
        detection would false-stop on; there the search leans on the ``n_steps``
        ceiling instead.

        Parameters
        ----------
        check_for_convergence
            Whether the global-best figure-of-merit is checked to terminate the
            search early. If ``True`` the search may stop before ``n_steps`` has
            been reached; if ``False`` all ``n_steps`` steps are taken.
        window
            The number of trailing steps of the best-fom history over which the
            plateau (improvement within tolerance) is measured. A longer window
            requires the best-fom to be flat for longer before stopping, trading
            a few extra steps for robustness against noise.
        rtol
            The relative tolerance on the best-fom improvement over the window.
        atol
            The absolute tolerance on the best-fom improvement over the window.
        min_steps
            The minimum number of steps that must be taken before the convergence
            check is allowed to terminate the search, so the population has time to
            descend out of its broad starts before a plateau can stop it.
        """
        self.check_for_convergence = check_for_convergence
        self.window = window
        self.rtol = rtol
        self.atol = atol
        self.min_steps = min_steps

        if is_test_mode():
            self.window = 1
            self.min_steps = 1

    def check_if_converged(self, fom_history) -> bool:
        """
        Whether the multi-start gradient search has converged: the global-best
        figure-of-merit has plateaued over the trailing ``window``.

        Parameters
        ----------
        fom_history
            The history of the global best figure-of-merit, one entry per step
            (monotonically non-increasing). Convergence is only assessed once at
            least ``max(min_steps, window)`` entries are present.
        """
        if not self.check_for_convergence:
            return False

        fom_history = np.asarray(fom_history)

        if fom_history.size < max(self.min_steps, self.window):
            return False

        latest = fom_history[-1]
        past = fom_history[-self.window]

        # Both non-finite (no finite basin found yet) -> not converged; the plateau
        # of the actual descent is what we want to detect, not a flat +inf run.
        if not np.isfinite(latest) or not np.isfinite(past):
            return False

        improvement = past - latest

        return bool(improvement <= self.atol + self.rtol * abs(past))
