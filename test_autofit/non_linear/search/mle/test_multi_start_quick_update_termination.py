import importlib.util

import pytest

import autofit as af

# The end-to-end test below runs the JAX-native multi-start step loop, which needs jax + optax
# (both ship via the `[optional]` extras). The Python-version matrix installs autofit without
# those extras, so skip there rather than fail with `No module named 'jax'`.
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None
    or importlib.util.find_spec("optax") is None,
    reason="requires jax + optax (installed via the [optional] extras; absent on the matrix env)",
)


class _QuickUpdateRecordingAnalysis(af.Analysis):
    """
    A zero-centred Gaussian log-likelihood, written through ``self._xp`` so it
    is JAX-traceable (and therefore differentiable) when built with
    ``use_jax=True``.

    ``perform_quick_update`` records rather than raising ``NotImplementedError``,
    so the assertions below stay valid whether or not the search wires quick
    updates into its step loop (today it does not — see the ``EXEMPT`` entry in
    ``test_quick_update_wiring.py``): fired or not, a quick update must never
    END the search.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quick_update_count = 0

    def log_likelihood_function(self, instance):
        xp = self._xp
        values = xp.asarray([instance.one, instance.two])
        return -0.5 * xp.sum(values**2)

    def perform_quick_update(self, paths, instance):
        self.quick_update_count += 1


@requires_jax
def test__quick_update_mid_search_does_not_end_the_search():
    """
    Regression for PyAutoFit#1553: an ``imaging/start_here.py`` run using
    ``MultiStartProdigy`` with ``iterations_per_quick_update=50`` was reported
    as "stopping at 49 steps". The 49 was ``search.summary``'s
    ``Total Samples`` line (1 best + ``n_starts=48`` per-start final points),
    not a step count: the quick-update cadence must never terminate the step
    loop, whose only stop paths are the auto-convergence plateau (gated behind
    ``min_steps``) and the ``n_steps`` ceiling.

    Pinned end-to-end: a fit whose quick-update boundary (25) falls well
    inside its budget (60) runs every one of its 60 steps. The convergence
    check is turned off explicitly, so the ceiling is provably the ONLY thing
    that can end this run, in any environment.
    """
    model = af.Model(af.m.MockClassx2)
    model.one = af.UniformPrior(lower_limit=-5.0, upper_limit=5.0)
    model.two = af.UniformPrior(lower_limit=-5.0, upper_limit=5.0)

    search = af.MultiStartProdigy(
        n_starts=4,
        n_steps=60,
        iterations_per_quick_update=25,
        convergence=af.MultiStartGradientConvergence(check_for_convergence=False),
    )

    analysis = _QuickUpdateRecordingAnalysis(use_jax=True)

    result = search.fit(model=model, analysis=analysis)

    search_internal = result.search_internal

    assert int(search_internal["total_steps"]) == 60
    assert search_internal["stop_reason"] == "max_steps"

    # The cadence is LIVE for this family since the multi-start step loop
    # wires ``iterations_per_quick_update`` (PyAutoFit#1552): quick updates
    # fire at the cadence boundaries, and the two ceiling asserts above prove
    # firing mid-search does not end the search — the real exercise of "a
    # quick update mid-search does not end the search", where a silent no-op
    # (the PyAutoFit#1434 failure class) or a boundary-triggered stop both
    # fail loudly. This flipped the inertness tripwire the test originally
    # carried (``== 0``), exactly as that tripwire's comment demanded.
    assert analysis.quick_update_count > 0
