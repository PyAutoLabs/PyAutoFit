"""Real scipy checkpoints survive interruption, including exhausted budgets."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import autofit as af
from autonerves import conf
from autofit.non_linear.paths.directory import DirectoryPaths


class QuadraticAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -0.5 * ((instance.x - 2) ** 2 + 10 * (instance.y + 1) ** 2)


@pytest.mark.parametrize("search_cls", [af.BFGS, af.LBFGS])
def test_interrupted_fit_resumes_from_checkpoint(search_cls, tmp_path, monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    monkeypatch.setattr(conf.instance, "output_path", str(tmp_path))
    model = af.Collection(
        x=af.GaussianPrior(mean=0, sigma=10),
        y=af.GaussianPrior(mean=0, sigma=10),
    )
    analysis = QuadraticAnalysis()

    def make_search(name):
        return search_cls(
            name=name, seed=42, maxiter=80, iterations_per_full_update=2,
        )

    save = DirectoryPaths.save_search_internal
    saved = []

    def interrupt_after_checkpoint(paths, obj):
        save(paths, obj)
        saved.append(obj)
        # Interrupt after multiple chunks to distinguish cumulative from last-chunk nit.
        if len(saved) == 2:
            raise KeyboardInterrupt("simulated interruption after durable checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(DirectoryPaths, "save_search_internal", interrupt_after_checkpoint)
        with pytest.raises(KeyboardInterrupt):
            make_search("resume").fit(model=model, analysis=analysis)

    assert saved[-1].total_iterations == sum(result.nit for result in saved)
    checkpoint_x = saved[-1].x.copy()
    from scipy import optimize
    minimize = optimize.minimize
    starts = []

    def recording_minimize(*args, **kwargs):
        starts.append(np.array(kwargs["x0"]))
        return minimize(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(optimize, "minimize", recording_minimize)
        resumed = make_search("resume").fit(model=model, analysis=analysis)
    uninterrupted = make_search("uninterrupted").fit(model=model, analysis=analysis)
    np.testing.assert_allclose(starts[0], checkpoint_x)
    np.testing.assert_allclose(resumed.search_internal.x, uninterrupted.search_internal.x)
    assert resumed.search_internal.total_iterations == uninterrupted.search_internal.total_iterations
    assert resumed.samples.total_iterations == resumed.search_internal.total_iterations


@pytest.mark.parametrize("search_cls", [af.BFGS, af.LBFGS])
@pytest.mark.parametrize("legacy", [False, True])
def test_exhausted_checkpoint_returns_samples_without_minimizing(search_cls, legacy, monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    model = af.Collection(x=af.UniformPrior(lower_limit=0, upper_limit=1))
    saved = OptimizeResult(x=np.array([0.5]), nit=5, log_posterior_list=-1.0)
    if not legacy:
        saved.total_iterations = 10
    search = search_cls(maxiter=5)
    monkeypatch.setattr(type(search), "should_plot_start_point", property(lambda _: False))
    ctx = SimpleNamespace(
        model=model, fitness=None,
        paths=SimpleNamespace(load_search_internal=lambda: saved),
    )
    assert search.run(ctx) is saved
    raw = search.raw_samples_from(model, saved)
    assert raw.parameters == [[0.5]]
    assert raw.info["total_iterations"] == (5 if legacy else 10)
